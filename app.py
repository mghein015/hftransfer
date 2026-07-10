import os, re, shutil, tempfile, uuid, zipfile, asyncio, urllib.parse, time, requests, mimetypes, json, fnmatch
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx, uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from supabase import create_client, Client
from huggingface_hub import (
    HfApi, HfFileSystem, hf_hub_download,
    CommitOperationCopy, CommitOperationDelete, RepoFile,
    get_bucket_file_metadata
)
from huggingface_hub.utils import RepositoryNotFoundError

# Google Drive
from google.oauth2 import service_account as gsa
from google.oauth2.credentials import Credentials as OAuthCredentials
from google.auth.transport.requests import Request as GoogleAuthRequest
from googleapiclient.discovery import build as gdrive_build
from googleapiclient.http import MediaFileUpload as GDriveMediaFileUpload
from googleapiclient.errors import HttpError as GDriveHttpError

# ─── CONFIG ───
MASTER_HF_TOKEN = os.environ.get("HF_TOKEN")
MASTER_HF_REPO  = os.environ.get("HF_REPO_ID")
GDRIVE_SA_JSON  = os.environ.get("GDRIVE_SA_JSON")   # full JSON key as env var string
GDRIVE_SCOPES   = ["https://www.googleapis.com/auth/drive"]

# OAuth2 client credentials (Desktop app type) — set via env vars or upload JSON
GDRIVE_OAUTH_CLIENT_ID     = os.environ.get("GDRIVE_OAUTH_CLIENT_ID", "")
GDRIVE_OAUTH_CLIENT_SECRET = os.environ.get("GDRIVE_OAUTH_CLIENT_SECRET", "")
GDRIVE_OAUTH_REDIRECT_URI  = os.environ.get("GDRIVE_OAUTH_REDIRECT_URI", "http://localhost")

MAX_CONCURRENT_JOBS = 5
CHUNK_1MB  = 1 << 20
CHUNK_8MB  = 8 << 20
CHUNK_50MB = 50 << 20
BATCH_OPS  = 90

sb_url, sb_key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Optional[Client] = create_client(sb_url, sb_key) if sb_url and sb_key else None
semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
_http: Optional[httpx.AsyncClient] = None

ORPHAN_STATUSES = [
    "processing", "downloading", "uploading", "queued_locally",
    "cleaning_up", "extracting", "initializing",
    "downloading_archive", "uploading_extracted",
]

# ─── HELPERS ───
def _hf_url(repo_type: str, repo_id: str, path: str) -> str:
    prefix = {"model": "", "bucket": "buckets/"}.get(repo_type, f"{repo_type}s/")
    return f"hf://{prefix}{repo_id}/{path}"

def _hf_status(e: Exception) -> int:
    return getattr(getattr(e, "response", None), "status_code", 400)

def _encode_filename(name: str) -> str:
    return urllib.parse.quote(name)

def _fs_escape(path: str) -> str:
    """
    Escape fsspec/glob-magic characters (* ? [ ]) in a path so HfFileSystem
    treats them as literal filename characters instead of a glob pattern.
    Without this, a real file like 'photo[1].png' gets parsed as a pattern
    (character class '[1]') and fails to resolve — this is the standard
    fsspec-wide gotcha (same issue in s3fs, gcsfs, etc.).
    """
    import glob as _glob_mod
    return _glob_mod.escape(path)

def _safe_name(name: str, prefix: str = "") -> str:
    uid = uuid.uuid4().hex[:8]
    safe = (name or "upload").replace(" ", "_")
    return f"{prefix}{uid}_{safe}"

async def _sb_query(fn, retries=3):
    if not supabase:
        return None
    transient = ("ConnectionTerminated", "Server disconnected", "RemoteProtocolError", "Connection reset")
    for attempt in range(retries):
        try:
            return await asyncio.to_thread(fn)
        except Exception as e:
            if attempt == retries - 1 or not any(t in str(e) or t in type(e).__name__ for t in transient):
                raise
            await asyncio.sleep(1)

# ─── GDRIVE HELPERS ──────────────────────────────────────────────────────────

def _get_gdrive_service(
    sa_json: Optional[str] = None,
    oauth_refresh_token: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
):
    """
    Build an authenticated Drive v3 service.

    Auth priority:
      1. oauth_refresh_token  — personal Google account (has real storage quota)
      2. sa_json arg          — service account JSON string or file path
      3. GDRIVE_SA_JSON env   — fallback service account

    For OAuth: requires client_id + client_secret (from env vars or passed in).
    The refresh_token is obtained once via /api/gdrive/oauth/url + /callback.
    """
    # ── OAuth2 personal account ──────────────────────────────────────────────
    if oauth_refresh_token:
        cid = client_id or GDRIVE_OAUTH_CLIENT_ID
        csec = client_secret or GDRIVE_OAUTH_CLIENT_SECRET
        if not cid or not csec:
            raise ValueError(
                "OAuth client_id and client_secret are required. "
                "Set GDRIVE_OAUTH_CLIENT_ID / GDRIVE_OAUTH_CLIENT_SECRET env vars."
            )
        creds = OAuthCredentials(
            token=None,
            refresh_token=oauth_refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=cid,
            client_secret=csec,
            scopes=GDRIVE_SCOPES,
        )
        # Refresh to get a valid access token
        creds.refresh(GoogleAuthRequest())
        return gdrive_build("drive", "v3", credentials=creds, cache_discovery=False)

    # ── Service account ──────────────────────────────────────────────────────
    raw = sa_json or GDRIVE_SA_JSON
    if not raw:
        raise ValueError(
            "No GDrive credentials provided. Either pass oauth_refresh_token "
            "or sa_json, or set GDRIVE_SA_JSON / GDRIVE_OAUTH_* env vars."
        )
    if raw.strip().endswith(".json") and os.path.exists(raw.strip()):
        with open(raw.strip()) as f:
            info = json.load(f)
    else:
        info = json.loads(raw)

    creds = gsa.Credentials.from_service_account_info(info, scopes=GDRIVE_SCOPES)
    return gdrive_build("drive", "v3", credentials=creds, cache_discovery=False)


def _build_gdrive_auth_url(
    redirect_uri: str,
    code_challenge: Optional[str] = None,
    code_challenge_method: Optional[str] = None,
    state: Optional[str] = None,
) -> str:
    """
    Build a Google OAuth2 authorisation URL directly via urllib — bypasses
    google-auth-oauthlib entirely so PKCE params are never dropped or mangled.
    """
    cid = GDRIVE_OAUTH_CLIENT_ID
    if not cid:
        raise ValueError(
            "GDRIVE_OAUTH_CLIENT_ID must be set. Upload your client_secret JSON first."
        )

    params: dict = {
        "client_id":     cid,
        "redirect_uri":  redirect_uri,
        "response_type": "code",
        "scope":         " ".join(GDRIVE_SCOPES),
        "access_type":   "offline",
        "prompt":        "consent",
    }
    if code_challenge:
        params["code_challenge"]        = code_challenge
        params["code_challenge_method"] = code_challenge_method or "S256"
    if state:
        params["state"] = state

    return "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)


def _gdrive_escape_q(value: str) -> str:
    """Escape a value for use inside a Drive API query string literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'")

def _gdrive_ensure_folder(service, name: str, parent_id: str) -> str:
    """Get-or-create a folder by name under parent_id. Returns Drive folder ID."""
    q = (
        f"name='{_gdrive_escape_q(name)}' and '{parent_id}' in parents "
        f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    res = service.files().list(
        q=q, fields="files(id)", pageSize=1,
        supportsAllDrives=True, includeItemsFromAllDrives=True,
    ).execute()
    if res["files"]:
        return res["files"][0]["id"]
    meta = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    return service.files().create(
        body=meta, fields="id", supportsAllDrives=True,
    ).execute()["id"]


def _gdrive_file_exists(service, name: str, parent_id: str) -> Optional[str]:
    """Return Drive file ID if a non-trashed file with this name exists in parent, else None."""
    q = f"name='{_gdrive_escape_q(name)}' and '{parent_id}' in parents and trashed=false"
    res = service.files().list(
        q=q, fields="files(id)", pageSize=1,
        supportsAllDrives=True, includeItemsFromAllDrives=True,
    ).execute()
    return res["files"][0]["id"] if res["files"] else None


def _gdrive_resolve_parent(service, rel_path: str, root_folder_id: str) -> tuple:
    """
    For a relative path like 'subdir/weights/model.safetensors',
    ensure all intermediate Drive folders exist and return (parent_folder_id, filename).
    """
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p]
    if not parts:
        raise ValueError(f"rel_path '{rel_path}' contains no filename component")
    filename = parts[-1]
    current = root_folder_id
    for part in parts[:-1]:
        current = _gdrive_ensure_folder(service, part, current)
    return current, filename


def _gdrive_upload_file_resumable(
    service,
    local_path: str,
    filename: str,
    parent_id: str,
    overwrite: bool = False,
) -> str:
    """
    Upload a single file to Drive using resumable upload (safe for large files).
    If overwrite=True and the file already exists, it will be replaced.
    Returns the Drive file ID.
    """
    mime, _ = mimetypes.guess_type(local_path)
    mime = mime or "application/octet-stream"

    existing_id = _gdrive_file_exists(service, filename, parent_id)

    media = GDriveMediaFileUpload(
        local_path, mimetype=mime,
        resumable=True, chunksize=CHUNK_50MB,
    )

    if existing_id:
        if overwrite:
            # Replace content in-place (keeps same Drive file ID)
            file = service.files().update(
                fileId=existing_id,
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            ).execute()
            return file["id"]
        else:
            # File already exists and overwrite=False → skip, return existing ID
            return existing_id

    # File does not exist yet → create it
    meta = {"name": filename, "parents": [parent_id]}
    file = service.files().create(
        body=meta, media_body=media, fields="id",
        supportsAllDrives=True,
    ).execute()
    return file["id"]


# ─── MODELS ───
class OAuthRequest(BaseModel):
    code: str
    redirect_uri: str

class ArchiveRequest(BaseModel):
    files: List[str]
    archive_name: str

class GDriveTransferRequest(BaseModel):
    # Source (HF)
    repo_id: str                            # e.g. "meta-llama/Llama-3.2-1B"
    repo_type: str = "model"               # model | dataset | space | bucket
    hf_token: Optional[str] = None         # required for private repos
    path_filter: Optional[str] = None      # only transfer files under this prefix
    ignore_patterns: Optional[List[str]] = None  # e.g. ["*.h5", "*.msgpack"]

    # Destination (GDrive)
    gdrive_folder_id: str                  # target Drive folder ID

    # Auth — provide ONE of these (oauth_refresh_token preferred; sa_json as fallback)
    oauth_refresh_token: Optional[str] = None  # personal Google account token
    oauth_client_id: Optional[str] = None      # override env var if needed
    oauth_client_secret: Optional[str] = None  # override env var if needed
    sa_json: Optional[str] = None              # service account JSON (has no storage quota)

    # Behaviour
    overwrite: bool = False                # replace existing files on Drive
    user_id: Optional[str] = None         # for Supabase job tracking


# ─── LIFESPAN ───
@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _http
    _http = httpx.AsyncClient(
        follow_redirects=True, timeout=httpx.Timeout(None),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
    )
    if supabase:
        try:
            await _sb_query(lambda: (
                supabase.table("transfer_jobs")
                .update({"status": "pending", "progress": 0})
                .in_("status", ORPHAN_STATUSES).execute()
            ))
            print("Startup: reset orphaned jobs.")
        except Exception as e:
            print(f"Orphan reset failed: {e}")

    worker = asyncio.create_task(worker_loop())
    yield
    worker.cancel()
    if _http and not _http.is_closed:
        await _http.aclose()


app = FastAPI(title="Hugging Face Manager API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


def http() -> httpx.AsyncClient:
    global _http
    if not _http or _http.is_closed:
        _http = httpx.AsyncClient(
            follow_redirects=True, timeout=httpx.Timeout(None),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
    return _http

# ─── STREAMING ───
def _bucket_stream_sync(repo_id: str, path: str, token: str, filename: str, range_hdr: Optional[str], inline: bool = False):
    fs = HfFileSystem(token=token)
    url = f"hf://buckets/{repo_id}/{_fs_escape(path.lstrip('/'))}"
    if not fs.exists(url):
        raise FileNotFoundError("File not found in bucket.")

    size = fs.info(url).get("size", 0)
    start, end, status = 0, size - 1, 200

    if range_hdr:
        m = re.match(r"bytes=(\d+)-(\d*)", range_hdr)
        if m:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else end
            status = 206

    length = end - start + 1

    def iterate():
        with fs.open(url, "rb") as f:
            f.seek(start)
            rem = length
            while rem > 0:
                chunk = f.read(min(CHUNK_1MB, rem))
                if not chunk:
                    break
                yield chunk
                rem -= len(chunk)

    disposition = "inline" if inline else "attachment"
    headers = {
        "Content-Disposition": f"{disposition}; filename*=UTF-8''{_encode_filename(filename)}",
        "Accept-Ranges": "bytes", "Content-Length": str(length),
    }
    if status == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return status, headers, iterate


async def _stream_bucket(repo_id: str, path: str, token: str, filename: str, request: Request, inline: bool = False):
    try:
        status, headers, it = await asyncio.to_thread(
            _bucket_stream_sync, repo_id, path, token, filename, request.headers.get("Range"), inline
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

    mime_type, _ = mimetypes.guess_type(filename)
    media_type = mime_type or "application/octet-stream"
    return StreamingResponse(it(), status_code=status, headers=headers, media_type=media_type)


# ─── DOWNLOAD ───
@app.get("/api/drive/download")
async def download_native(request: Request, path: str, filename: str, inline: bool = False):
    if not MASTER_HF_TOKEN or not MASTER_HF_REPO:
        raise HTTPException(500, "Missing master token/repo config.")
    return await _stream_bucket(MASTER_HF_REPO, path, MASTER_HF_TOKEN, filename, request, inline)


@app.get("/api/drive/download-external")
async def download_external(
    request: Request, path: str, drive_id: Optional[str] = None,
    repo_id: Optional[str] = None, token: Optional[str] = None, repo_type: str = "bucket",
    inline: bool = False
):
    if drive_id and supabase:
        res = await _sb_query(lambda: (
            supabase.table("connected_drives").select("*").eq("id", drive_id).execute()
        ))
        if not res or not res.data:
            raise HTTPException(404, "Shared drive not found.")
        d = res.data[0]
        repo_id, token, repo_type = d.get("hf_username"), d.get("hf_token"), d.get("repo_type", "bucket")
    elif not repo_id or not token:
        raise HTTPException(400, "Provide drive_id or both repo_id and token.")

    filename = path.split("/")[-1]

    if repo_type == "bucket":
        return await _stream_bucket(repo_id, path, token, filename, request, inline)

    clean = urllib.parse.quote(path.lstrip("/"), safe="/")
    type_map = {"dataset": "datasets/", "space": "spaces/"}
    prefix = type_map.get(repo_type, "")
    url = f"https://huggingface.co/{prefix}{repo_id}/resolve/main/{clean}?download=true"

    req_headers = {}
    if token:
        req_headers["Authorization"] = f"Bearer {token}"
    rng = request.headers.get("Range")
    if rng:
        req_headers["Range"] = rng

    client = http()
    hf_res = await client.send(client.build_request("GET", url, headers=req_headers), stream=True)

    if hf_res.status_code not in (200, 206):
        await hf_res.aclose()
        raise HTTPException(hf_res.status_code, "File not found. Folder downloads unsupported.")

    mime_type, _ = mimetypes.guess_type(filename)
    fallback_media_type = mime_type or hf_res.headers.get("content-type", "application/octet-stream")
    disposition = "inline" if inline else "attachment"

    resp_headers = {
        "Content-Disposition": f"{disposition}; filename*=UTF-8''{_encode_filename(filename)}",
        "Accept-Ranges": "bytes",
        **{h: hf_res.headers[h] for h in ("Content-Length", "Content-Range") if h in hf_res.headers},
    }

    async def stream():
        try:
            async for chunk in hf_res.aiter_bytes(CHUNK_8MB):
                yield chunk
        finally:
            await hf_res.aclose()

    return StreamingResponse(
        stream(), status_code=hf_res.status_code,
        headers=resp_headers, media_type=fallback_media_type,
    )

# ─── UPLOAD ───
@app.post("/api/drive/upload")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    if not MASTER_HF_TOKEN or not MASTER_HF_REPO:
        raise HTTPException(500, "Server config error.")

    target = f"private_files/{user_id}/{_safe_name(file.filename)}"
    spooled = file.file

    def _write():
        fs = HfFileSystem(token=MASTER_HF_TOKEN)
        written = 0
        with fs.open(f"hf://buckets/{MASTER_HF_REPO}/{target}", "wb") as dest:
            while chunk := spooled.read(CHUNK_1MB):
                dest.write(chunk)
                written += len(chunk)
        return written

    try:
        size = await asyncio.to_thread(_write)
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"status": "success", "storage_path": target, "size_bytes": size, "mime_type": file.content_type}


@app.post("/api/drive/remote-upload")
async def remote_upload(url: str = Form(...), user_id: str = Form(...), filename: Optional[str] = Form(None)):
    if not MASTER_HF_TOKEN or not MASTER_HF_REPO:
        raise HTTPException(500, "Server config error.")

    parsed = urllib.parse.urlparse(url)
    orig = filename or os.path.basename(parsed.path) or "remote_file.bin"
    safe = urllib.parse.unquote(orig).replace(" ", "_")
    target = f"private_files/{user_id}/{uuid.uuid4().hex[:8]}_{safe}"

    def _fetch():
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            ct = r.headers.get("content-type", "application/octet-stream")
            fs = HfFileSystem(token=MASTER_HF_TOKEN)
            written = 0
            with fs.open(f"hf://buckets/{MASTER_HF_REPO}/{target}", "wb") as dest:
                for chunk in r.iter_content(CHUNK_1MB):
                    if chunk:
                        dest.write(chunk)
                        written += len(chunk)
        return {"name": urllib.parse.unquote(orig), "storage_path": target, "size_bytes": written, "mime_type": ct}

    try:
        return await asyncio.to_thread(_fetch)
    except Exception as e:
        raise HTTPException(400, f"Remote upload failed: {e}")

# ─── OAUTH ───
@app.post("/api/oauth/token")
async def oauth_exchange(req: OAuthRequest):
    cid, csec = os.environ.get("HF_CLIENT_ID"), os.environ.get("HF_CLIENT_SECRET")
    if not cid or not csec:
        raise HTTPException(500, "OAuth not configured.")
    r = await http().post("https://huggingface.co/oauth/token", data={
        "client_id": cid, "client_secret": csec,
        "grant_type": "authorization_code", "code": req.code, "redirect_uri": req.redirect_uri,
    })
    if r.status_code != 200:
        raise HTTPException(400, "OAuth exchange failed.")
    return r.json()

# ─── HF MANAGER ───
class HFManager:
    __slots__ = ("src_token", "dst_token", "_src_api", "_dst_api", "_src_fs", "_dst_fs")

    def __init__(self, token: str = None, src_token: str = None, dst_token: str = None):
        self.src_token = src_token or token
        self.dst_token = dst_token or token
        self._src_api = self._dst_api = self._src_fs = self._dst_fs = None

    @property
    def src_api(self) -> HfApi:
        if not self._src_api:
            self._src_api = HfApi(token=self.src_token)
        return self._src_api

    @property
    def dst_api(self) -> HfApi:
        if not self._dst_api:
            self._dst_api = HfApi(token=self.dst_token)
        return self._dst_api

    @property
    def src_fs(self) -> HfFileSystem:
        if not self._src_fs:
            self._src_fs = HfFileSystem(token=self.src_token, use_listings_cache=False)
        return self._src_fs

    @property
    def dst_fs(self) -> HfFileSystem:
        if not self._dst_fs:
            self._dst_fs = HfFileSystem(token=self.dst_token, use_listings_cache=False)
        return self._dst_fs

    def _retry_db(self, fn):
        for attempt in range(3):
            try:
                return fn()
            except Exception as e:
                if attempt == 2 or "ConnectionTerminated" not in str(e):
                    raise
                time.sleep(1)

    def _db(self, job_id: str, **kw):
        if supabase and job_id:
            self._retry_db(lambda: supabase.table("transfer_jobs").update(kw).eq("id", job_id).execute())

    def _cancelled(self, job_id: str):
        if supabase and job_id:
            r = self._retry_db(lambda: supabase.table("transfer_jobs").select("status").eq("id", job_id).execute())
            if r and r.data and r.data[0]["status"] == "cancelled":
                raise Exception("Job cancelled by user")

    def duplicate_repo(self, from_id, to_id, repo_type="model"):
        return self.src_api.duplicate_repo(from_id=from_id, to_id=to_id, repo_type=repo_type)

    def rename_repo(self, old_id, new_id, repo_type="model"):
        return self.src_api.move_repo(from_id=old_id, to_id=new_id, repo_type=repo_type)

    def delete_repo(self, repo_id, repo_type="model"):
        return self.src_api.delete_repo(repo_id=repo_id, repo_type=repo_type)

    def list_files(self, repo_id: str, repo_type: str = "model") -> List[dict]:
        if repo_type == "bucket":
            try:
                self.src_fs.invalidate_cache(f"hf://buckets/{repo_id}")
                paths = self.src_fs.find(f"hf://buckets/{repo_id}", detail=True)
            except FileNotFoundError:
                return []
            result = []
            prefix = f"buckets/{repo_id}/"
            for p in paths.values():
                if p.get("type") != "directory":
                    name = p["name"]
                    if name.startswith(prefix):
                        name = name[len(prefix):]
                    result.append({"path": name, "size": p.get("size", 0)})
            return result

        return [
            {"path": f.path, "size": getattr(f, "size", 0)}
            for f in self.src_api.list_repo_tree(repo_id=repo_id, repo_type=repo_type, recursive=True)
            if isinstance(f, RepoFile)
        ]

    def _get_folder_files(self, repo_id: str, repo_type: str, folder: str) -> List[str]:
        prefix = folder.rstrip("/") + "/"
        if repo_type == "bucket":
            paths = self.src_fs.find(_hf_url("bucket", repo_id, _fs_escape(folder)), detail=False)
            bucket_prefix = f"buckets/{repo_id}/"
            return [
                (p[len(bucket_prefix):] if p.startswith(bucket_prefix) else p.split("/")[-1])
                for p in paths if not p.endswith("/")
            ]
        tree = self.src_api.list_repo_tree(repo_id=repo_id, repo_type=repo_type, recursive=True)
        return [f.path for f in tree if f.path.startswith(prefix) and isinstance(f, RepoFile)]

    def _batch_commit(self, repo_id: str, repo_type: str, ops: list, msg_prefix: str):
        for i in range(0, len(ops), BATCH_OPS):
            self.src_api.create_commit(
                repo_id=repo_id, repo_type=repo_type, operations=ops[i:i + BATCH_OPS],
                commit_message=f"{msg_prefix} part {i // BATCH_OPS + 1}",
            )

    def delete_file(self, repo_id: str, path_in_repo: str, repo_type: str = "model"):
        path = path_in_repo.rstrip("/")
        if repo_type == "bucket":
            try:
                self.src_fs.rm(f"hf://buckets/{repo_id}/{path}", recursive=True)
            except Exception as e:
                if "404" not in str(e) and "not found" not in str(e).lower():
                    raise
            return {"message": "Deleted"}

        try:
            return self.src_api.delete_file(path_in_repo=path, repo_id=repo_id, repo_type=repo_type)
        except Exception as e:
            if "404" not in str(e).lower() and "not found" not in str(e).lower():
                raise
            children = self._get_folder_files(repo_id, repo_type, path)
            if children:
                self._batch_commit(repo_id, repo_type,
                                   [CommitOperationDelete(path_in_repo=f) for f in children],
                                   f"Delete folder {path}")
            return {"message": "Deleted"}

    def move_file(self, repo_id: str, old_path: str, new_path: str, repo_type: str = "model"):
        old, new = old_path.rstrip("/"), new_path.rstrip("/")
        if repo_type == "bucket":
            self.src_fs.mv(f"hf://buckets/{repo_id}/{old}", f"hf://buckets/{repo_id}/{new}", recursive=True)
            return {"message": "Moved"}

        try:
            return self.src_api.create_commit(
                repo_id=repo_id, repo_type=repo_type,
                operations=[
                    CommitOperationCopy(src_path_in_repo=old, path_in_repo=new),
                    CommitOperationDelete(path_in_repo=old),
                ],
                commit_message=f"Move {old} → {new}",
            )
        except Exception as e:
            if "404" not in str(e).lower() and "not found" not in str(e).lower():
                raise
            prefix = old + "/"
            children = self._get_folder_files(repo_id, repo_type, old)
            if not children:
                raise ValueError(f"Source '{old}' not found.")
            ops = []
            for f in children:
                rel = f[len(prefix):]
                ops += [
                    CommitOperationCopy(src_path_in_repo=f, path_in_repo=f"{new}/{rel}"),
                    CommitOperationDelete(path_in_repo=f),
                ]
            self._batch_commit(repo_id, repo_type, ops, f"Move folder {old} → {new}")
            return {"message": f"Moved folder {old}"}

    def archive_files(self, repo_id: str, repo_type: str, files: List[str], archive_name: str):
        low = archive_name.lower()
        if low.endswith(".rar"):
            raise Exception("RAR creation unsupported. Use .zip.")
        if not low.endswith(".zip"):
            raise Exception("Only .zip supported.")

        tmp = tempfile.mkdtemp()
        try:
            expanded = []
            if repo_type == "bucket":
                for f in files:
                    cf = f.rstrip("/")
                    try:
                        info = self.src_fs.info(_hf_url("bucket", repo_id, cf))
                        if info.get("type") == "directory":
                            expanded.extend(self._get_folder_files(repo_id, "bucket", cf))
                        else:
                            expanded.append(cf)
                    except Exception:
                        expanded.append(cf)
            else:
                tree = None
                for f in files:
                    cf = f.rstrip("/")
                    if tree is None:
                        tree = list(self.src_api.list_repo_tree(
                            repo_id=repo_id, repo_type=repo_type, recursive=True))
                    if any(x.path == cf and isinstance(x, RepoFile) for x in tree):
                        expanded.append(cf)
                    else:
                        children = [x.path for x in tree if x.path.startswith(cf + "/") and isinstance(x, RepoFile)]
                        expanded.extend(children or [cf])

            arc_path = os.path.join(tmp, archive_name)
            with zipfile.ZipFile(arc_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for fp in expanded:
                    if repo_type == "bucket":
                        local = os.path.join(tmp, f"dl_{os.path.basename(fp)}")
                        with self.src_fs.open(_hf_url("bucket", repo_id, fp), "rb") as f_in:
                            with open(local, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)
                    else:
                        local = hf_hub_download(
                            repo_id=repo_id, filename=fp, repo_type=repo_type,
                            token=self.src_token, cache_dir=tmp,
                        )
                    zf.write(local, arcname=fp)

            if repo_type == "bucket":
                self.dst_fs.put_file(arc_path, _hf_url("bucket", repo_id, archive_name))
            else:
                self.dst_api.upload_file(
                    path_or_fileobj=arc_path, path_in_repo=archive_name,
                    repo_id=repo_id, repo_type=repo_type,
                    commit_message=f"Created archive {archive_name}",
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ─── GDRIVE TRANSFER (Option B: one file at a time) ──────────────────────

    def transfer_to_gdrive_sync(self, job_id: Optional[str], req: "GDriveTransferRequest") -> dict:
        """
        Transfer HF repo/bucket → Google Drive one file at a time (Option B).

        Each file is:
          1. Downloaded to a temp file
          2. Uploaded to Drive via resumable upload (50 MB chunks)
          3. Deleted from disk immediately

        Max disk usage at any point = size of the single largest file.
        Directory structure is preserved on Drive.
        """
        self._db(job_id, status="initializing", progress=2)

        # ── 1. Authenticate GDrive ──────────────────────────────────────────
        try:
            service = _get_gdrive_service(
                sa_json=req.sa_json,
                oauth_refresh_token=getattr(req, "oauth_refresh_token", None),
                client_id=getattr(req, "oauth_client_id", None),
                client_secret=getattr(req, "oauth_client_secret", None),
            )
        except Exception as e:
            raise Exception(f"GDrive auth failed: {e}")

        # ── 2. List all files in the HF repo/bucket ─────────────────────────
        self._db(job_id, status="initializing", progress=5)
        all_files = self.list_files(req.repo_id, req.repo_type)

        # ── 3. Apply path_filter ─────────────────────────────────────────────
        if req.path_filter and req.path_filter.strip() not in ("", "*"):
            prefix = req.path_filter.rstrip("/") + "/"
            all_files = [
                f for f in all_files
                if f["path"].startswith(prefix) or f["path"] == req.path_filter.rstrip("/")
            ]

        # ── 4. Apply ignore_patterns ─────────────────────────────────────────
        if req.ignore_patterns:
            all_files = [
                f for f in all_files
                if not any(fnmatch.fnmatch(f["path"], pat) for pat in req.ignore_patterns)
            ]

        total = len(all_files)
        if total == 0:
            return {"transferred": 0, "skipped": 0, "errors": [], "total": 0}

        self._db(job_id, status="uploading", progress=8,
                 error_message=f"Found {total} files to transfer")

        transferred, skipped = 0, 0
        errors: List[dict] = []
        tmp_dir = tempfile.mkdtemp()

        try:
            for idx, file_info in enumerate(all_files):
                rel_path = file_info["path"]
                self._cancelled(job_id)

                # Progress: 10 → 98 across all files
                pct = 10 + int((idx / total) * 88)
                self._db(
                    job_id, progress=pct, status="uploading",
                    error_message=f"[{idx + 1}/{total}] {rel_path}",
                )

                local_path = None
                try:
                    # ── Download single file ──────────────────────────────
                    if req.repo_type == "bucket":
                        local_path = os.path.join(tmp_dir, f"_dl_{uuid.uuid4().hex[:6]}_{os.path.basename(rel_path)}")
                        
                        # Escape glob-magic chars (* ? [ ]) so real filenames like
                        # 'photo[1].png' aren't parsed as glob patterns by fsspec.
                        with self.src_fs.open(f"hf://buckets/{req.repo_id}/{_fs_escape(rel_path)}", "rb") as f_in:
                            with open(local_path, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)
                    else:
                        local_path = hf_hub_download(
                            repo_id=req.repo_id,
                            filename=rel_path,
                            repo_type=req.repo_type,
                            token=self.src_token,
                            local_dir=tmp_dir,
                            local_dir_use_symlinks=False,  # always write real file, never a symlink
                        )

                    # ── Verify the file actually landed on disk ───────────
                    # Do this BEFORE touching Drive so we never create empty
                    # folders on Drive for files that failed to download.
                    if not local_path or not os.path.isfile(local_path):
                        raise FileNotFoundError(
                            f"Download produced no file on disk for '{rel_path}' "
                            f"(expected at {local_path})"
                        )

                    # ── Resolve Drive parent folder ───────────────────────
                    # Only reach here once the local file is confirmed present.
                    drive_rel = rel_path
                    if req.path_filter and req.path_filter.strip() not in ("", "*"):
                        prefix = req.path_filter.rstrip("/") + "/"
                        if drive_rel.startswith(prefix):
                            drive_rel = drive_rel[len(prefix):]

                    parent_id, filename = _gdrive_resolve_parent(
                        service, drive_rel, req.gdrive_folder_id
                    )

                    # ── Upload to Drive (resumable) ───────────────────────
                    _gdrive_upload_file_resumable(
                        service, local_path, filename,
                        parent_id, overwrite=req.overwrite,
                    )
                    transferred += 1

                except Exception as e:
                    errors.append({"path": rel_path, "error": str(e)})
                    skipped += 1
                    print(f"GDrive transfer error [{rel_path}]: {e}")

                finally:
                    # ── Free disk immediately (Option B key behaviour) ─────
                    if local_path and os.path.exists(local_path):
                        try:
                            os.remove(local_path)
                        except OSError:
                            pass

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return {
            "transferred": transferred,
            "skipped": skipped,
            "errors": errors,
            "total": total,
        }

    def process_job_sync(self, job: dict):
        jid = job["id"]
        src_id, dst_id = job["src_id"], job["dst_id"]
        src_type, dst_type = job["src_type"], job["dst_type"]
        op = job["operation"]
        raw = job["file_path"]

        file_path, dst_path = (raw.split("::", 1) if "::" in raw else (raw, raw))

        def _spawn_children(children: list):
            rows = [{
                "user_id": job.get("user_id"), "src_id": src_id, "src_type": src_type,
                "dst_id": dst_id, "dst_type": dst_type, "operation": op,
                "file_path": f"{csrc}::{dst_path.rstrip('/')}/{rel}",
                "status": "pending", "progress": 0,
            } for csrc, rel in children]
            if supabase and rows:
                for i in range(0, len(rows), 100):
                    supabase.table("transfer_jobs").insert(rows[i:i + 100]).execute()
                self._db(jid, status="completed", progress=100)

        try:
            self._db(jid, progress=5, status="initializing")
            src_url = _hf_url(src_type, src_id, file_path)
            dst_url = _hf_url(dst_type, dst_id, dst_path)

            if op in ("copy", "move"):
                self._cancelled(jid)
                xet_success = False

                if dst_type == "bucket":
                    try:
                        self._db(jid, progress=20, status="copying_server_side")
                        safe_src = file_path.lstrip("/")
                        safe_dst = dst_path.lstrip("/")
                        xet_hash = None

                        if src_type == "bucket":
                            paths_gen = self.src_api.get_bucket_paths_info(src_id, paths=[safe_src])
                            info = next(paths_gen, None)
                            if info:
                                xet_hash = getattr(info, "xet_hash", None)
                        else:
                            tree = self.src_api.list_repo_tree(repo_id=src_id, repo_type=src_type, recursive=True)
                            for f in tree:
                                if isinstance(f, RepoFile) and f.path == safe_src:
                                    xet_hash = getattr(f, "blob_id", None) or getattr(f, "xet_hash", None)
                                    break

                        if xet_hash:
                            self.dst_api.batch_bucket_files(
                                bucket_id=dst_id,
                                copy=[(src_type, src_id, xet_hash, safe_dst)]
                            )
                            xet_success = True
                        else:
                            print(f"Could not resolve xet_hash for {safe_src}, falling back to proxy.")
                    except Exception as e:
                        resp = getattr(e, "response", None)
                        detail = f"{resp.status_code} {resp.text}" if resp is not None else repr(e)
                        if resp is not None and resp.status_code == 400 and "Unable to duplicate Xet hashes" in resp.text:
                            print(f"Xet dedup not available for {safe_src}, falling back to proxy (expected).")
                        else:
                            print(f"Server-side Xet copy failed: {detail}")

                if not xet_success:
                    tmp = tempfile.mkdtemp()
                    try:
                        self._cancelled(jid)
                        self._db(jid, progress=10, status="downloading")

                        if src_type == "bucket":
                            try:
                                info = self.src_fs.info(src_url)
                                if info.get("type") == "directory":
                                    prefix = f"buckets/{src_id}/{file_path.rstrip('/')}/"
                                    paths = self.src_fs.find(src_url, detail=False)
                                    children = [
                                        (f"{file_path.rstrip('/')}/{(p.split(prefix)[-1] if prefix in p else p.split('/')[-1])}",
                                         p.split(prefix)[-1] if prefix in p else p.split("/")[-1])
                                        for p in paths if not p.endswith("/")
                                    ]
                                    if children:
                                        _spawn_children(children)
                                        return
                            except Exception:
                                pass

                            local = os.path.join(tmp, os.path.basename(file_path))
                            try:
                                with self.src_fs.open(src_url, "rb") as f_in:
                                    with open(local, "wb") as f_out:
                                        shutil.copyfileobj(f_in, f_out)
                            except FileNotFoundError:
                                raise Exception(f"Access denied or missing: {src_url}")
                            downloaded = local
                        else:
                            try:
                                downloaded = hf_hub_download(
                                    repo_id=src_id, filename=file_path,
                                    repo_type=src_type, token=self.src_token, cache_dir=tmp,
                                )
                            except Exception as e:
                                err = str(e).lower()
                                if any(k in err for k in ("404", "entry not found", "not found")):
                                    prefix = file_path.rstrip("/") + "/"
                                    tree = self.src_api.list_repo_tree(
                                        repo_id=src_id, repo_type=src_type, recursive=True)
                                    children = [(f.path, f.path[len(prefix):])
                                                for f in tree if f.path.startswith(prefix) and isinstance(f, RepoFile)]
                                    if children:
                                        _spawn_children(children)
                                        return
                                raise

                        self._cancelled(jid)
                        self._db(jid, progress=50, status="uploading")

                        if dst_type == "bucket":
                            self.dst_fs.put_file(downloaded, dst_url)
                        else:
                            self.dst_api.upload_file(
                                path_or_fileobj=downloaded, path_in_repo=dst_path,
                                repo_id=dst_id, repo_type=dst_type,
                                commit_message=f"{op.capitalize()} {dst_path} from {src_id}",
                            )
                    finally:
                        shutil.rmtree(tmp, ignore_errors=True)

                if op == "move":
                    self._db(jid, progress=95, status="cleaning_up")
                    self.delete_file(repo_id=src_id, path_in_repo=file_path, repo_type=src_type)
                    if supabase:
                        try:
                            self._retry_db(lambda: supabase.table("drive_items").delete().eq("storage_path", file_path).execute())
                        except Exception as db_err:
                            print(f"Failed to clear db row after move: {db_err}")

            elif op == "extract":
                self._extract(jid, src_id, dst_id, src_type, dst_type, file_path)

            self._db(jid, progress=100, status="completed")

        except Exception as e:
            if str(e) != "Job cancelled by user":
                self._db(jid, status="failed", error_message=str(e))

    def _extract(self, jid, src_id, dst_id, src_type, dst_type, file_path):
        tmp = tempfile.mkdtemp()
        ext_dir = os.path.join(tmp, "extracted")
        os.makedirs(ext_dir, exist_ok=True)

        try:
            self._db(jid, progress=5, status="downloading_archive")

            files_dl = [file_path]
            target = file_path

            if re.search(r"\.part\d+\.rar$", file_path, re.IGNORECASE):
                base = re.sub(r"\.part\d+\.rar$", "", file_path, flags=re.IGNORECASE)
                all_f = self.list_files(src_id, src_type)
                files_dl = sorted(
                    f["path"] for f in all_f
                    if f["path"].startswith(base + ".part") and f["path"].lower().endswith(".rar")
                )
                target = files_dl[0] if files_dl else file_path

            main_dl = None
            total = len(files_dl)

            for idx, f in enumerate(files_dl):
                self._cancelled(jid)
                if src_type == "bucket":
                    local = os.path.join(tmp, os.path.basename(f))
                    with self.src_fs.open(_hf_url(src_type, src_id, f), "rb") as f_in:
                        with open(local, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    dl = local
                else:
                    dl = hf_hub_download(
                        repo_id=src_id, filename=f, repo_type=src_type,
                        token=self.src_token, cache_dir=tmp,
                    )
                if f == target:
                    main_dl = dl
                self._db(jid, progress=5 + int(((idx + 1) / total) * 45))

            self._cancelled(jid)
            self._db(jid, progress=50, status="extracting")

            low = main_dl.lower()
            if low.endswith(".zip"):
                with zipfile.ZipFile(main_dl, "r") as z:
                    z.extractall(ext_dir)
            elif low.endswith(".7z"):
                import py7zr
                with py7zr.SevenZipFile(main_dl, "r") as z:
                    z.extractall(ext_dir)
            elif low.endswith(".rar"):
                import rarfile
                with rarfile.RarFile(main_dl) as z:
                    z.extractall(ext_dir)
            else:
                raise Exception("Unsupported format. Use .zip, .7z, or .rar.")

            self._cancelled(jid)
            self._db(jid, progress=70, status="uploading_extracted")

            if dst_type == "bucket":
                for root, _, fnames in os.walk(ext_dir):
                    for fn in fnames:
                        lp = os.path.join(root, fn)
                        rel = os.path.relpath(lp, ext_dir).replace("\\", "/")
                        self.dst_fs.put_file(lp, _hf_url(dst_type, dst_id, rel))
            else:
                self.dst_api.upload_folder(
                    folder_path=ext_dir, repo_id=dst_id, repo_type=dst_type,
                    commit_message=f"Extracted from {src_id}",
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

# ─── JOB PROCESSOR ───
async def process_single_job(job: dict):
    try:
        user_id = job.get("user_id")
        for k in ("src_id", "dst_id"):
            if job.get(k) == "native-drive":
                job[k] = MASTER_HF_REPO

        global_tok = MASTER_HF_TOKEN or os.environ.get("GLOBAL_HF_TOKEN")
        src_tok = dst_tok = global_tok

        if supabase and user_id:
            res = await _sb_query(lambda: (
                supabase.table("user_settings").select("settings").eq("id", user_id).execute()
            ))
            if res and res.data:
                settings = res.data[0].get("settings", {})
                if settings.get("hf_token"):
                    src_tok = dst_tok = settings["hf_token"]

                src_user = job.get("src_id", "").split("/")[0] if "/" in job.get("src_id", "") else ""
                dst_user = job.get("dst_id", "").split("/")[0] if "/" in job.get("dst_id", "") else ""
                for e in settings.get("hfTokenMap", []):
                    if e.get("user") == src_user and e.get("token"):
                        src_tok = e["token"]
                    if e.get("user") == dst_user and e.get("token"):
                        dst_tok = e["token"]

        if not src_tok or not dst_tok:
            raise Exception("Missing HF token. Configure in settings.")

        await asyncio.to_thread(HFManager(src_token=src_tok, dst_token=dst_tok).process_job_sync, job)
    except Exception as e:
        print(f"Job {job.get('id', '?')} failed: {e}")
        if supabase:
            await _sb_query(lambda: (
                supabase.table("transfer_jobs")
                .update({"status": "failed", "error_message": str(e)})
                .eq("id", job.get("id", "")).execute()
            ))

# ─── WORKER LOOP ───
async def worker_loop():
    if not supabase:
        print("No Supabase — worker disabled.")
        return
    while True:
        try:
            if semaphore._value == 0:
                await asyncio.sleep(2)
                continue

            res = await _sb_query(lambda: (
                supabase.table("transfer_jobs").select("*")
                .eq("status", "pending")
                .neq("dst_type", "gdrive")   # GDrive jobs run via their own background task
                .order("created_at").limit(1).execute()
            ))
            if not res or not res.data:
                await asyncio.sleep(3)
                continue

            job = res.data[0]
            claim = await _sb_query(lambda: (
                supabase.table("transfer_jobs")
                .update({"status": "queued_locally", "progress": 1})
                .eq("id", job["id"]).eq("status", "pending").execute()
            ))
            if not claim or not claim.data:
                continue

            async def _run(j=job):
                async with semaphore:
                    await process_single_job(j)

            asyncio.create_task(_run())
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Queue error: {e}")
            await asyncio.sleep(5)

# ─── ROUTES ───
@app.get("/")
def root():
    return {"status": "ok", "message": "HF Manager API is live"}

@app.get("/api/list-files")
async def api_list(repo_id: str, token: str, repo_type: str = "model"):
    try:
        files = await asyncio.to_thread(HFManager(token).list_files, repo_id, repo_type)
        return {"repo": repo_id, "type": repo_type, "files": files}
    except RepositoryNotFoundError:
        raise HTTPException(404, f"Repository '{repo_id}' not found.")
    except Exception as e:
        raise HTTPException(_hf_status(e), str(e))

@app.delete("/api/delete-file")
async def api_delete_file(repo_id: str, path: str, token: str, repo_type: str = "model"):
    try:
        await asyncio.to_thread(HFManager(token).delete_file, repo_id, path, repo_type)
        return {"message": f"Deleted '{path}' from {repo_type} '{repo_id}'."}
    except RepositoryNotFoundError:
        raise HTTPException(404, f"Repository '{repo_id}' not found.")
    except Exception as e:
        raise HTTPException(_hf_status(e), str(e))

@app.get("/api/duplicate")
async def api_duplicate(
    from_id: str, to_id: str,
    repo_type: str = "model",
    token: Optional[str] = None, write_token: Optional[str] = None, read_token: Optional[str] = None
):
    active_token = write_token or token
    if not active_token:
        raise HTTPException(400, "Authentication token missing.")
    try:
        await asyncio.to_thread(HFManager(active_token).duplicate_repo, from_id, to_id, repo_type)
        return {"message": f"Successfully duplicated to {to_id}"}
    except Exception as e:
        status_code = getattr(getattr(e, "response", None), "status_code", 400)
        raise HTTPException(status_code=status_code, detail=str(e))

@app.post("/api/move-file")
async def api_move_file(repo_id: str, old_path: str, new_path: str, token: str, repo_type: str = "model"):
    try:
        await asyncio.to_thread(HFManager(token).move_file, repo_id, old_path, new_path, repo_type)
        return {"message": f"Moved '{old_path}' → '{new_path}'."}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/rename-repo")
async def api_rename_repo(old_id: str, new_id: str, token: str, repo_type: str = "model"):
    try:
        await asyncio.to_thread(HFManager(token).rename_repo, old_id, new_id, repo_type)
        return {"message": f"Renamed '{old_id}' → '{new_id}'."}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.delete("/api/delete-repo")
async def api_delete_repo(repo_id: str, token: str, repo_type: str = "model"):
    try:
        await asyncio.to_thread(HFManager(token).delete_repo, repo_id, repo_type)
        return {"message": f"Deleted repo '{repo_id}'."}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/create-folder")
async def api_create_folder(repo_id: str, folder_path: str, token: str, repo_type: str = "model"):
    try:
        def _create():
            HfApi(token=token).upload_file(
                path_or_fileobj=b"", path_in_repo=f"{folder_path.rstrip('/')}/.gitkeep",
                repo_id=repo_id, repo_type=repo_type,
                commit_message=f"Create folder {folder_path}",
            )
        await asyncio.to_thread(_create)
        return {"message": f"Folder '{folder_path}' created."}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/archive-files")
async def api_archive_files(repo_id: str, token: str, req: ArchiveRequest, repo_type: str = "model"):
    try:
        await asyncio.to_thread(
            HFManager(token).archive_files, repo_id, repo_type, req.files, req.archive_name)
        return {"message": f"Created {req.archive_name}"}
    except Exception as e:
        raise HTTPException(400, str(e))


# ─── GDRIVE OAUTH ENDPOINTS ─────────────────────────────────────────────────

class GDriveOAuthConfigRequest(BaseModel):
    client_secret_json: str   # full JSON string of the downloaded client_secret_*.json


@app.post("/api/gdrive/oauth/config")
async def api_gdrive_oauth_config(req: GDriveOAuthConfigRequest):
    try:
        data = json.loads(req.client_secret_json)
        client_type = "web" if "web" in data else "installed"   # ← detect type
        info = data.get("web") or data.get("installed")
        if not info:
            raise ValueError("JSON must contain 'installed' or 'web' key.")
        cid  = info["client_id"]
        csec = info["client_secret"]
    except Exception as e:
        raise HTTPException(400, f"Invalid client secret JSON: {e}")

    os.environ["GDRIVE_OAUTH_CLIENT_ID"]     = cid
    os.environ["GDRIVE_OAUTH_CLIENT_SECRET"] = csec
    os.environ["GDRIVE_OAUTH_CLIENT_TYPE"]   = client_type   # ← ADD THIS LINE
    global GDRIVE_OAUTH_CLIENT_ID, GDRIVE_OAUTH_CLIENT_SECRET
    GDRIVE_OAUTH_CLIENT_ID     = cid
    GDRIVE_OAUTH_CLIENT_SECRET = csec
    return {"status": "ok", "client_id": cid}


@app.get("/api/gdrive/oauth/url")
async def api_gdrive_oauth_url(
    redirect_uri: Optional[str] = None,
    code_challenge: Optional[str] = None,
    code_challenge_method: Optional[str] = None,
    state: Optional[str] = None,
):
    """
    Generate the Google OAuth2 authorisation URL.

    Uses direct urllib URL construction — bypasses google-auth-oauthlib so PKCE
    params are never dropped and any registered redirect_uri is accepted regardless
    of whether the credential is type "web" or "installed".

    The frontend encodes { apiBase, codeVerifier } as base64 JSON in `state`.
    The callback page at /oauth-callback reads state and POSTs the code directly
    to the backend — no manual copy-paste needed.

    Supports PKCE: pass code_challenge + code_challenge_method=S256.
    The matching code_verifier must be sent to /callback when exchanging the code.
    """
    ruri = redirect_uri or GDRIVE_OAUTH_REDIRECT_URI or "http://localhost"
    try:
        auth_url = _build_gdrive_auth_url(
            redirect_uri=ruri,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            state=state,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"auth_url": auth_url}


class GDriveOAuthCallbackRequest(BaseModel):
    code: str
    redirect_uri: Optional[str] = None
    code_verifier: Optional[str] = None   # PKCE verifier — must match the challenge sent during /url


@app.post("/api/gdrive/oauth/callback")
async def api_gdrive_oauth_callback(req: GDriveOAuthCallbackRequest):
    import urllib.parse
    code = urllib.parse.unquote(req.code.strip())
    redirect_uri = req.redirect_uri or GDRIVE_OAUTH_REDIRECT_URI or "http://localhost"

    try:
        # Always use direct POST — avoids google-auth-oauthlib mangling
        # the redirect_uri or dropping the code_verifier in PKCE flows.
        token_data = {
            "code":          code,
            "client_id":     GDRIVE_OAUTH_CLIENT_ID,
            "client_secret": GDRIVE_OAUTH_CLIENT_SECRET,
            "redirect_uri":  redirect_uri,
            "grant_type":    "authorization_code",
        }
        if req.code_verifier:
            token_data["code_verifier"] = req.code_verifier

        token_resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data=token_data,
            timeout=15,
        )
        token_json = token_resp.json()

        if not token_resp.ok:
            err = token_json.get("error", "unknown_error")
            desc = token_json.get("error_description", "")
            if err == "invalid_grant":
                raise HTTPException(400,
                    "invalid_grant — code already used or expired (>10 min). "
                    "Click 'Authorise with Google' to start a fresh flow."
                )
            raise HTTPException(400, f"Token exchange failed: {err} — {desc}")

        refresh_token = token_json.get("refresh_token")
        access_token  = token_json.get("access_token")

        if not refresh_token:
            raise HTTPException(400,
                "No refresh_token returned — revoke access at "
                "https://myaccount.google.com/permissions and try again."
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Token exchange failed: {e}")

    # Fetch email
    email = ""
    try:
        ui = requests.get(
            "https://www.googleapis.com/oauth2/v1/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=5,
        )
        if ui.ok:
            email = ui.json().get("email", "")
    except Exception:
        pass

    return {
        "refresh_token": refresh_token,
        "token":         access_token,
        "client_id":     GDRIVE_OAUTH_CLIENT_ID,
        "scopes":        token_json.get("scope", "").split(),
        "email":         email,
    }


@app.get("/api/gdrive/oauth/verify")
async def api_gdrive_oauth_verify(refresh_token: str):
    """Verify a refresh_token still works and return the account email + storage info."""
    try:
        service = _get_gdrive_service(oauth_refresh_token=refresh_token)
        about   = service.about().get(fields="user,storageQuota").execute()
        email   = about.get("user", {}).get("emailAddress", "unknown")
        quota   = about.get("storageQuota", {})
        used    = int(quota.get("usage", 0))
        total   = int(quota.get("limit", 0))
        return {
            "status": "ok",
            "email":  email,
            "storage_used_gb":  round(used  / (1 << 30), 2),
            "storage_total_gb": round(total / (1 << 30), 2) if total else None,
        }
    except Exception as e:
        raise HTTPException(400, f"Token verification failed: {e}")


# ─── GDRIVE TRANSFER ENDPOINTS ───────────────────────────────────────────────

@app.post("/api/gdrive/transfer")
async def api_gdrive_transfer(req: GDriveTransferRequest, background_tasks: BackgroundTasks):
    """
    Transfer an entire HF repo/dataset/space/bucket to a Google Drive folder.

    Uses Option B: downloads one file at a time, uploads to Drive, deletes from
    disk immediately. Max disk usage = size of largest single file.

    Auth priority for GDrive:
      1. sa_json field in request body (JSON string or file path)
      2. GDRIVE_SA_JSON environment variable

    The Drive folder must be shared with the service account email as Editor.

    With Supabase: returns immediately with job_id, runs in background.
    Without Supabase: blocks until complete (use for testing/small repos).
    """
    # Resolve HF token: request body → user settings → master token
    hf_token = req.hf_token or MASTER_HF_TOKEN

    if supabase and req.user_id:
        res = await _sb_query(lambda: (
            supabase.table("user_settings").select("settings").eq("id", req.user_id).execute()
        ))
        if res and res.data:
            settings = res.data[0].get("settings", {})
            if settings.get("hf_token"):
                hf_token = hf_token or settings["hf_token"]

    # Enqueue as a Supabase job if available
    if supabase:
        job_row = {
            "user_id":   req.user_id,
            "src_id":    req.repo_id,
            "src_type":  req.repo_type,
            "dst_id":    f"gdrive:{req.gdrive_folder_id}",
            "dst_type":  "gdrive",
            "operation": "copy",
            "file_path": req.path_filter or "",   # empty = "all files" (avoid literal "*" as a path)
            "status":    "pending",
            "progress":  0,
        }
        res = await _sb_query(lambda: supabase.table("transfer_jobs").insert(job_row).execute())
        job_id = res.data[0]["id"] if res and res.data else None

        async def _run_gdrive_job():
            manager = HFManager(token=hf_token)
            try:
                result = await asyncio.to_thread(
                    manager.transfer_to_gdrive_sync, job_id, req
                )
                summary = (
                    f"Done: {result['transferred']}/{result['total']} transferred"
                    + (f", {result['skipped']} errors" if result["skipped"] else "")
                )
                manager._db(job_id, status="completed", progress=100, error_message=summary)
            except Exception as e:
                manager._db(job_id, status="failed", error_message=str(e))
                print(f"GDrive job {job_id} failed: {e}")

        background_tasks.add_task(_run_gdrive_job)
        return {
            "status":   "queued",
            "job_id":   job_id,
            "repo_id":  req.repo_id,
            "repo_type": req.repo_type,
            "gdrive_folder_id": req.gdrive_folder_id,
        }

    # No Supabase — run inline (blocks)
    manager = HFManager(token=hf_token)
    try:
        result = await asyncio.to_thread(manager.transfer_to_gdrive_sync, None, req)
        return {"status": "completed", **result}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/gdrive/transfer/{job_id}")
async def api_gdrive_transfer_status(job_id: str):
    """
    Poll the status of a GDrive transfer job.
    Returns the full job row: status, progress (0-100), error_message.
    Requires Supabase to be configured.
    """
    if not supabase:
        raise HTTPException(503, "Supabase not configured; job tracking unavailable.")
    res = await _sb_query(lambda: (
        supabase.table("transfer_jobs").select("*").eq("id", job_id).execute()
    ))
    if not res or not res.data:
        raise HTTPException(404, "Job not found.")
    return res.data[0]


@app.delete("/api/gdrive/transfer/{job_id}")
async def api_gdrive_transfer_cancel(job_id: str):
    """
    Cancel a running or queued GDrive transfer job.
    The worker checks for cancellation between each file upload.
    """
    if not supabase:
        raise HTTPException(503, "Supabase not configured.")
    res = await _sb_query(lambda: (
        supabase.table("transfer_jobs")
        .update({"status": "cancelled"})
        .eq("id", job_id)
        .in_("status", ["pending", "queued_locally", "initializing", "uploading"])
        .execute()
    ))
    if not res or not res.data:
        raise HTTPException(404, "Job not found or already completed.")
    return {"status": "cancelled", "job_id": job_id}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
