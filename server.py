"""
Studio Trainer — FastAPI server layer.

Owns the live training process (TrainingManager: single-run 409 lock,
stop_requested flag, rolling stdout tail, terminal done/error classification),
exposes REST commands + SSE streams, serves previews with a path-traversal
guard, and serves the offline frontend from assets/. The engine itself lives in
trainer_core.py and knows nothing about the process or the framework.
"""

import json
import os
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Dict, Iterator, List, Optional

# The portable python_embeded (._pth config) does not auto-add the script's own
# directory to sys.path, so a sibling import of trainer_core fails. Make this
# entry point self-locating regardless of how it's launched.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import trainer_core as core


app = FastAPI(title="Studio Trainer")

ASSETS_DIR = core.ROOT / "assets"
HOST = "127.0.0.1"
PORT = 7860


# ==========================================
# TRAINING MANAGER — process ownership lives here, not in the engine.
# ==========================================
class AlreadyRunning(Exception):
    pass


class TrainingManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.proc = None
        self.stop_requested = False
        self.tail: List[str] = []

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def begin(self, launch: Dict):
        """Spawn under the lock. Rejects a second concurrent run (no two
        accelerate processes over one GPU)."""
        with self._lock:
            if self.is_running():
                raise AlreadyRunning()
            self.stop_requested = False
            self.tail = []
            self.proc = core.spawn(launch)
            return self.proc

    def stop(self) -> str:
        with self._lock:
            if not self.is_running():
                return "Not running."
            self.stop_requested = True
            return core.kill_process(self.proc)

    def reset(self):
        with self._lock:
            self.proc = None


manager = TrainingManager()


# ==========================================
# SSE HELPERS
# ==========================================
def sse(event: Dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def sse_stream(generator: Iterator[Dict]) -> StreamingResponse:
    def body():
        for ev in generator:
            yield sse(ev)
    return StreamingResponse(body(), media_type="text/event-stream")


def _merge_body_into_settings(body: Optional[Dict]) -> Dict:
    """Persisted settings with an optional request body merged on top (known keys only)."""
    settings = core.get_settings()
    if body:
        for k, v in body.items():
            if k in core.DEFAULT_SETTINGS:
                settings[k] = v
    return settings


async def _json_body(req: Request) -> Dict:
    try:
        data = await req.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ==========================================
# SETTINGS
# ==========================================
@app.get("/settings")
def get_settings():
    return JSONResponse(core.get_settings())


@app.post("/settings")
async def post_settings(req: Request):
    body = await _json_body(req)
    merged = core.save_settings(body)
    return JSONResponse(merged)


# ==========================================
# TRAINING
# ==========================================
def _train_stream(settings: Dict) -> Iterator[Dict]:
    errors = core.validate_training(settings)
    if errors:
        yield core.ev_error("; ".join(errors))
        return

    yield core.ev_log(f"Preparing: {core.project_name_from_trigger(settings.get('trigger_word', ''))}...")
    launch = core.build_launch(settings)
    for w in launch["warnings"]:
        yield core.ev_log(w)

    try:
        proc = manager.begin(launch)
    except AlreadyRunning:
        yield core.ev_error("A training run is already active.")
        return

    try:
        for ev in core.tail_process(proc, launch["sample_dir"], manager.tail):
            yield ev
        proc.wait()
        rc = proc.returncode
        if rc == 0:
            yield core.ev_done("complete")
        elif manager.stop_requested:
            # Tree-kill exits nonzero; expected — replaces the old "exit status 15" match.
            yield core.ev_done("stopped by user")
        else:
            yield core.ev_error(f"Training failed (exit {rc}).", exit_code=rc, tail=manager.tail[-30:])
    except Exception as e:
        yield core.ev_error(f"Stream error: {e}")
    finally:
        manager.reset()


@app.post("/train/start")
async def train_start(req: Request):
    if manager.is_running():
        return JSONResponse({"error": "A training run is already active."}, status_code=409)
    body = await _json_body(req)
    settings = _merge_body_into_settings(body)
    return sse_stream(_train_stream(settings))


@app.post("/train/stop")
def train_stop():
    return JSONResponse({"message": manager.stop()})


@app.get("/train/status")
def train_status():
    return JSONResponse({"running": manager.is_running()})


# ==========================================
# DATASET TOOLS (SSE progress)
# ==========================================
@app.post("/bucket")
async def bucket(req: Request):
    s = _merge_body_into_settings(await _json_body(req))
    return sse_stream(core.run_smart_crop(s.get("dataset_path", ""), s.get("side_min", 512), s.get("side_max", 768)))


@app.post("/tag")
async def tag(req: Request):
    s = _merge_body_into_settings(await _json_body(req))
    return sse_stream(core.run_auto_tagging(
        s.get("dataset_path", ""), s.get("tagger_gen_thresh", 0.35),
        s.get("tagger_char_thresh", 0.85), s.get("tagger_overwrite", False)))


@app.post("/prune")
async def prune(req: Request):
    s = _merge_body_into_settings(await _json_body(req))
    return sse_stream(core.run_prune_tags(s.get("dataset_path", ""), s.get("prune_tags", "")))


# ==========================================
# FOLDER OPEN (local-only)
# ==========================================
@app.post("/folder/output")
async def folder_output(req: Request):
    s = _merge_body_into_settings(await _json_body(req))
    return JSONResponse({"message": core.open_output_folder(s.get("trigger_word", ""))})


@app.post("/folder/dataset")
async def folder_dataset(req: Request):
    s = _merge_body_into_settings(await _json_body(req))
    return JSONResponse({"message": core.open_dataset_folder(s.get("dataset_path", ""))})


# ==========================================
# PREVIEW IMAGE SERVING (path-traversal guarded)
# ==========================================
def _is_within(resolved: Path, root: Path) -> bool:
    try:
        return resolved.is_relative_to(root)          # py3.9+
    except AttributeError:
        try:
            resolved.relative_to(root)                # older fallback
            return True
        except ValueError:
            return False


@app.get("/preview")
def preview(path: str):
    resolved = Path(path).resolve()
    out_root = core.OUTPUT_BASE.resolve()
    if not _is_within(resolved, out_root) or not resolved.is_file():
        return Response(status_code=404)
    return FileResponse(str(resolved))


# ==========================================
# UPDATER (Part 3.4) — stage source files from main, back up, prompt restart.
# Touches only the known source files; never user data, models, runtime, settings.
# ==========================================
RAW_BASE = "https://raw.githubusercontent.com/ToxicHost/Anima-TrainFlow-TH/main/"
MANIFEST_NAME = "update_manifest.json"
# Used only if the manifest can't be fetched/parsed, so the updater still works.
FALLBACK_FILES = ["trainer_core.py", "server.py"]
# Never updated, even if a manifest somehow lists them (belt-and-suspenders).
PROTECTED_PREFIXES = ("settings.json", "models/", "python_embeded/", "training/", "assets/fonts/")


def _fetch(url: str, timeout: int = 20) -> bytes:
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def _is_protected(rel: str) -> bool:
    r = rel.replace("\\", "/").lstrip("./")
    return any(r == p.rstrip("/") or r.startswith(p) for p in PROTECTED_PREFIXES)


def _safe_target(rel: str):
    """Resolve a repo-relative manifest entry to a path INSIDE the app root, or None."""
    r = rel.replace("\\", "/").strip()
    if not r or r.startswith("/") or ".." in r.split("/") or ":" in r:
        return None
    target = (core.ROOT / r).resolve()
    if not _is_within(target, core.ROOT.resolve()):
        return None
    return target


def _validate_payload(rel: str, data: bytes) -> bool:
    if not data:
        return False
    if rel.endswith(".py"):
        return len(data) >= 200 and b"def " in data
    if rel.endswith((".html", ".css", ".js")):
        return len(data) >= 30
    if rel.endswith(".bat"):
        return len(data) >= 10
    return len(data) >= 1


@app.post("/update/check")
def update_check():
    import re
    try:
        remote = _fetch(RAW_BASE + "trainer_core.py").decode("utf-8")
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Update check failed: {e}"}, status_code=200)
    m = re.search(r'VERSION\s*=\s*["\']([^"\']+)["\']', remote)
    remote_ver = m.group(1) if m else "unknown"
    up_to_date = (remote_ver == core.VERSION)
    return JSONResponse({
        "status": "ok",
        "current": core.VERSION,
        "remote": remote_ver,
        "up_to_date": up_to_date,
        "message": f"You're up to date (v{core.VERSION})." if up_to_date else f"v{remote_ver} available (you have v{core.VERSION}).",
    })


@app.post("/update/apply")
def update_apply():
    import tempfile, os, shutil, json as _json

    # 1. Learn the current file set from the manifest on main (fallback to core files).
    used_fallback = False
    try:
        manifest = _json.loads(_fetch(RAW_BASE + MANIFEST_NAME).decode("utf-8"))
        files = manifest.get("files", [])
        if not isinstance(files, list) or not files:
            raise ValueError("empty manifest")
    except Exception:
        files = list(FALLBACK_FILES)
        used_fallback = True

    # 2. Resolve paths, download, validate — stage EVERYTHING before writing anything.
    staged = []  # (rel, target_path, data)
    for rel in files:
        if not isinstance(rel, str) or _is_protected(rel):
            continue
        target = _safe_target(rel)
        if target is None:
            return JSONResponse({"status": "error", "message": f"Refused unsafe manifest path: {rel}. No changes made."})
        try:
            data = _fetch(RAW_BASE + rel.replace("\\", "/"))
        except Exception as e:
            return JSONResponse({"status": "error", "message": f"Download failed for {rel}: {e}. No changes made."})
        if not _validate_payload(rel, data):
            return JSONResponse({"status": "error", "message": f"{rel} looks invalid. No changes made."})
        staged.append((rel, target, data))

    # 3. Back up + atomically swap each file (only now that every download is good).
    swapped: List[str] = []
    for rel, target, data in staged:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                shutil.copy2(str(target), str(target.with_suffix(target.suffix + ".bak")))
            fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=str(target.parent))
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, str(target))
            swapped.append(rel)
        except Exception as e:
            return JSONResponse({"status": "error", "message": f"Failed writing {rel}: {e}. Updated so far: {swapped}."})

    note = " (manifest unavailable — core files only)" if used_fallback else ""
    return JSONResponse({
        "status": "ok",
        "message": f"Updated {len(swapped)} file(s){note}: {', '.join(swapped)}. Restart Studio Trainer to apply (backups saved as *.bak).",
    })


# ==========================================
# FRONTEND (offline static assets). Mounted last so it doesn't shadow the API.
# ==========================================
@app.get("/")
def index():
    index_html = ASSETS_DIR / "index.html"
    if index_html.is_file():
        return FileResponse(str(index_html))
    return JSONResponse({
        "app": "Studio Trainer",
        "version": core.VERSION,
        "note": "Backend is up. Frontend assets/ not built yet — API is curl-testable.",
    })


if ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


def main():
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("uvicorn is required to run the server.")
    threading.Timer(1.5, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    # Single worker: module/app state (the live Popen) does not survive --workers N,
    # and this is a single-GPU localhost app.
    uvicorn.run(app, host=HOST, port=PORT, workers=1, log_level="warning")


if __name__ == "__main__":
    main()
