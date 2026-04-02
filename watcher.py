"""
watcher.py — Event-Driven Ingestion Trigger
============================================
Watches the /docs folder for new .txt files.
When a new file appears, it automatically calls ingest_document().

Two modes (select via WATCHER_MODE env var):
  1. "watchdog"  — uses watchdog library to monitor filesystem events (default)
  2. "webhook"   — starts a tiny HTTP server; POST /webhook triggers ingestion

To later connect to OneDrive webhooks:
  - Register a change notification on your OneDrive folder via MS Graph API
  - Point the webhook URL to this server's /webhook endpoint
  - Microsoft will POST a notification when files change
  - This server calls ingest_document() on the new file path
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

# Ensure backend/ is importable
sys.path.insert(0, str(Path(__file__).parent))
from ingest import ingest_document, DOCS_FOLDER

logging.basicConfig(
    level  =logging.INFO,
    format ="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("watcher")


# ════════════════════════════════════════════════════════════════════════════════
# MODE 1 — Filesystem watcher (watchdog)
# ════════════════════════════════════════════════════════════════════════════════

def start_filesystem_watcher(watch_folder: Path = DOCS_FOLDER) -> None:
    """
    Use watchdog to monitor *watch_folder*.
    Triggers ingestion when a .txt file is created or modified.
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events    import FileSystemEventHandler
    except ImportError:
        log.error("watchdog not installed — run: pip install watchdog")
        sys.exit(1)

    class IngestionHandler(FileSystemEventHandler):
        def _should_process(self, event) -> bool:
            return (
                not event.is_directory
                and event.src_path.endswith(".txt")
            )

        def on_created(self, event):
            if self._should_process(event):
                log.info(f"📄  New file detected: {event.src_path}")
                _safe_ingest(str(event.src_path))

        def on_modified(self, event):
            if self._should_process(event):
                log.info(f"✏️   File modified: {event.src_path}")
                _safe_ingest(str(event.src_path))

    watch_folder.mkdir(parents=True, exist_ok=True)
    observer = Observer()
    observer.schedule(IngestionHandler(), str(watch_folder), recursive=True)
    observer.start()
    log.info(f"👁️  Watching folder: {watch_folder}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# ════════════════════════════════════════════════════════════════════════════════
# MODE 2 — Webhook server (OneDrive / any HTTP POST)
# ════════════════════════════════════════════════════════════════════════════════

class WebhookHandler(BaseHTTPRequestHandler):
    """
    Minimal HTTP handler that accepts POST /webhook.

    Expected body (JSON):
      { "file_path": "/absolute/path/to/document.txt" }

    OneDrive integration:
      Microsoft's webhook sends a POST to your URL with a 'value' array
      containing changed resource URLs. You would then download the file
      from OneDrive and call ingest_document() on the local copy.
    """

    def do_POST(self):
        if self.path != "/webhook":
            self._respond(404, {"error": "Not found"})
            return

        length  = int(self.headers.get("Content-Length", 0))
        raw     = self.rfile.read(length)

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._respond(400, {"error": "Invalid JSON"})
            return

        file_path = payload.get("file_path", "")
        if not file_path:
            self._respond(400, {"error": "'file_path' is required"})
            return

        log.info(f"🔔  Webhook received — ingesting: {file_path}")
        # Run ingestion in a background thread so we can respond immediately
        Thread(target=_safe_ingest, args=(file_path,), daemon=True).start()
        self._respond(202, {"status": "accepted", "file": file_path})

    def do_GET(self):
        """Health check endpoint."""
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "Not found"})

    def _respond(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        log.info(f"HTTP  {self.client_address[0]}  {format % args}")


def start_webhook_server(host: str = "0.0.0.0", port: int = 9000) -> None:
    server = HTTPServer((host, port), WebhookHandler)
    log.info(f"🔗  Webhook server listening on http://{host}:{port}/webhook")
    log.info("    POST body: { \"file_path\": \"/docs/my_file.txt\" }")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down webhook server …")


# ════════════════════════════════════════════════════════════════════════════════
# Shared helper
# ════════════════════════════════════════════════════════════════════════════════

def _safe_ingest(file_path: str) -> None:
    """Wrap ingest_document() with error handling so the watcher never crashes."""
    try:
        n = ingest_document(file_path)
        log.info(f"✅  Ingested {n} chunks from {Path(file_path).name}")
    except Exception as exc:
        log.error(f"❌  Ingestion failed for {file_path}: {exc}")


# ════════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mode = os.getenv("WATCHER_MODE", "watchdog").lower()
    log.info(f"Starting in mode: {mode.upper()}")

    if mode == "webhook":
        port = int(os.getenv("WEBHOOK_PORT", "9000"))
        start_webhook_server(port=port)
    else:
        start_filesystem_watcher()
