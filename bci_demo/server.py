"""Hands-free PDF / audio / video demo controlled by EEG.

Reads EEG from LSL, runs the SSVEP + jaw-clench detectors, and pushes
commands to the web page at http://localhost:8780.

Run (from the ant-hackathon folder):
  uv run python bci_demo/server.py --fake            # fake headset, keyboard-controlled
  uv run python bci_demo/server.py                   # real headset (first LSL stream of type EEG)
  uv run python bci_demo/server.py --stream NAME     # real headset, pick stream by name
Options: --pdf FILE, --ssvep-threshold 0.42, --clench-factor 5
"""
import argparse
import json
import queue
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium

from engine import EEGEngine

HERE = Path(__file__).parent
STATIC = HERE / "static"


# ---------- sending events to every open browser tab ----------
class Broadcaster:
    def __init__(self):
        self.clients = []
        self.lock = threading.Lock()

    def add(self):
        q = queue.Queue(maxsize=200)
        with self.lock:
            self.clients.append(q)
        return q

    def remove(self, q):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def send(self, msg):
        with self.lock:
            for q in self.clients:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass


bus = Broadcaster()


# ---------- PDF pages as images ----------
class PDF:
    def __init__(self, path):
        self.doc = pdfium.PdfDocument(str(path))
        self.lock = threading.Lock()
        self.cache = {}

    def __len__(self):
        return len(self.doc)

    def png(self, i):
        with self.lock:
            if i not in self.cache:
                img = self.doc[i].render(scale=2).to_pil()
                out = BytesIO()
                img.save(out, "PNG")
                self.cache[i] = out.getvalue()
            return self.cache[i]


def make_handler(worker, pdf, fake):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(STATIC), **kw)

        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/events":
                return self._events()
            if self.path == "/api/pdf":
                return self._json({"pages": len(pdf), "fake": fake is not None})
            if self.path.startswith("/api/page/"):
                try:
                    i = int(self.path.rsplit("/", 1)[1].split(".")[0])
                    data = pdf.png(i)
                except (ValueError, IndexError):
                    return self._json({"error": "bad page"}, 404)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers()
                self.wfile.write(data)
                return
            if self.path.startswith("/media/") and self.headers.get("Range"):
                return self._range()
            return super().do_GET()

        def _range(self):
            """Serve part of a media file, so the browser can seek (skip 15 s) in video/audio."""
            path = Path(self.translate_path(self.path))
            if not path.is_file():
                return self.send_error(404)
            size = path.stat().st_size
            spec = self.headers["Range"].split("=", 1)[1].split(",")[0]
            a, _, b = spec.partition("-")
            start = int(a) if a else max(0, size - int(b))
            end = min(int(b), size - 1) if a and b else size - 1
            if start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                return self.end_headers()
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(str(path)))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                try:
                    while left > 0:
                        data = f.read(min(65536, left))
                        if not data:
                            break
                        self.wfile.write(data)
                        left -= len(data)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/targets":
                worker.set_freqs(body.get("freqs", []))
                return self._json({"ok": True})
            if self.path == "/api/sim" and fake:
                if "look" in body:
                    fake.look(body["look"])
                if body.get("clench"):
                    fake.clench(body["clench"])
                return self._json({"ok": True})
            return self._json({"error": "not found"}, 404)

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q = bus.add()
            try:
                while True:
                    try:
                        msg = q.get(timeout=15)
                        self.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            finally:
                bus.remove(q)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="use a fake headset (keyboard controlled)")
    ap.add_argument("--stream", help="LSL stream name (default: first stream of type EEG)")
    ap.add_argument("--pdf", default=str(HERE / "media" / "sample.pdf"))
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--ssvep-threshold", type=float, default=0.42)
    ap.add_argument("--clench-factor", type=float, default=5.0)
    args = ap.parse_args()

    fake = None
    if args.fake:
        from fake_eeg import NAME, FakeHeadset
        fake = FakeHeadset()
        fake.start()
        args.stream = NAME
        print("Fake headset running. In the page: hold Shift+arrow to fake a 'look', Shift+J to fake a 'clench'.")

    if not Path(args.pdf).exists():
        raise SystemExit(f"No PDF at {args.pdf}. Run: uv run python bci_demo/make_media.py  (or pass --pdf FILE)")
    worker = EEGEngine(bus.send, stream=args.stream, fake=args.fake,
                       ssvep_threshold=args.ssvep_threshold, clench_factor=args.clench_factor)
    worker.start()
    pdf = PDF(args.pdf)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(worker, pdf, fake))
    server.daemon_threads = True
    print(f"Open http://localhost:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
