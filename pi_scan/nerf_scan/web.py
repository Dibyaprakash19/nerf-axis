"""
nerf_scan/web.py - Minimal HTTP server for mesh viewing.
"""
import os, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from .config import WEB_PORT, GLB_PATH

rescan_event = threading.Event()

_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NeRF-Axis 3D Viewer</title>
<script type="module" src="https://unpkg.com/@google/model-viewer@3/dist/model-viewer.min.js"></script>
<style>
  body { margin:0; background:#000; color:#fff; font-family:sans-serif; display:flex; flex-direction:column; height:100vh; overflow:hidden; }
  model-viewer { flex:1; width:100%; }
  footer { height:100px; display:flex; gap:20px; align-items:center; justify-content:center; background:#111; }
  button { padding:15px 30px; font-size:18px; cursor:pointer; background:#f44; color:#fff; border:none; border-radius:5px; font-weight:bold; }
  button:hover { background:#d33; }
  a { color:#888; text-decoration:none; font-size:14px; }
</style>
<script>
  async function rescan() {
    const btn = document.querySelector('button');
    btn.disabled = true; btn.innerText = 'Scanning...';
    await fetch('/rescan', {method: 'POST'});
    setTimeout(() => location.reload(), 15000);
  }
</script>
</head>
<body>
<model-viewer src="/mesh.glb" auto-rotate camera-controls shadow-intensity="1" environment-image="neutral"></model-viewer>
<footer>
  <button onclick="rescan()">NEW SCAN</button>
  <a href="/mesh.glb">Download GLB</a>
</footer>
</body>
</html>
""".encode()

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._reply(200, "text/html", _HTML)
        elif self.path == "/mesh.glb":
            if os.path.isfile(GLB_PATH):
                self._reply(200, "model/gltf-binary", open(GLB_PATH, "rb").read())
            else:
                self._reply(404, "text/plain", b"No scan data yet.")
        else:
            self._reply(404, "text/plain", b"Not found")

    def do_POST(self):
        if self.path == "/rescan":
            rescan_event.set()
            self._reply(200, "text/plain", b"OK")
        else:
            self._reply(404, "text/plain", b"Not found")

    def _reply(self, code, ct, body):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

_srv = None
def start():
    global _srv
    _srv = HTTPServer(("0.0.0.0", WEB_PORT), _Handler)
    threading.Thread(target=_srv.serve_forever, daemon=True).start()
    print(f"[web] Listening on port {WEB_PORT}")

def stop():
    if _srv: _srv.shutdown()
