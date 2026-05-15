"""
nerf_scan/web.py  —  stdlib HTTP server, port 8080.
No Flask. No Jinja. Zero extra dependencies.

Routes:
    GET /          HTML page embedding <model-viewer> (Google web component)
    GET /mesh.glb  Serve current_mesh.glb; 404 if no scan yet.

Call start() from main.py; runs in a daemon thread.
"""

import os, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from .config import WEB_PORT, GLB_PATH

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeRF-Axis — Live Scan</title>
<script type="module"
  src="https://unpkg.com/@google/model-viewer@3/dist/model-viewer.min.js"></script>
<style>
  * { margin:0; padding:0; box-sizing:border-box }
  body { background:#0c0c0c; height:100vh; display:flex;
         flex-direction:column; align-items:center; justify-content:center }
  model-viewer { width:100vw; height:93vh }
  footer { font:11px/2 monospace; color:#444; letter-spacing:.05em }
  footer a { color:#555; text-decoration:none }
</style>
</head>
<body>
<model-viewer src="/mesh.glb" auto-rotate camera-controls
  shadow-intensity="1" environment-image="neutral" exposure="1.1"
  alt="3D scan"></model-viewer>
<footer>nerf-axis &nbsp;·&nbsp; <a href="/mesh.glb">mesh.glb</a></footer>
</body>
</html>
""".encode()

_NO_SCAN = b"No scan available. Trigger a scan on the device first."


class _H(BaseHTTPRequestHandler):
    def log_message(self, *_): pass  # silence access log

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._reply(200, "text/html; charset=utf-8", _HTML)
        elif self.path == "/mesh.glb":
            if os.path.isfile(GLB_PATH):
                data = open(GLB_PATH, "rb").read()
                self._reply(200, "model/gltf-binary", data)
            else:
                self._reply(404, "text/plain", _NO_SCAN)
        else:
            self._reply(404, "text/plain", b"not found")

    def _reply(self, code, ct, body):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


_srv: HTTPServer = None


def start():
    global _srv
    _srv = HTTPServer(("0.0.0.0", WEB_PORT), _H)
    threading.Thread(target=_srv.serve_forever, daemon=True).start()
    print(f"[web] http://gp5.local:{WEB_PORT}/")


def stop():
    if _srv:
        _srv.shutdown()
