import os
import subprocess
import signal
from flask import Flask, render_template, jsonify, send_from_directory

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/live_scan')
PYTHON_BIN = os.path.join(BASE_DIR, 'venv/bin/python3')
FULL_SCAN_SCRIPT = os.path.join(BASE_DIR, 'full_scan.py')

os.makedirs(DATA_DIR, exist_ok=True)

@app.route('/')
@app.route('/gallery')
def gallery():
    """Renders the 3D mesh viewer."""
    return render_template('mesh.html')

@app.route('/scan')
def run_scan():
    """Triggers the full 4-view capture and 3D processing pipeline."""
    try:
        # Run full_scan.py as a subprocess to handle hardware and display
        # We use --no-display if we just want the mesh, but the user wants TFT too
        # So we run it normally. Note: this will block until the scan is done
        # but the visualize loop in full_scan.py is infinite. 
        # We'll modify full_scan.py slightly or run it in background.
        
        # Actually, for a web request, we want it to finish mesh generation 
        # and then keep the TFT visualization running in background.
        
        # Let's run it with a timeout for the generation part.
        process = subprocess.Popen([PYTHON_BIN, FULL_SCAN_SCRIPT])
        
        # We wait a bit for it to start processing
        return jsonify({
            "status": "success",
            "message": "Scan started. Check TFT and refresh gallery in 30s.",
            "mesh": "/data/mesh_4view.glb"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/data/<path:filename>')
def serve_data(filename):
    """Serves generated mesh files."""
    return send_from_directory(DATA_DIR, filename)

@app.route('/status')
def status():
    """Checks if a scan is currently running."""
    # Simple check for running processes
    return jsonify({"running": True}) # Placeholder

if __name__ == '__main__':
    # Kill any existing scan processes on startup
    os.system("pkill -f full_scan.py")
    app.run(host='0.0.0.0', port=5000, debug=False)
