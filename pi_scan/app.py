from flask import Flask, render_template, send_from_directory, jsonify
import os
import subprocess

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'midas_small.tflite')
SAMPLE_IMG = os.path.join(BASE_DIR, 'sample.jpg')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

@app.route('/')
@app.route('/mesh')
def index():
    return render_template('mesh.html')

@app.route('/scan')
def scan():
    output_mesh = os.path.join(DATA_DIR, 'mesh.glb')
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not found. Run download_model.py first."}), 500
    
    # Check if sample image exists
    if not os.path.exists(SAMPLE_IMG):
        # Create a dummy or download one? For now, error.
        return jsonify({"error": "Sample image not found."}), 500

    # Run scanner.py as a subprocess to keep Flask light
    try:
        cmd = [
            os.path.join(BASE_DIR, 'venv/bin/python'),
            os.path.join(BASE_DIR, 'scanner.py'),
            SAMPLE_IMG,
            MODEL_PATH,
            output_mesh
        ]
        subprocess.run(cmd, check=True)
        return jsonify({"status": "success", "mesh": "/data/mesh.glb"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory(DATA_DIR, filename)

if __name__ == '__main__':
    # Listen on all interfaces
    app.run(host='0.0.0.0', port=5000, debug=False)
