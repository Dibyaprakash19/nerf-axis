import cv2
import numpy as np
import os
import sys

# Try to import tflite-runtime, fallback to tensorflow.lite
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from tensorflow import lite as tflite
    except ImportError:
        print("Error: tflite-runtime or tensorflow not found.")
        sys.exit(1)

import trimesh

class MiDaSScanner:
    def __init__(self, model_path):
        self.model_path = model_path
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = self.input_details[0]['shape'][1]

    def process_image(self, image_path):
        # Load and prep image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_size, self.input_size))
        
        # Normalize
        img_input = img_resized.astype(np.float32) / 255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_input = (img_input - mean) / std
        img_input = img_input[np.newaxis, ...].astype(np.float32)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], img_input)
        self.interpreter.invoke()
        depth = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # Resize depth back to original
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]))
        return depth, img_rgb

    def generate_mesh(self, depth, img_rgb, output_path, scale=0.5):
        h, w = depth.shape
        # Subsample to keep Pi happy (don't create millions of vertices)
        step = 4 
        depth_small = depth[::step, ::step]
        img_small = img_rgb[::step, ::step]
        h_s, w_s = depth_small.shape

        # Create grid
        x, y = np.meshgrid(np.arange(w_s), np.arange(h_s))
        
        # Scale depth
        d_min, d_max = depth_small.min(), depth_small.max()
        depth_norm = (depth_small - d_min) / (d_max - d_min + 1e-6)
        
        # MiDaS gives inverse depth (large value = close)
        # We want Z to be distance
        z = depth_norm * (w_s * scale)

        # Vertices and Colors
        vertices = np.stack([x.ravel(), -y.ravel(), z.ravel()], axis=1)
        colors = img_small.reshape(-1, 3)

        # Faces
        faces = []
        for i in range(h_s - 1):
            for j in range(w_s - 1):
                v0 = i * w_s + j
                v1 = v0 + 1
                v2 = (i + 1) * w_s + j
                v3 = v2 + 1
                faces.append([v0, v2, v1])
                faces.append([v1, v2, v3])
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
        mesh.export(output_path)
        return output_path

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scanner.py <image_path> <model_path> [output_path]")
        sys.exit(1)
    
    img_p = sys.argv[1]
    mod_p = sys.argv[2]
    out_p = sys.argv[3] if len(sys.argv) > 3 else "data/mesh.glb"
    
    scanner = MiDaSScanner(mod_p)
    depth, img = scanner.process_image(img_p)
    scanner.generate_mesh(depth, img, out_p)
    print(f"Mesh saved to {out_p}")
