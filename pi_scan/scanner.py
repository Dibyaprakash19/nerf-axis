import os
import sys
import cv2
import numpy as np
import trimesh
from PIL import Image

# MiDaS dependencies
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from tensorflow import lite as tflite
    except ImportError:
        print("Error: MiDaS requires tflite-runtime or tensorflow.")
        sys.exit(1)

# Background removal dependency
try:
    from rembg import remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False

class DepthEngine:
    """
    Handles MiDaS depth inference and 3D point cloud / mesh generation.
    """
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = self.input_details[0]['shape'][1]

    def _preprocess(self, image_path, remove_bg=False):
        """Prepares image for MiDaS inference."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]

        if HAS_REMBG and remove_bg:
            pil_img = Image.fromarray(img_rgb)
            no_bg = remove(pil_img)
            black_bg = Image.new("RGB", no_bg.size, (0, 0, 0))
            black_bg.paste(no_bg, mask=no_bg.split()[3])
            img_rgb = np.array(black_bg)

        img_input = cv2.resize(img_rgb, (self.input_size, self.input_size))
        img_input = img_input.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_input = (img_input - mean) / std
        img_input = img_input[np.newaxis, ...].astype(np.float32)
        
        return img_input, img_rgb, (h_orig, w_orig)

    def get_depth(self, image_path, remove_bg=False):
        """Runs MiDaS inference and returns depth map and original RGB image."""
        img_input, img_rgb, (h_orig, w_orig) = self._preprocess(image_path, remove_bg)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], img_input)
        self.interpreter.invoke()
        depth = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        depth = cv2.resize(depth.astype(np.float32), (w_orig, h_orig))
        return depth, img_rgb

    def back_project(self, depth, img_rgb, step=4, scale=0.5, angle_deg=0):
        """Converts depth map to 3D vertices rotated by angle_deg."""
        h, w = depth.shape
        d_s = depth[::step, ::step]
        c_s = img_rgb[::step, ::step]
        h_s, w_s = d_s.shape

        x, y = np.meshgrid(np.arange(w_s), np.arange(h_s))
        
        # Normalize depth
        d_min, d_max = d_s.min(), d_s.max()
        d_norm = (d_s - d_min) / (d_max - d_min + 1e-6)

        # Center coordinates
        x_c = x.ravel() - w_s / 2.0
        y_c = -(y.ravel() - h_s / 2.0)
        z_c = d_norm.ravel() * (w_s * scale)

        # Turntable rotation (around Y axis)
        rad = np.radians(angle_deg)
        x_rot = x_c * np.cos(rad) + z_c * np.sin(rad)
        z_rot = -x_c * np.sin(rad) + z_c * np.cos(rad)
        y_rot = y_c

        vertices = np.stack([x_rot, y_rot, z_rot], axis=1).astype(np.float32)
        colors = c_s.reshape(-1, 3)
        return vertices, colors

    def generate_fused_mesh(self, image_paths, output_path, step=6, scale=0.55, remove_bg=True):
        """Fuses multiple views into a single GLB mesh with optional background removal."""
        angles = [0, 90, 180, 270]
        all_v, all_c, all_f = [], [], []
        offset = 0

        for i, path in enumerate(image_paths[:4]):
            angle = angles[i]
            # Use remove_bg=True to extract object and remove background
            depth, img_rgb = self.get_depth(path, remove_bg=remove_bg)
            
            # Subsample for mesh generation
            d_s = depth[::step, ::step]
            h_s, w_s = d_s.shape
            
            v, c = self.back_project(depth, img_rgb, step=step, scale=scale, angle_deg=angle)
            
            # Create grid faces (only for points that aren't background if remove_bg is on)
            # Simple approach: filter out vertices where depth is too low/zero if needed,
            # but for now, we'll keep the grid and let the renderer handle the black pixels.
            faces = []
            for r in range(h_s - 1):
                for col in range(w_s - 1):
                    v0 = r * w_s + col
                    v1 = v0 + 1
                    v2 = (r + 1) * w_s + col
                    v3 = v2 + 1
                    
                    # Optional: skip faces where all vertices are background (black)
                    # For simplicity, we keep them for now as rembg makes background black.
                    faces.extend([[v0, v2, v1], [v1, v2, v3]])
            
            all_v.append(v)
            all_c.append(c)
            all_f.append(np.array(faces) + offset)
            offset += len(v)

        mesh = trimesh.Trimesh(
            vertices=np.vstack(all_v),
            faces=np.vstack(all_f),
            vertex_colors=np.vstack(all_c),
            process=False
        )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mesh.export(output_path)
        return output_path

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scanner.py <model> <image1> [image2...] <output.glb>")
        sys.exit(1)
        
    engine = DepthEngine(sys.argv[1])
    images = [p for p in sys.argv[2:] if not p.endswith('.glb')]
    output = sys.argv[-1] if sys.argv[-1].endswith('.glb') else 'output.glb'
    
    engine.generate_fused_mesh(images, output)
