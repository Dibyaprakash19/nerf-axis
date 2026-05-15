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

class MiDaSScanner:
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
        """Prepares image for MiDaS inference with optional background removal."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]

        if remove_bg:
            if HAS_REMBG:
                try:
                    pil_img = Image.fromarray(img_rgb)
                    no_bg = remove(pil_img)
                    black_bg = Image.new("RGB", no_bg.size, (0, 0, 0))
                    black_bg.paste(no_bg, mask=no_bg.split()[3])
                    img_rgb = np.array(black_bg)
                except Exception as e:
                    print(f"rembg failed, falling back to simple masking: {e}")
                    img_rgb = self._remove_bg_simple(img_rgb)
            else:
                img_rgb = self._remove_bg_simple(img_rgb)

        img_input = cv2.resize(img_rgb, (self.input_size, self.input_size))
        img_input = img_input.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_input = (img_input - mean) / std
        img_input = img_input[np.newaxis, ...].astype(np.float32)
        
        return img_input, img_rgb, (h_orig, w_orig)

    def _remove_bg_simple(self, img_rgb):
        """Lighter background removal using color thresholding (OpenCV)."""
        # Convert to HSV for better color-based segmentation
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        # Assuming background is not the same color as the object
        # This is a simple threshold that clears out very dark or very light areas
        # depending on your turntable setup. 
        # Here we just do a simple 'not too dark' mask as a placeholder.
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        
        # Apply mask
        result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        return result

    def get_depth(self, image_path, remove_bg=False):
        """Runs MiDaS inference and returns depth map and original RGB image."""
        img_input, img_rgb, (h_orig, w_orig) = self._preprocess(image_path, remove_bg)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], img_input)
        self.interpreter.invoke()
        depth = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        depth = cv2.resize(depth.astype(np.float32), (w_orig, h_orig))
        return depth, img_rgb

    def back_project(self, depth, img_rgb, step=4, scale=0.5, angle_deg=0):
        """Converts depth map to 3D vertices rotated by angle_deg, filtering background."""
        h, w = depth.shape
        d_s = depth[::step, ::step]
        c_s = img_rgb[::step, ::step]
        h_s, w_s = d_s.shape

        # Normalize depth
        d_min, d_max = d_s.min(), d_s.max()
        d_norm = (d_s - d_min) / (d_max - d_min + 1e-6)

        vertices = []
        colors = []
        # Index map to translate (r, col) to vertex index
        idx_map = {}
        
        rad = np.radians(angle_deg)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)

        v_idx = 0
        for r in range(h_s):
            for col in range(w_s):
                color = c_s[r, col]
                # Skip if background (black/very dark)
                if np.sum(color) < 15: # Threshold for 'black' background
                    continue
                
                # Center and scale coordinates
                x_c = col - w_s / 2.0
                y_c = -(r - h_s / 2.0)
                z_c = d_norm[r, col] * (w_s * scale)

                # Turntable rotation (around Y axis)
                x_rot = x_c * cos_a + z_c * sin_a
                z_rot = -x_c * sin_a + z_c * cos_a
                y_rot = y_c

                vertices.append([x_rot, y_rot, z_rot])
                colors.append(color)
                idx_map[(r, col)] = v_idx
                v_idx += 1

        return np.array(vertices, dtype=np.float32), np.array(colors, dtype=np.uint8), idx_map, (h_s, w_s)

    def generate_fused_mesh(self, image_paths, output_path, step=6, scale=0.55, remove_bg=True):
        """Fuses multiple views into a single GLB mesh with background filtering."""
        angles = [0, 90, 180, 270]
        all_v, all_c, all_f = [], [], []
        offset = 0

        for i, path in enumerate(image_paths[:4]):
            angle = angles[i]
            depth, img_rgb = self.get_depth(path, remove_bg=remove_bg)
            
            v, c, idx_map, (h_s, w_s) = self.back_project(depth, img_rgb, step=step, scale=scale, angle_deg=angle)
            
            # Create grid faces only for valid (non-filtered) vertices
            faces = []
            for r in range(h_s - 1):
                for col in range(w_s - 1):
                    # Check if all 4 corners of a grid cell exist
                    corners = [(r, col), (r, col+1), (r+1, col), (r+1, col+1)]
                    if all(p in idx_map for p in corners):
                        v0 = idx_map[(r, col)]
                        v1 = idx_map[(r, col+1)]
                        v2 = idx_map[(r+1, col)]
                        v3 = idx_map[(r+1, col+1)]
                        faces.extend([[v0, v2, v1], [v1, v2, v3]])
            
            if len(v) > 0:
                all_v.append(v)
                all_c.append(c)
                if faces:
                    all_f.append(np.array(faces) + offset)
                offset += len(v)

        if not all_v:
            print("Warning: No object detected in any view.")
            return None

        mesh = trimesh.Trimesh(
            vertices=np.vstack(all_v),
            faces=np.vstack(all_f) if all_f else None,
            vertex_colors=np.vstack(all_c),
            process=False # Avoid scikit-image and other heavy processing
        )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mesh.export(output_path)
        return output_path

    def depth_to_vertices(self, depth, rgb, view_angle=0):
        h, w = depth.shape
        # Subsample for speed if needed, but here we follow the request
        y, x = np.mgrid[0:h, 0:w]
        
        # Back-projection (simplified as requested)
        # Using a focal length approximation
        focal = 180.0
        X = (x - w/2) * depth / focal
        Y = (y - h/2) * depth / focal
        Z = depth
        
        vertices = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        
        # Apply view rotation
        rad = np.radians(view_angle)
        c, s = np.cos(rad), np.sin(rad)
        Xr = vertices[:,0] * c - vertices[:,2] * s
        Zr = vertices[:,0] * s + vertices[:,2] * c
        vertices[:,0] = Xr
        vertices[:,2] = Zr
        
        # Generate faces (grid)
        faces = []
        for r in range(h-1):
            for c in range(w-1):
                v0 = r*w + c
                v1 = v0 + 1
                v2 = (r+1)*w + c
                v3 = v2 + 1
                faces.extend([[v0,v2,v1], [v1,v2,v3]])
        
        return vertices.astype(np.float32), np.array(faces, dtype=np.int32)

    def process_image(self, image_path):
        """Wrapper for get_depth to match the new pipeline's naming."""
        return self.get_depth(image_path, remove_bg=False)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scanner.py <model> <image1> [image2...] <output.glb>")
        sys.exit(1)
        
    engine = MiDaSScanner(sys.argv[1])
    images = [p for p in sys.argv[2:] if not p.endswith('.glb')]
    output = sys.argv[-1] if sys.argv[-1].endswith('.glb') else 'output.glb'
    
    # generate_fused_mesh still exists but now uses MiDaSScanner name
    engine.generate_fused_mesh(images, output)
