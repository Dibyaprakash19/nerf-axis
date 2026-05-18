import os
import sys
import base64
import cv2
import numpy as np
from PIL import Image

try:
    import pygltflib
except ImportError:
    pygltflib = None

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from tensorflow import lite as tflite
    except ImportError:
        print("Error: Please install tflite-runtime or tensorflow to run inference.")
        sys.exit(1)


class MiDaSScanner:
    """Depth estimation engine using MiDaS small TFLite model with automated background removal."""

    def __init__(self, model_path: str = "midas_small.tflite"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found at: {model_path}")
            
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = self.input_details[0]["shape"][1]

    def remove_background(self, image_path: str, output_path: str = None) -> np.ndarray:
        """
        Segment the foreground object from a turntable scan using contours.
        Returns the segmented RGBA image.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from path: {image_path}")

        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)

        # Combine Otsu's threshold with a fixed threshold to capture darker edges
        _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh_fixed = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_or(thresh_otsu, thresh_fixed)

        # Close gaps in the shape and dilate slightly
        kernel = np.ones((9, 9), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)

        # Create foreground mask using the largest contour
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros((height, width), np.uint8)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            # Dilate mask to prevent cutting off fine edges of the object
            mask = cv2.dilate(mask, np.ones((20, 20), np.uint8), iterations=2)
        else:
            mask = np.ones((height, width), np.uint8) * 255

        result = cv2.bitwise_and(img, img, mask=mask)
        blue, green, red = cv2.split(result)
        rgba = cv2.merge((blue, green, red, mask))

        if output_path:
            cv2.imwrite(output_path, rgba)

        return rgba

    def process_image(self, source) -> tuple:
        """
        Run depth estimation on an image path or pre-loaded NumPy array.
        Returns:
            depth: (H, W) float32 array normalized to 0.0 - 2.5
            rgb:   (H, W, 3) uint8 RGB image
        """
        if isinstance(source, str):
            img = cv2.imread(source)
        else:
            img = source

        if img is None:
            raise ValueError("Input image is empty or invalid.")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = cv2.resize(rgb, (self.input_size, self.input_size))
        input_data = np.expand_dims(input_data.astype(np.float32) / 255.0, axis=0)

        # Run model inference
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        depth = self.interpreter.get_tensor(self.output_details[0]["index"])
        depth = np.squeeze(depth)

        # Post-process depth map
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))
        depth = np.clip(depth, 0.01, 1.0)
        
        # Invert depth (MiDaS output is inverse, where closer points have larger depth values)
        depth = 1.0 - depth
        
        # Scale range to 0 - 1
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        
        # Scale factor for turntable scans
        depth = depth * 2.5

        return depth.astype(np.float32), rgb

    def depth_to_vertices(self, depth: np.ndarray, rgb: np.ndarray, angle_degrees: float = 0.0) -> tuple:
        """
        Project depth values back to 3D coordinate space and rotate to current turntable angle.
        Returns:
            vertices: (N, 3) float32 coordinates
            colors:   (N, 3) float32 colors normalized to 0.0 - 1.0
        """
        h, w = depth.shape
        y, x = np.mgrid[0:h, 0:w]

        # Standard perspective back-projection
        focal_length = 160.0
        x_coords = (x - w / 2) * depth / focal_length
        y_coords = (y - h / 2) * depth / focal_length
        z_coords = depth * 1.8

        vertices = np.stack([x_coords.ravel(), y_coords.ravel(), z_coords.ravel()], axis=1)

        # Apply rotation around vertical Y-axis
        rad = np.radians(angle_degrees)
        cos_val, sin_val = np.cos(rad), np.sin(rad)
        rot_x = vertices[:, 0] * cos_val - vertices[:, 2] * sin_val
        rot_z = vertices[:, 0] * sin_val + vertices[:, 2] * cos_val
        vertices[:, 0] = rot_x
        vertices[:, 2] = rot_z

        # Center vertices to keep alignment between different views
        vertices -= np.mean(vertices, axis=0)
        colors = rgb.reshape(-1, 3) / 255.0

        return vertices.astype(np.float32), colors

    def create_faces(self, height: int, width: int) -> np.ndarray:
        """Generate triangulation faces for a regular 2D grid of size height x width."""
        faces = []
        for r in range(height - 1):
            for c in range(width - 1):
                v0 = r * width + c
                v1 = v0 + 1
                v2 = (r + 1) * width + c
                v3 = v2 + 1
                faces.extend([[v0, v2, v1], [v1, v2, v3]])
        return np.array(faces, dtype=np.int32)


def fuse_views(views: list) -> tuple:
    """
    Merge multiple views of (vertices, faces, colors) into a single coordinate system.
    """
    all_vertices, all_faces, all_colors = [], [], []
    index_offset = 0

    for verts, faces, colors in views:
        all_vertices.append(verts)
        all_colors.append(colors)
        if len(faces) > 0:
            all_faces.append(faces + index_offset)
        index_offset += len(verts)

    fused_vertices = np.vstack(all_vertices)
    fused_faces = np.vstack(all_faces) if all_faces else np.zeros((0, 3), np.int32)
    fused_colors = np.vstack(all_colors)

    # Re-center and scale fused mesh to fit rendering boundaries
    fused_vertices -= fused_vertices.mean(axis=0)
    max_bound = np.abs(fused_vertices).max()
    if max_bound > 0:
        fused_vertices *= (50.0 / max_bound)

    return fused_vertices.astype(np.float32), fused_faces, fused_colors.astype(np.uint8)


def save_glb(vertices: np.ndarray, faces: np.ndarray, output_path: str) -> bool:
    """
    Export vertices and faces to a binary GLB file using pygltflib.
    """
    if pygltflib is None:
        print("Warning: pygltflib is not installed, skipping GLB export.")
        return False

    try:
        from pygltflib import GLTF2, Asset, Scene, Node, Mesh, Primitive, Accessor, BufferView, Buffer, Attributes
        
        v = vertices.astype(np.float32)
        f = faces.astype(np.uint32)

        v_bytes = v.tobytes()
        f_bytes = f.tobytes()
        blob = f_bytes + v_bytes

        gltf = GLTF2(
            asset=Asset(version="2.0"),
            scenes=[Scene(nodes=[0])],
            nodes=[Node(mesh=0)],
            meshes=[Mesh(primitives=[
                Primitive(attributes=Attributes(POSITION=1), indices=0)
            ])],
            accessors=[
                Accessor(bufferView=0, componentType=5125, count=len(f) * 3, type="SCALAR"),
                Accessor(bufferView=1, componentType=5126, count=len(v), type="VEC3",
                         max=v.max(axis=0).tolist(), min=v.min(axis=0).tolist()),
            ],
            bufferViews=[
                BufferView(buffer=0, byteOffset=0, byteLength=len(f_bytes), target=34963),
                BufferView(buffer=0, byteOffset=len(f_bytes), byteLength=len(v_bytes), target=34962),
            ],
            buffers=[Buffer(
                uri="data:application/octet-stream;base64," + base64.b64encode(blob).decode(),
                byteLength=len(blob)
            )],
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        gltf.save(output_path)
        print(f"Mesh successfully saved to {output_path}")
        return True
    except Exception as e:
        print(f"Failed to export GLB mesh: {e}")
        return False
