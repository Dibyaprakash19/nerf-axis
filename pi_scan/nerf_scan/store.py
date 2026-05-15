"""
nerf_scan/store.py — Scan data persistence.

Responsibilities:
  - Clean up previous scan artifacts before a new scan
  - Export mesh to GLB (pygltflib, optional) with vertex colors
  - Fall back to compressed NPZ if pygltflib unavailable
  - Return the GLB path so the web server can serve it

Usage:
    from nerf_scan import store
    store.cleanup()                              # delete old data
    path = store.save(verts, faces, colors)      # write GLB/NPZ
    ok   = store.has_mesh()                      # check if GLB ready
"""

import os, glob, shutil, base64
import numpy as np
from .config import DATA_DIR, GLB_PATH


def cleanup():
    """Remove all files in DATA_DIR (old scan data)."""
    if os.path.isdir(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)


def has_mesh() -> bool:
    return os.path.isfile(GLB_PATH)


def save(verts: np.ndarray, faces: np.ndarray,
         colors: np.ndarray, path: str = GLB_PATH) -> str:
    """
    Persist fused mesh.  Tries GLB first; falls back to NPZ.
    Returns the path of the written file.

    GLB includes:
      - POSITION accessor  (float32 VEC3)
      - indices accessor   (uint32 SCALAR)
      - COLOR_0 accessor   (float32 VEC3 — vertex RGB 0..1)
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        _save_glb(verts, faces, colors, path)
        print(f"[store] GLB → {path}  ({len(verts)} verts, {len(faces)} faces)")
        return path
    except Exception as e:
        print(f"[store] GLB failed ({e}), saving NPZ")
        npz = path.replace(".glb", ".npz")
        np.savez_compressed(npz, verts=verts, faces=faces, colors=colors)
        print(f"[store] NPZ → {npz}")
        return npz


# ── GLB writer ────────────────────────────────────────────────────────────────

def _save_glb(verts: np.ndarray, faces: np.ndarray,
              colors: np.ndarray, path: str):
    from pygltflib import (GLTF2, Asset, Scene, Node, Mesh, Primitive,
                           Accessor, BufferView, Buffer, Attributes)

    v  = verts.astype(np.float32)
    f  = faces.astype(np.uint32)
    c  = (colors.astype(np.float32) / 255.0)   # normalize → VEC3 FLOAT

    v_bytes = v.tobytes()
    f_bytes = f.tobytes()
    c_bytes = c.tobytes()
    blob    = f_bytes + v_bytes + c_bytes

    v_off = len(f_bytes)
    c_off = v_off + len(v_bytes)

    gltf = GLTF2(
        asset=Asset(version="2.0"),
        scenes=[Scene(nodes=[0])],
        nodes=[Node(mesh=0)],
        meshes=[Mesh(primitives=[
            Primitive(
                attributes=Attributes(POSITION=1, COLOR_0=2),
                indices=0,
            )
        ])],
        accessors=[
            Accessor(bufferView=0, componentType=5125,        # UNSIGNED_INT
                     count=len(f) * 3, type="SCALAR"),
            Accessor(bufferView=1, componentType=5126,        # FLOAT
                     count=len(v), type="VEC3",
                     max=v.max(axis=0).tolist(),
                     min=v.min(axis=0).tolist()),
            Accessor(bufferView=2, componentType=5126,        # FLOAT
                     count=len(c), type="VEC3",
                     max=[1.0, 1.0, 1.0], min=[0.0, 0.0, 0.0]),
        ],
        bufferViews=[
            BufferView(buffer=0, byteOffset=0,
                       byteLength=len(f_bytes), target=34963),    # ELEMENT_ARRAY
            BufferView(buffer=0, byteOffset=v_off,
                       byteLength=len(v_bytes), target=34962),    # ARRAY_BUFFER
            BufferView(buffer=0, byteOffset=c_off,
                       byteLength=len(c_bytes), target=34962),
        ],
        buffers=[Buffer(
            uri="data:application/octet-stream;base64,"
                + base64.b64encode(blob).decode(),
            byteLength=len(blob),
        )],
    )
    gltf.save(path)
