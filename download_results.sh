#!/bin/bash
# Run this from your Mac to pull all the latest 3D renders and sample images from the Raspberry Pi

PI_HOST="gp5.local"
PI_USER="gp"
RESULTS_DIR="$(pwd)/results"

echo "Pulling 3D renders and samples from Pi to $RESULTS_DIR..."

mkdir -p "$RESULTS_DIR/mvs"
mkdir -p "$RESULTS_DIR/colmap"
mkdir -p "$RESULTS_DIR/midas"

# 1. Pull MVS results
echo "Pulling MVS results..."
rsync -avP $PI_USER@$PI_HOST:~/pi_scan/scan/scene_dense_mesh_refine.mvs "$RESULTS_DIR/mvs/" 2>/dev/null
rsync -avP $PI_USER@$PI_HOST:~/pi_scan/scan/images/ "$RESULTS_DIR/mvs/" 2>/dev/null

# 2. Pull COLMAP results
echo "Pulling COLMAP results..."
rsync -avP $PI_USER@$PI_HOST:~/pi_scan/scan/mesh.ply "$RESULTS_DIR/colmap/" 2>/dev/null
rsync -avP $PI_USER@$PI_HOST:~/pi_scan/scan/images/ "$RESULTS_DIR/colmap/" 2>/dev/null

# 3. Pull legacy TSDF/MiDaS results
echo "Pulling MiDaS results..."
rsync -avP $PI_USER@$PI_HOST:~/pi_scan/nerf_scan/data/current_mesh.glb "$RESULTS_DIR/midas/" 2>/dev/null
rsync -avP $PI_USER@$PI_HOST:~/pi_scan/original.jpg "$RESULTS_DIR/midas/" 2>/dev/null
rsync -avP $PI_USER@$PI_HOST:~/pi_scan/no_bg.jpg "$RESULTS_DIR/midas/" 2>/dev/null

echo "Sync complete! Check the $RESULTS_DIR folder."
