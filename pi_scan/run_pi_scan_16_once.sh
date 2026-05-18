#!/usr/bin/env bash
set -euo pipefail

PI_HOST="${PI_HOST:-gp5.local}"
PI_USER="${PI_USER:-gp}"
PI_DIR="${PI_DIR:-~/pi_scan}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$HERE/results/scan16_latest"
KEEP_WEB_SECONDS="${KEEP_WEB_SECONDS:-600}"

echo "[1/5] Syncing local code to $PI_USER@$PI_HOST:$PI_DIR"
"$HERE/sync.sh"

echo "[2/5] Stopping auto-scan service if it is running"
ssh "$PI_USER@$PI_HOST" "sudo -n systemctl stop nerf_scan 2>/dev/null || true; pgrep -f '[p]ython3 -m nerf_scan.main' | xargs -r kill || true; pgrep -f '[p]ython3 -m nerf_scan.main_16' | xargs -r kill || true"

echo "[3/5] Running one 16-view scan on the Pi"
ssh "$PI_USER@$PI_HOST" "cd $PI_DIR && source venv/bin/activate && python3 -m nerf_scan.main_16 --views 16 --keep-web-seconds $KEEP_WEB_SECONDS" &
SCAN_PID=$!

echo "[4/5] Waiting for the Pi process to finish or be stopped"
wait "$SCAN_PID" || true

echo "[5/5] Pulling latest mesh data back to this Mac"
mkdir -p "$RESULTS_DIR"
rsync -avz --delete "$PI_USER@$PI_HOST:$PI_DIR/nerf_scan/data/" "$RESULTS_DIR/"
echo "Mesh: $RESULTS_DIR/current_mesh.glb"
echo "Web UI during preview: http://$PI_HOST:8000/"
