#!/usr/bin/env bash
set -euo pipefail

PI_USER="${PI_USER:-gp}"
PI_HOST="${PI_HOST:-}"
PI_DIR="${PI_DIR:-~/pi_scan}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$HERE/results/surface_shell_aligned_latest"

pick_host() {
    if [ -n "$PI_HOST" ]; then
        echo "$PI_HOST"
        return
    fi
    for host in gp5.local 192.168.137.128; do
        if ssh -o ConnectTimeout=5 -o BatchMode=yes "$PI_USER@$host" "true" >/dev/null 2>&1; then
            echo "$host"
            return
        fi
    done
    echo "gp5.local"
}

HOST="$(pick_host)"

cleanup() {
    trap - EXIT INT TERM
    echo
    echo "Restoring 4-image automated scanner service on $PI_USER@$HOST..."
    ssh "$PI_USER@$HOST" "
        pkill -f '[p]ython3 -m nerf_scan.main_4view_surface_shell_aligned' 2>/dev/null || true
        sudo systemctl start nerf_scan || true
    " || true

    mkdir -p "$RESULTS_DIR"
    rsync -az "$PI_USER@$HOST:$PI_DIR/data/surface_shell_aligned/" "$RESULTS_DIR/" >/dev/null 2>&1 || true
    echo "Local result folder: $RESULTS_DIR"
}

trap cleanup EXIT INT TERM

echo "Running aligned 4-view MiDaS surface-shell mesh."
echo "Uses clockwise capture order and seam strips between neighboring views."
echo "This does not modify systemd/autostart."

"$HERE/sync.sh"

ssh "$PI_USER@$HOST" "
    sudo systemctl stop nerf_scan || true
    pkill -f '[p]ython3 tft_cam.py' 2>/dev/null || true
    pkill -f '[p]ython3 -m nerf_scan.main_4view_surface_shell_aligned' 2>/dev/null || true
"

ssh -t "$PI_USER@$HOST" "
    cd $PI_DIR
    . venv/bin/activate
    exec python3 -m nerf_scan.main_4view_surface_shell_aligned
"
