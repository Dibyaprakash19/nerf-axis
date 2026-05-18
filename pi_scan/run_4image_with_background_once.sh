#!/usr/bin/env bash
set -euo pipefail

PI_USER="${PI_USER:-gp}"
PI_HOST="${PI_HOST:-}"
PI_DIR="${PI_DIR:-~/pi_scan}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$HERE/results/with_background_latest"

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
        pkill -f '[p]ython3 full_scan_no_bg.py' 2>/dev/null || true
        sudo systemctl start nerf_scan || true
    " || true

    mkdir -p "$RESULTS_DIR"
    rsync -az "$PI_USER@$HOST:$PI_DIR/data/live_scan_no_bg/" "$RESULTS_DIR/" >/dev/null 2>&1 || true
    echo "Local result folder: $RESULTS_DIR"
}

trap cleanup EXIT INT TERM

echo "Running 4-image MiDaS scan WITH BACKGROUND preserved on $PI_USER@$HOST"
echo "This does not edit systemd. It only pauses the service while this manual scan runs."

echo "Syncing current local files to Pi..."
"$HERE/sync.sh"

ssh "$PI_USER@$HOST" "
    echo 'Stopping automated service for manual test...'
    sudo systemctl stop nerf_scan || true
    pkill -f '[p]ython3 tft_cam.py' 2>/dev/null || true
    pkill -f '[p]ython3 full_scan_no_bg.py' 2>/dev/null || true
"

ssh -t "$PI_USER@$HOST" "
    echo 'Starting 4-image background-preserved scan.'
    echo 'MiDaS receives the full image; no background removal before depth.'
    cd $PI_DIR
    . venv/bin/activate
    exec python3 full_scan_no_bg.py
"
