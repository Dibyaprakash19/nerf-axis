#!/usr/bin/env bash
set -euo pipefail

PI_USER="${PI_USER:-gp}"
PI_HOST="${PI_HOST:-}"
PI_DIR="${PI_DIR:-~/pi_scan}"

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

echo "Pausing 4-image autoscanner on $PI_USER@$HOST"
ssh "$PI_USER@$HOST" "
    echo 'Stopping nerf_scan service...'
    sudo systemctl stop nerf_scan || true

    echo 'Killing only known old preview/scanner processes...'
    pkill -f '[p]ython3 tft_cam.py' 2>/dev/null || true
    pkill -f '[p]ython3 full_scan.py' 2>/dev/null || true
    pkill -f '[p]ython3 full_scan_no_bg.py' 2>/dev/null || true
    pkill -f '[p]ython3 -m nerf_scan.main_16' 2>/dev/null || true
"

ssh -t "$PI_USER@$HOST" "
    echo 'Starting live camera preview on TFT. Press Ctrl+C to stop preview.'
    cd $PI_DIR
    . venv/bin/activate
    exec python3 tft_cam.py
"

echo "Preview stopped. The autoscanner is still paused."
echo "Run ./restart_4image_auto.sh to restore the automated 4-image scanner."
