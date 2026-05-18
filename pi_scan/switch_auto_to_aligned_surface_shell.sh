#!/usr/bin/env bash
set -euo pipefail

PI_USER="${PI_USER:-gp}"
PI_HOST="${PI_HOST:-}"

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

echo "Switching default automation to aligned 4-view surface-shell scanner on $PI_USER@$HOST"
ssh -t "$PI_USER@$HOST" "
    set -e
    sudo systemctl stop nerf_scan || true

    sudo sed -i 's|^ExecStart=.*|ExecStart=/usr/bin/python3 -m nerf_scan.main_4view_surface_shell_aligned|g' /etc/systemd/system/nerf_scan.service
    sudo sed -i 's|^WorkingDirectory=.*|WorkingDirectory=/home/gp/pi_scan|g' /etc/systemd/system/nerf_scan.service
    sudo sed -i 's|^Environment=.*|Environment=PYTHONPATH=/home/gp/pi_scan|g' /etc/systemd/system/nerf_scan.service

    sudo systemctl daemon-reload
    sudo systemctl enable nerf_scan
    sudo systemctl restart nerf_scan

    systemctl is-active nerf_scan
    systemctl cat nerf_scan | sed -n '1,40p'
"

echo "Default automation is now aligned 4-view surface-shell."
