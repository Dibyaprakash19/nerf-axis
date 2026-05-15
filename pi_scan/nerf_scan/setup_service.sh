#!/bin/bash
# nerf_scan/setup_service.sh
# Sets up the 3D scanner as a standalone systemd service.

SERVICE_NAME="nerf_scan"
USER_NAME=$(whoami)
BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN="/usr/bin/python3"

echo "[setup] Creating systemd service for $USER_NAME at $BASE_DIR"

cat <<EOF | sudo tee /etc/systemd/system/$SERVICE_NAME.service
[Unit]
Description=NeRF-Axis Autonomous 3D Scanner
After=network.target

[Service]
ExecStart=$PYTHON_BIN -m nerf_scan.main
WorkingDirectory=$BASE_DIR
StandardOutput=inherit
StandardError=inherit
Restart=always
User=$USER_NAME
Environment=PYTHONPATH=$BASE_DIR

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl restart $SERVICE_NAME

echo "[setup] Done. Scanner is now running and will start on power-on."
echo "[setup] Check status: sudo systemctl status $SERVICE_NAME"
echo "[setup] View logs: journalctl -u $SERVICE_NAME -f"
