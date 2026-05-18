#!/bin/bash

echo "Setting up OpenMVS capture and pipeline..."

SERVICE_FILE="/etc/systemd/system/nerf_scan.service"
SCRIPT_PATH="$(pwd)/capture_mvs.py"
PYTHON_BIN="/usr/bin/python3"

chmod +x $SCRIPT_PATH

if [ -f "$SERVICE_FILE" ]; then
    echo "Updating systemd service..."
    sudo sed -i "s|ExecStart=.*|ExecStart=$PYTHON_BIN $SCRIPT_PATH|g" $SERVICE_FILE
    sudo systemctl daemon-reload
    sudo systemctl restart nerf_scan
    echo "Service updated and restarted with MVS pipeline!"
else
    echo "Service file not found. Running standalone."
    $PYTHON_BIN $SCRIPT_PATH
fi
