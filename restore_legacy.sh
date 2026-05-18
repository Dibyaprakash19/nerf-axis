#!/bin/bash
# switch the Raspberry Pi back to the original TSDF pipeline

PI_HOST="gp5.local"
PI_USER="gp"

echo "Restoring legacy TSDF pipeline on Raspberry Pi ($PI_HOST)..."

ssh $PI_USER@$PI_HOST "sudo sed -i 's|ExecStart=.*|ExecStart=/usr/bin/python3 /home/gp/pi_scan/nerf_scan/main.py|g' /etc/systemd/system/nerf_scan.service && sudo systemctl daemon-reload && sudo systemctl restart nerf_scan"

echo "Successfully restored original pipeline!"
