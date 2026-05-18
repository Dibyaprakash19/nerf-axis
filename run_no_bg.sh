#!/bin/bash
# run_no_bg.sh
# Local orchestrator script to:
#  1. Sync the latest scanner code to the Raspberry Pi.
#  2. Stop the background autostart service (nerf_scan) and clear any hardware locks.
#  3. Trigger the 3D capture/reconstruction without background removal, retaining ImageOps.autocontrast preprocessing.
#  4. Automatically restart the autostart service on exit/Ctrl+C to return the Pi to its default state.

# Exit immediately if any command fails (except in cleanup)
set -e

# Resolve directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI_HOST="gp5.local"
PI_USER="gp"

# Cleanup function to return the Pi to its default state when this script exits or is interrupted (Ctrl+C)
cleanup() {
    # Disable the trap to prevent recursion
    trap - EXIT INT TERM
    echo -e "\n\033[1;36m=================================================================\033[0m"
    echo -e "\033[1;36m   Restoring Raspberry Pi to Autostart Scanner Loop...           \033[0m"
    echo -e "\033[1;36m=================================================================\033[0m"
    
    ssh "$PI_USER@$PI_HOST" "
        echo 'Starting autostart service (nerf_scan)...'
        sudo systemctl start nerf_scan || true
    "
    echo -e "\033[1;32m+++ Pi has successfully returned to its default autostart state.\033[0m"
}

# Register the cleanup function to run on SIGINT (Ctrl+C), SIGTERM, and normal exit
trap cleanup EXIT INT TERM

echo -e "\033[1;33m=================================================================\033[0m"
echo -e "\033[1;33m   NeRF-Axis: Raw 4-View Scanner Loop (No Background Removal)    \033[0m"
echo -e "\033[1;33m=================================================================\033[0m"

# 1. Sync files to the Pi first
echo "Syncing code to $PI_HOST..."
if [ -f "$SCRIPT_DIR/pi_scan/sync.sh" ]; then
    bash "$SCRIPT_DIR/pi_scan/sync.sh"
else
    echo "Warning: sync.sh not found in $SCRIPT_DIR/pi_scan/, skipping sync..."
fi

# 2. Control service and run script on Pi
echo -e "\nConnecting to $PI_HOST via SSH..."
# Using ssh -t for interactive tty output so keyboard interrupts (Ctrl+C) propagate cleanly
ssh -t "$PI_USER@$PI_HOST" "
    echo 'Stopping autostart service (nerf_scan)...'
    sudo systemctl stop nerf_scan || true
    
    echo 'Killing any stray camera/python processes...'
    sudo pkill -f python3 || true
    sudo pkill -f libcamera || true
    
    echo 'Starting scan without background removal (with autocontrast preprocessing)...'
    cd ~/pi_scan
    source venv/bin/activate
    python3 full_scan_no_bg.py
"
