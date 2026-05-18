#!/bin/bash

# Configuration
PI_HOST="gp5.local"
PI_USER="spass"
PI_PATH="~/nerf-axis/pi_scan"

# Help message
function show_help {
    echo "Usage: ./motor_control.sh [degrees]"
    echo "Default rotation is 90 degrees."
    echo "Example: ./motor_control.sh 180"
}

# If help requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Degrees (default 90)
DEGS=${1:-90}

echo "Rotating stepper motor on $PI_HOST by $DEGS degrees..."

# Try API first (if app is running)
RESPONSE=$(curl -s -X POST "http://$PI_HOST:5000/rotate" \
    -H "Content-Type: application/json" \
    -d "{\"degrees\": $DEGS}")

if [[ $RESPONSE == *"success"* ]]; then
    echo "Done (via Web API)."
else
    # Fallback to SSH script execution
    echo "Web API not responding, falling back to SSH..."
    ssh "$PI_USER@$PI_HOST" "cd $PI_PATH && python3 stepper90.py"
    echo "Done (via SSH script)."
fi
