#!/bin/bash

# Configuration
PI_HOST="gp5.local"
PI_USER="gp"
DEST_PATH="~/pi_scan"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Syncing $SCRIPT_DIR/ to $PI_USER@$PI_HOST:$DEST_PATH..."

ssh "$PI_USER@$PI_HOST" "mkdir -p $DEST_PATH"

# Sync using rsync for efficiency
# -a: archive mode
# -v: verbose
# -z: compress
# --exclude: ignore certain patterns
rsync -avz \
  --exclude 'venv' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '*.tflite' \
  --exclude 'data/*.glb' \
  --exclude 'data/images' \
  --exclude 'data/live_scan' \
  "$SCRIPT_DIR/" "$PI_USER@$PI_HOST:$DEST_PATH/"

if [ -f "$SCRIPT_DIR/../context.md" ]; then
  rsync -avz "$SCRIPT_DIR/../context.md" "$PI_USER@$PI_HOST:$DEST_PATH/context.md"
fi

if [ $? -eq 0 ]; then
    echo "Sync successful!"
else
    echo "Sync failed. Check your SSH connection to $PI_HOST."
fi
