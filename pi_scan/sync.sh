#!/bin/bash
# sync.sh — High-performance sync script with progress tracking.

# ── Configuration ─────────────────────────────────────────────────────────────
PI_HOST="gp5.local"
PI_USER="gp"
DEST_PATH="~/pi_scan"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GREEN='\033[0;32m'
RED='\033[0;31m'

# ── Sync Engine ───────────────────────────────────────────────────────────────
# Flags:
#  -a: Archive mode (perms, times, symlinks)
#  -v: Verbose
#  -z: Compression (fast for text/code)
#  -P: --partial (keep partially transferred files) + --progress (speed/ETA)
#  --delete: Cleanup files on Pi that no longer exist locally (keeps it clean)

echo " Syncing NeRF-Axis pipeline to $PI_HOST"

rsync -aviP --stats \
  --exclude 'venv' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '*.tflite' \
  --exclude 'data/' \
  --exclude '*.glb' \
  --exclude 'context.md' \
  "$SRC_DIR/" "$PI_USER@$PI_HOST:$DEST_PATH/"

EXIT_CODE=$?

# Optional: Sync context.md if it exists in parent
if [ -f "$SRC_DIR/../context.md" ]; then
  rsync -az "$SRC_DIR/../context.md" "$PI_USER@$PI_HOST:$DEST_PATH/context.md"
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}+++ Sync Complete. System is consistent."
else
    echo -e "\n${RED}--- Sync Failed. Check SSH connection to $PI_HOST."
fi
