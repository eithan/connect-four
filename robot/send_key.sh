#!/usr/bin/env bash
# =============================================================
#  send_key.sh
#  Run this in a SECOND terminal while record_first_dataset.sh
#  is running. Press Enter here to send a Right Arrow keystroke
#  to lerobot (ends current phase early).
#
#  PREREQS:  sudo apt install xdotool
#
#  USAGE:
#    Terminal 1: ./robot/record_first_dataset.sh
#    Terminal 2: ./robot/send_key.sh
# =============================================================

set -euo pipefail

command -v xdotool &>/dev/null || {
    echo "ERROR: xdotool not found. Run: sudo apt install xdotool"
    exit 1
}

echo "Ready. Press ENTER to send Right Arrow to lerobot (end phase early)."
echo "Press Ctrl-C to quit."
echo ""

while IFS= read -r; do
    xdotool key Right
    echo "  → Right Arrow sent  $(date +%H:%M:%S)"
done
