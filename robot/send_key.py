#!/usr/bin/env python3
"""
send_key.py
===========
Run in a SECOND terminal while record_first_dataset.sh runs.

lerobot uses pynput, which on RDP/Wayland sessions uses the X11 backend.
This script reads lerobot's DISPLAY env var from /proc and uses xdotool to
inject X11 key events — which pynput sees globally, regardless of focus.

PREREQS:
    sudo apt install xdotool

USAGE:
    python3 robot/send_key.py

CONTROLS:
    ENTER        → Right Arrow  (start episode / end episode)
    d + ENTER    → Left Arrow   (discard current episode)
    Ctrl-C       → quit
"""

import os
import subprocess
import sys


def get_lerobot_display():
    """Read DISPLAY from lerobot-record's /proc environ."""
    try:
        result = subprocess.check_output(
            ['pgrep', '-f', 'lerobot-record'], text=True)
        pids = result.strip().split()
    except subprocess.CalledProcessError:
        return None, None

    for pid in pids:
        try:
            with open(f'/proc/{pid}/environ', 'rb') as f:
                for var in f.read().split(b'\x00'):
                    if var.startswith(b'DISPLAY='):
                        return pid, var.split(b'=', 1)[1].decode()
        except (OSError, PermissionError):
            continue

    # Fall back to our own DISPLAY if lerobot isn't found yet
    return None, os.environ.get('DISPLAY')


def send_key(key_name):
    pid, display = get_lerobot_display()
    if pid is None:
        print("  ERROR: lerobot-record not found — is it running?")
        return False
    if not display:
        print("  ERROR: could not determine DISPLAY from lerobot's environment")
        return False

    env = os.environ.copy()
    env['DISPLAY'] = display
    try:
        subprocess.run(
            ['xdotool', 'key', key_name],
            env=env, check=True, capture_output=True)
        return True
    except FileNotFoundError:
        print("  ERROR: xdotool not found.  Install: sudo apt install xdotool")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: xdotool failed: {e.stderr.decode().strip()}")
        return False


# ── Startup ────────────────────────────────────────────────────────────────────
print("Waiting for lerobot-record to start...", end=' ', flush=True)
_, display = get_lerobot_display()
if display:
    print(f"found  (DISPLAY={display})")
else:
    print("not running yet — key injection will detect it on first press")

print()
print("  ENTER  → Right Arrow  (start / end episode)")
print("  d      → Left Arrow   (discard episode)")
print("  Ctrl-C → quit")
print()

# ── Main loop ──────────────────────────────────────────────────────────────────
while True:
    try:
        cmd = input()
    except (EOFError, KeyboardInterrupt):
        print("\nDone.")
        break

    if cmd.strip().lower() == 'd':
        if send_key('Left'):
            print("  ← Left Arrow sent (discard)")
    else:
        if send_key('Right'):
            print("  → Right Arrow sent (start/end)")
