#!/usr/bin/env python3
"""
send_key.py
===========
Run in a SECOND terminal while record_first_dataset.sh runs.
Press ENTER to send a Right Arrow keystroke to lerobot.

Uses kernel-level evdev injection — works on Wayland.

PREREQS:
    pip install evdev
    sudo chmod a+rw /dev/uinput   (or add yourself to uinput group)

USAGE:
    python3 robot/send_key.py
"""

import sys

try:
    import evdev
    from evdev import UInput, ecodes as e
except ImportError:
    print("ERROR: Run:  pip install evdev")
    sys.exit(1)

try:
    ui = UInput()
except PermissionError:
    print("ERROR: Permission denied on /dev/uinput")
    print("Fix:  sudo chmod a+rw /dev/uinput")
    sys.exit(1)

print("Ready. Press ENTER to send Right Arrow to lerobot.")
print("Press Ctrl-C to quit.\n")

while True:
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        print("\nDone.")
        break
    ui.write(e.EV_KEY, e.KEY_RIGHT, 1)
    ui.write(e.EV_KEY, e.KEY_RIGHT, 0)
    ui.syn()
    print("  → Right Arrow sent")
