#!/usr/bin/env bash
# =============================================================
#  send_key.sh
#  Run in a SECOND terminal while record_first_dataset.sh runs.
#  Press ENTER here to send a Right Arrow keystroke to lerobot.
#
#  Uses kernel-level evdev injection (bypasses Wayland/X11).
#  PREREQS:  pip install evdev
#            User must be in the 'input' group (already done after reboot).
# =============================================================

python3 - <<'EOF'
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
    print("Fix:  sudo chmod a+rw /dev/uinput  (temporary)")
    print("  or: sudo usermod -a -G uinput $USER  then reboot (permanent)")
    sys.exit(1)

print("Ready. Press ENTER to send Right Arrow to lerobot.")
print("Press Ctrl-C to quit.\n")

while True:
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        break
    ui.write(e.EV_KEY, e.KEY_RIGHT, 1)
    ui.write(e.EV_KEY, e.KEY_RIGHT, 0)
    ui.syn()
    print(f"  → Right Arrow sent")
EOF
