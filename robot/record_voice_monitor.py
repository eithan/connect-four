#!/usr/bin/env python3
"""
record_voice_monitor.py
=======================
Pipe lerobot-record output through this script for spoken phase announcements.
Tells you clearly when you're in setup time vs. actively recording, and when
to reload the chute.

USAGE (from record_first_dataset.sh — do not run directly):
    lerobot-record [args] 2>&1 | python3 robot/record_voice_monitor.py

INSTALL TTS (Ubuntu, one-time):
    sudo apt install espeak
"""

import re
import sys
import subprocess
import threading

# ── Config ────────────────────────────────────────────────────────────────────
CHUTE_SIZE = 5      # must match n_pieces in connect_four_top_grab_chute.scad
ESPEAK_SPEED = 145  # words per minute — adjust if too fast/slow

# ── TTS ───────────────────────────────────────────────────────────────────────

def speak(text):
    """Speak text using the first available TTS engine."""
    engines = [
        ['espeak', '-s', str(ESPEAK_SPEED), text],  # Ubuntu: sudo apt install espeak
        ['spd-say', '-w', text],                     # Ubuntu speech-dispatcher (pre-installed)
        ['say', text],                               # macOS fallback
    ]
    for cmd in engines:
        try:
            subprocess.run(cmd, timeout=15, capture_output=True)
            return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    # No TTS engine found — print visually instead
    print(f"\n[ANNOUNCE] {text}\n", file=sys.stderr, flush=True)


def say_async(text):
    threading.Thread(target=speak, args=(text,), daemon=True).start()

# ── Monitor ───────────────────────────────────────────────────────────────────

def monitor():
    episode_num = 0

    for raw_line in sys.stdin:
        # Always pass the line through to the terminal
        sys.stdout.write(raw_line)
        sys.stdout.flush()
        line = raw_line.strip()

        # ── Episode recording started ──────────────────────────────────────
        m = re.search(r'[Rr]ecording episode[^\d]*(\d+)', line)
        if m:
            episode_num = int(m.group(1))
            disc_pos = ((episode_num - 1) % CHUTE_SIZE) + 1
            say_async(
                f"Recording. "
                f"Disc {disc_pos} of {CHUTE_SIZE}. "
                f"Episode {episode_num}. "
                f"Go."
            )
            continue

        # ── Episode ended / saving ─────────────────────────────────────────
        if re.search(r'[Ss]aving episode', line) or re.search(r'[Ss]top record', line):
            say_async("Episode saved. Return arm to start position.")
            continue

        # ── Setup / reset phase ────────────────────────────────────────────
        # LeRobot logs a reset message before each episode's setup window.
        if re.search(r'[Rr]eset', line):
            next_ep = episode_num + 1
            next_disc = ((next_ep - 1) % CHUTE_SIZE) + 1

            if next_disc == 1 and episode_num > 0:
                # Chute just ran out
                reload_msg = f"Reload the chute with {CHUTE_SIZE} discs. "
            else:
                reload_msg = ""

            say_async(
                f"Setup time. "
                f"{reload_msg}"
                f"Next disc: {next_disc} of {CHUTE_SIZE}. "
                f"Return arm to start position, then press Right Arrow."
            )
            continue

        # ── Warmup / first setup ───────────────────────────────────────────
        if re.search(r'[Ww]arm.?up', line):
            say_async(
                f"Setup time. Load {CHUTE_SIZE} discs into the chute. "
                f"Move arm to start position. Press Right Arrow when ready."
            )
            continue

        # ── Session complete ───────────────────────────────────────────────
        if re.search(r'[Ee]xiting', line):
            say_async("Recording session complete. Well done.")
            continue


if __name__ == '__main__':
    monitor()
