#!/usr/bin/env python3
"""
record_voice_monitor.py
=======================
Pipe lerobot-record output through this script for spoken phase announcements.

USAGE (called automatically by record_first_dataset.sh):
    lerobot-record [args] 2>&1 | python3 robot/record_voice_monitor.py

INSTALL TTS (Ubuntu, one-time):
    sudo apt install espeak
"""

import queue
import re
import subprocess
import sys
import threading

# ── Config ────────────────────────────────────────────────────────────────────
CHUTE_SIZE  = 5    # must match n_pieces in connect_four_top_grab_chute.scad
ESPEAK_SPEED = 145  # words per minute — increase to speed up


# ── Speaker: one background thread, no overlap ────────────────────────────────

class Speaker:
    """
    Serialises all speech through a single worker thread.
    Calling say() while something is playing immediately kills the current
    speech and starts the new one — no queuing, no overlap.
    """

    def __init__(self):
        self._q      = queue.Queue()
        self._proc   = None
        self._lock   = threading.Lock()
        self._engine = self._detect()
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def _detect(self):
        """Return the first available TTS command name, or None."""
        candidates = {
            'espeak':  ['espeak', '--version'],
            'spd-say': ['spd-say', '--version'],
            'say':     ['say', '--version'],        # macOS fallback
        }
        for name, probe in candidates.items():
            try:
                subprocess.run(probe, capture_output=True, timeout=2)
                return name
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        return None

    def say(self, text):
        """Queue text for speaking. Kills current speech so new phrase starts immediately."""
        # Kill whatever is playing right now
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=0.3)
                except subprocess.TimeoutExpired:
                    self._proc.kill()

        # Drain any pending messages so we never build a backlog
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

        self._q.put(text)

    def _worker(self):
        while True:
            text = self._q.get()

            if self._engine is None:
                print(f"\n[ANNOUNCE] {text}\n", file=sys.stderr, flush=True)
                continue

            if self._engine == 'espeak':
                cmd = ['espeak', '-s', str(ESPEAK_SPEED), text]
            elif self._engine == 'spd-say':
                cmd = ['spd-say', '-w', text]
            else:
                cmd = ['say', text]

            with self._lock:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            self._proc.wait()   # block this worker until speech finishes


_speaker = Speaker()


# ── Monitor ───────────────────────────────────────────────────────────────────

def monitor():
    episode_num = 0

    for raw_line in sys.stdin:
        sys.stdout.write(raw_line)
        sys.stdout.flush()
        line = raw_line.strip()

        # ── Episode recording started ──────────────────────────────────────
        m = re.search(r'[Rr]ecording episode[^\d]*(\d+)', line)
        if m:
            episode_num = int(m.group(1))
            disc_pos    = ((episode_num - 1) % CHUTE_SIZE) + 1
            _speaker.say(
                f"Recording. Disc {disc_pos} of {CHUTE_SIZE}. "
                f"Episode {episode_num}. Go."
            )
            continue

        # ── Episode saved ──────────────────────────────────────────────────
        if re.search(r'[Ss]aving episode', line):
            _speaker.say("Saved. Return arm to start position.")
            continue

        # ── Reset / setup window ───────────────────────────────────────────
        if re.search(r'[Rr]eset', line):
            next_ep   = episode_num + 1
            next_disc = ((next_ep - 1) % CHUTE_SIZE) + 1
            reload    = f"Reload the chute with {CHUTE_SIZE} discs. " \
                        if (next_disc == 1 and episode_num > 0) else ""
            _speaker.say(
                f"Setup time. {reload}"
                f"Next: disc {next_disc} of {CHUTE_SIZE}. "
                f"Press Right Arrow when ready."
            )
            continue

        # ── First warmup (very start of session) ───────────────────────────
        if re.search(r'[Ww]arm.?up', line):
            _speaker.say(
                f"Setup time. Load {CHUTE_SIZE} discs. "
                f"Move arm to start position. Press Right Arrow when ready."
            )
            continue

        # ── Session done ───────────────────────────────────────────────────
        if re.search(r'[Ee]xiting', line):
            _speaker.say("Session complete. Well done.")
            continue


if __name__ == '__main__':
    monitor()
