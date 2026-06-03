#!/usr/bin/env python3
"""
record_voice_monitor.py
=======================
Adds chute-specific voice announcements on top of lerobot's built-in audio.
lerobot already handles setup/recording phase transitions with its own sounds.
This script only announces disc position and when to reload the chute —
the one thing lerobot doesn't know about.

USAGE (called by record_first_dataset.sh — do not run directly):
    lerobot-record [args] 2>&1 | python3 robot/record_voice_monitor.py

INSTALL TTS (Ubuntu, one-time):
    sudo apt install espeak

DEBUG LOG: /tmp/lerobot_monitor.log
"""

import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
CHUTE_SIZE   = 5    # must match n_pieces in connect_four_top_grab_chute.scad
ESPEAK_SPEED = 145
DEBUG_LOG    = '/tmp/lerobot_monitor.log'

# ── Debug log ─────────────────────────────────────────────────────────────────

_dbg = open(DEBUG_LOG, 'a', buffering=1)

def dbg(label, msg):
    ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    _dbg.write(f"[{ts}] {label}: {msg}\n")

# ── Speaker ───────────────────────────────────────────────────────────────────

class Speaker:
    """Single worker thread — new say() kills current speech immediately."""

    def __init__(self):
        self._q      = queue.Queue()
        self._proc   = None
        self._lock   = threading.Lock()
        self._engine = self._detect()
        threading.Thread(target=self._worker, daemon=True).start()

    def _detect(self):
        for name, probe in [
            ('espeak',  ['espeak',  '--version']),
            ('spd-say', ['spd-say', '--version']),
            ('say',     ['say',     '--version']),
        ]:
            try:
                subprocess.run(probe, capture_output=True, timeout=2)
                dbg('TTS', f'engine: {name}')
                return name
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        dbg('TTS', 'no engine found — printing to stderr')
        return None

    def say(self, text):
        dbg('SAY', text)
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=0.3)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
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
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._proc.wait()


_speaker = Speaker()

# ── Monitor ───────────────────────────────────────────────────────────────────

def monitor():
    dbg('START', 'monitor running')

    # Debounce: track last announced episode + time to avoid double-firing
    # when lerobot prints multiple matching lines for the same episode
    # (e.g. "Recording setup for episode 1" then "Recording episode 1 / 50").
    _last_ep   = -1
    _last_time = 0.0
    DEBOUNCE_S = 3.0

    for raw_line in sys.stdin:
        sys.stdout.write(raw_line)
        sys.stdout.flush()
        line = raw_line.strip()
        dbg('LINE', line)

        # Only act when lerobot starts an episode setup window.
        # Episode 0 is a camera warmup — skip it.
        m = re.search(r'Recording episode[^\d]*(\d+)', line)
        if not m:
            continue

        ep  = int(m.group(1))
        now = time.monotonic()
        dbg('MATCH', f'episode {ep}')

        if ep == 0:
            dbg('SKIP', 'warmup episode')
            continue

        if ep == _last_ep and (now - _last_time) < DEBOUNCE_S:
            dbg('SKIP', f'debounce duplicate for episode {ep}')
            continue

        _last_ep   = ep
        _last_time = now

        disc_pos     = ((ep - 1) % CHUTE_SIZE) + 1
        needs_reload = (disc_pos == 1) and (ep > 1)

        if needs_reload:
            _speaker.say(f"Reload the chute. Disc 1 of {CHUTE_SIZE}.")
        else:
            _speaker.say(f"Disc {disc_pos} of {CHUTE_SIZE}.")


if __name__ == '__main__':
    monitor()
