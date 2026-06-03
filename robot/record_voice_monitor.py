#!/usr/bin/env python3
"""
record_voice_monitor.py
=======================
Pipe lerobot-record output through this script for spoken phase announcements.
Announces exactly once per phase transition (setup → recording → setup → ...).

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
CHUTE_SIZE   = 5    # must match n_pieces in connect_four_top_grab_chute.scad
ESPEAK_SPEED = 145  # words per minute


# ── Speaker: one worker thread, no overlap ────────────────────────────────────

class Speaker:
    """
    One background thread processes speech. say() kills any current speech
    and queues the new message — so only one thing ever plays at a time.
    """

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
                return name
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        return None

    def say(self, text):
        # 1. Kill whatever is playing
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=0.3)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        # 2. Drain stale pending messages
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        # 3. Queue the new message
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

# Phases — we only speak when the phase CHANGES, so multiple log lines
# that all say "Reset" don't trigger multiple announcements.
PHASE_SETUP     = 'setup'
PHASE_RECORDING = 'recording'

def monitor():
    phase       = None   # current known phase
    episode_num = 0

    for raw_line in sys.stdin:
        sys.stdout.write(raw_line)
        sys.stdout.flush()
        line = raw_line.strip()

        # ── Recording started ──────────────────────────────────────────────
        m = re.search(r'[Rr]ecording episode[^\d]*(\d+)', line)
        if m:
            episode_num = int(m.group(1))
            if phase != PHASE_RECORDING:
                phase    = PHASE_RECORDING
                disc_pos = ((episode_num - 1) % CHUTE_SIZE) + 1
                _speaker.say(
                    f"Recording. Disc {disc_pos} of {CHUTE_SIZE}. "
                    f"Episode {episode_num}. Go."
                )
            continue

        # ── Reset / setup window ───────────────────────────────────────────
        # Match any line that signals a reset or warmup phase.
        if re.search(r'[Rr]eset|[Ww]arm.?up', line):
            if phase != PHASE_SETUP:
                phase     = PHASE_SETUP
                next_ep   = episode_num + 1
                next_disc = ((next_ep - 1) % CHUTE_SIZE) + 1
                if episode_num == 0:
                    # Very first setup
                    _speaker.say(
                        f"Setup time. Load {CHUTE_SIZE} discs into the chute. "
                        f"Move arm to start position. Press Right Arrow when ready."
                    )
                else:
                    reload = (
                        f"Reload the chute with {CHUTE_SIZE} discs. "
                        if next_disc == 1 else ""
                    )
                    _speaker.say(
                        f"Setup time. {reload}"
                        f"Next: disc {next_disc} of {CHUTE_SIZE}. "
                        f"Press Right Arrow when ready."
                    )
            continue

        # ── Session done ───────────────────────────────────────────────────
        if re.search(r'[Ee]xiting', line):
            _speaker.say("Session complete. Well done.")
            continue


if __name__ == '__main__':
    monitor()
