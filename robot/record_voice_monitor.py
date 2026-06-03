#!/usr/bin/env python3
"""
record_voice_monitor.py
=======================
Pipe lerobot-record output through this script for spoken phase announcements.

USAGE (called automatically by record_first_dataset.sh):
    lerobot-record [args] 2>&1 | python3 robot/record_voice_monitor.py

INSTALL TTS (Ubuntu, one-time):
    sudo apt install espeak

DEBUG: all matched log lines are written to /tmp/lerobot_monitor.log
so you can see exactly what triggered each announcement.
"""

import queue
import re
import subprocess
import sys
import threading
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
CHUTE_SIZE   = 5    # must match n_pieces in connect_four_top_grab_chute.scad
ESPEAK_SPEED = 145  # words per minute
DEBUG_LOG    = '/tmp/lerobot_monitor.log'


# ── Debug log ─────────────────────────────────────────────────────────────────

_debug_file = open(DEBUG_LOG, 'a', buffering=1)

def dbg(label, line):
    ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    _debug_file.write(f"[{ts}] {label}: {line}\n")


# ── Speaker: one worker thread, no overlap ────────────────────────────────────

class Speaker:
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
                dbg('TTS', f'using engine: {name}')
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

PHASE_SETUP     = 'setup'
PHASE_RECORDING = 'recording'


# ── Monitor ───────────────────────────────────────────────────────────────────

def monitor():
    phase       = None
    episode_num = 0

    dbg('START', 'monitor running')

    for raw_line in sys.stdin:
        sys.stdout.write(raw_line)
        sys.stdout.flush()
        line = raw_line.strip()

        # Log every line so we can see the full lerobot output
        dbg('LINE', line)

        # ── Recording started ──────────────────────────────────────────────
        m = re.search(r'[Rr]ecording episode[^\d]*(\d+)', line)
        if m:
            ep = int(m.group(1))
            dbg('MATCH', f'recording episode {ep} (phase={phase})')

            # Episode 0 is lerobot's warmup/preview — skip it
            if ep == 0:
                dbg('SKIP', 'episode 0 is warmup, ignoring')
                continue

            if phase != PHASE_RECORDING:
                phase       = PHASE_RECORDING
                episode_num = ep
                disc_pos    = ((episode_num - 1) % CHUTE_SIZE) + 1
                _speaker.say(
                    f"Recording. Disc {disc_pos} of {CHUTE_SIZE}. "
                    f"Episode {episode_num}. Go."
                )
            continue

        # ── Reset / setup window ───────────────────────────────────────────
        if re.search(r'[Rr]eset|[Ww]arm.?up', line):
            dbg('MATCH', f'reset/warmup (phase={phase}, episode_num={episode_num})')

            if phase != PHASE_SETUP:
                phase     = PHASE_SETUP
                next_ep   = episode_num + 1
                next_disc = ((next_ep - 1) % CHUTE_SIZE) + 1

                if episode_num == 0:
                    # Very start of session — no episodes recorded yet
                    _speaker.say(
                        f"Setup time. Load {CHUTE_SIZE} discs into the chute. "
                        f"Move arm to start position. "
                        f"Press Right Arrow when ready."
                    )
                elif next_disc == 1:
                    # Cycle boundary — chute needs refilling
                    _speaker.say(
                        f"Setup time. Reload the chute with {CHUTE_SIZE} discs. "
                        f"Press Right Arrow when ready."
                    )
                else:
                    _speaker.say(
                        f"Setup time. Disc {next_disc} of {CHUTE_SIZE} next. "
                        f"Press Right Arrow when ready."
                    )
            continue

        # ── Session done ───────────────────────────────────────────────────
        if re.search(r'[Ee]xiting', line):
            dbg('MATCH', 'exiting')
            _speaker.say("Session complete. Well done.")
            continue


if __name__ == '__main__':
    monitor()
