"""
TTS Announcer — Connect Four
=============================

Standalone text-to-speech helper.  No ROS, no game logic — just speaks.

On macOS the built-in ``say`` command is used (no dependencies, no threading
issues — pyttsx3's nsss driver crashes when called from a background thread).

On Linux/other platforms pyttsx3 is used, falling back to silent no-op if
it is not installed.

Usage
-----
    from tts_announcer import GameAnnouncer

    ann = GameAnnouncer()                    # auto-detected backend
    ann.speak("Your turn")                   # non-blocking
    ann.speak("Column 3", interrupt=True)    # clears queue first
    ann.stop()                               # clean shutdown on exit

Install (Linux):  pip3 install pyttsx3
macOS:            nothing extra needed (uses /usr/bin/say)
"""

import platform
import queue
import subprocess
import threading
import time
from typing import Optional


class GameAnnouncer:
    """
    Non-blocking text-to-speech announcer.

    On macOS: uses the built-in ``say`` command via subprocess — avoids the
    pyttsx3/nsss thread-safety crash.

    On other platforms: uses pyttsx3 with a dedicated speaker thread.
    """

    def __init__(self,
                 rate:     int   = 155,
                 volume:   float = 0.92,
                 voice_id: str   = "",
                 enabled:  bool  = True):
        """
        Parameters
        ----------
        rate:     words-per-minute
        volume:   0.0 – 1.0
        voice_id: optional pyttsx3 voice ID string (Linux) or macOS voice name
        enabled:  set False to silence all speech without removing call sites
        """
        self.enabled  = enabled
        self._queue   = queue.Queue(maxsize=6)
        self._shutdown = threading.Event()
        self._thread:  Optional[threading.Thread] = None

        # ── macOS: use `say` subprocess ───────────────────────────────────────
        self._macos = (platform.system() == "Darwin")
        self._macos_voice  = voice_id or ""   # e.g. "Samantha"
        self._macos_rate   = rate              # wpm; `say -r <rate>`
        self._pyttsx3_engine = None

        if not enabled:
            return

        if self._macos:
            self._thread = threading.Thread(
                target=self._speaker_loop_macos, daemon=True, name="tts-speaker"
            )
            self._thread.start()
            print("[TTS] Ready (macOS say)")
            return

        # ── Linux / other: pyttsx3 ────────────────────────────────────────────
        try:
            import pyttsx3
            self._pyttsx3_engine = pyttsx3.init()
            self._pyttsx3_engine.setProperty("rate",   rate)
            self._pyttsx3_engine.setProperty("volume", volume)
            if voice_id:
                self._pyttsx3_engine.setProperty("voice", voice_id)

            self._thread = threading.Thread(
                target=self._speaker_loop_pyttsx3, daemon=True, name="tts-speaker"
            )
            self._thread.start()
            print("[TTS] Ready (pyttsx3)")

        except ImportError:
            print("[TTS] pyttsx3 not found — speech disabled.  "
                  "Install with: pip3 install pyttsx3")
            self.enabled = False
        except Exception as e:
            print(f"[TTS] Init failed ({e}) — speech disabled")
            self.enabled = False

    # ── Public API ────────────────────────────────────────────────────────────

    def speak(self, text: str, interrupt: bool = False):
        """
        Queue ``text`` for speech.

        Parameters
        ----------
        text:      what to say
        interrupt: if True, clear any queued-but-not-yet-spoken utterances
                   first so this message is heard promptly
        """
        if not self.enabled or not text:
            return

        if interrupt:
            self._clear_queue()

        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass   # drop silently rather than block the game loop

    def stop(self):
        """Signal the speaker thread to finish and wait for it."""
        self._shutdown.set()
        self._clear_queue()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._pyttsx3_engine = None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _clear_queue(self):
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _speaker_loop_macos(self):
        """macOS speaker thread: uses subprocess ``say``."""
        current_proc: Optional[subprocess.Popen] = None

        while not self._shutdown.is_set():
            try:
                text = self._queue.get(timeout=0.4)
            except queue.Empty:
                continue

            # Kill any currently-speaking process if we're in interrupt mode
            # (interrupt already cleared the queue; just let the current finish
            #  naturally since it's usually short — or kill if still running)
            if current_proc is not None and current_proc.poll() is None:
                current_proc.terminate()
                current_proc.wait()

            cmd = ["say"]
            if self._macos_voice:
                cmd += ["-v", self._macos_voice]
            if self._macos_rate:
                cmd += ["-r", str(self._macos_rate)]
            cmd.append(text)

            try:
                current_proc = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                current_proc.wait()
            except Exception as e:
                print(f"[TTS] say error: {e}")

            time.sleep(0.10)

    def _speaker_loop_pyttsx3(self):
        """Linux speaker thread: uses pyttsx3."""
        while not self._shutdown.is_set():
            try:
                text = self._queue.get(timeout=0.4)
            except queue.Empty:
                continue

            if self._pyttsx3_engine is None:
                continue

            try:
                self._pyttsx3_engine.say(text)
                self._pyttsx3_engine.runAndWait()
                time.sleep(0.15)
            except Exception as e:
                print(f"[TTS] Speak error: {e}")
