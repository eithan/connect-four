"""
TTS Announcer — Connect Four
=============================

Standalone text-to-speech helper.  No ROS, no game logic — just speaks.

Adapted from the JetAuto tts_node pattern: a dedicated speaker thread pulls
text from a queue so the caller never blocks.

Usage
-----
    from tts_announcer import GameAnnouncer

    ann = GameAnnouncer()                    # pyttsx3 auto-detected
    ann.speak("Your turn")                   # non-blocking
    ann.speak("Column 3", interrupt=True)    # clears queue first
    ann.stop()                               # clean shutdown on exit

If pyttsx3 is not installed the announcer becomes a silent no-op so the
rest of the game loop continues unaffected.

Install:  pip3 install pyttsx3
"""

import queue
import threading
import time
from typing import Optional


class GameAnnouncer:
    """
    Non-blocking text-to-speech announcer.

    Speak requests are queued and consumed by a dedicated thread so the
    game loop is never blocked waiting for audio to finish.
    """

    def __init__(self,
                 rate:     int   = 155,
                 volume:   float = 0.92,
                 voice_id: str   = "",
                 enabled:  bool  = True):
        """
        Parameters
        ----------
        rate:     words-per-minute (pyttsx3 default ~200; lower = clearer)
        volume:   0.0 – 1.0
        voice_id: optional pyttsx3 voice ID string (e.g. for a specific OS voice)
        enabled:  set False to silence all speech without removing call sites
        """
        self.enabled = enabled
        self._engine   = None
        self._queue    = queue.Queue(maxsize=6)
        self._shutdown = threading.Event()
        self._thread:  Optional[threading.Thread] = None

        if not enabled:
            return

        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate",   rate)
            self._engine.setProperty("volume", volume)
            if voice_id:
                self._engine.setProperty("voice", voice_id)

            self._thread = threading.Thread(
                target=self._speaker_loop, daemon=True, name="tts-speaker"
            )
            self._thread.start()
            print("[TTS] Ready")

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
        self._engine = None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _clear_queue(self):
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _speaker_loop(self):
        """Dedicated thread: pull text → speak → repeat."""
        while not self._shutdown.is_set():
            try:
                text = self._queue.get(timeout=0.4)
            except queue.Empty:
                continue

            if self._engine is None:
                continue

            try:
                self._engine.say(text)
                self._engine.runAndWait()
                time.sleep(0.15)        # brief pause between utterances
            except Exception as e:
                print(f"[TTS] Speak error: {e}")
