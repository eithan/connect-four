"""
Game Loop - Full cooperative Connect Four game via live camera.

COOPERATIVE MODE: You place both your piece and the AI's piece.
The system detects your move, the AI picks a column, and you drop the AI's
piece there. The camera confirms the placement.

Usage:
    python3 game_loop.py
    python3 game_loop.py --camera 1
    python3 game_loop.py --human-color yellow   # Play as yellow (AI is red)
    python3 game_loop.py --no-mirror            # Disable mirror flip
    python3 game_loop.py --stable-seconds 1.5  # Slower confirmation

Controls:
    q / ESC  - Quit
    r        - Reset game
    f        - Freeze / unfreeze frame
    l        - Re-lock board detection
"""

import cv2
import numpy as np
import argparse
import json
import time
import os
import sys
import traceback
from datetime import datetime
from enum import Enum
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

class _LogTee:
    """Tee stdout to both console and a log file simultaneously."""
    def __init__(self, log_path: str):
        self._term = sys.stdout
        self._file = open(log_path, "w", buffering=1, encoding="utf-8")

    def write(self, msg: str):
        self._term.write(msg)
        self._file.write(msg)

    def flush(self):
        self._term.flush()
        self._file.flush()

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass


def setup_logging(log_dir: str = "logs") -> tuple:
    """
    Create a timestamped log file and screenshot directory.
    Redirects stdout so all print() calls are captured automatically.

    Returns (log_path, screenshot_dir, tee_object).
    Call tee.close() on exit to flush the log.
    """
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"game_{stamp}.log")
    ss_dir   = os.path.join(log_dir, f"game_{stamp}_shots")
    os.makedirs(ss_dir, exist_ok=True)
    tee = _LogTee(log_path)
    sys.stdout = tee
    print(f"Logging to {log_path}")
    print(f"Screenshots → {ss_dir}")
    return log_path, ss_dir, tee


from board_detector import BoardDetector, LockedBoardDetector, YOLOEnhancedBoardDetector, DetectionConfig, SCREEN_CONFIG, PHYSICAL_CONFIG
from turn_tracker import TurnTracker
from ai_player import AIPlayer
from tts_announcer import GameAnnouncer


# ─────────────────────────────────────────────────────────────────────────────
# Stable-state detector
# ─────────────────────────────────────────────────────────────────────────────

class StableStateDetector:
    """
    Detects confirmed board-state CHANGES from a known baseline.

    A move is accepted only when ALL of the following hold:
      1. The board differs from the last accepted state (something actually changed).
      2. The new board state is identical for ``required_frames`` consecutive frames.
      3. Every one of those frames has confidence ≥ ``min_confidence``.

    Why this beats "N identical frames of anything":
      - If skin/hand creates a transient piece that then disappears (board
        returns to baseline), the buffer is cleared — no false trigger.
      - If confidence drops mid-window (hand still partly covering the board),
        the buffer is cleared — hand must be fully withdrawn before we count.
      - A real piece that has just been placed sits there indefinitely, easily
        surviving the stability window once the hand is gone.

    Call ``reset(new_baseline)`` after each accepted move to update the
    reference state.  Call ``reset()`` with no argument to start fresh
    (baseline is set from the first high-confidence frame).
    """

    def __init__(self, required_frames: int = 5, min_confidence: float = 0.75):
        self.required_frames = required_frames
        self.min_confidence  = min_confidence
        self._buffer:   list                   = []
        self._baseline: Optional[np.ndarray]  = None

    def reset(self, baseline: Optional[np.ndarray] = None):
        """Clear buffer and optionally set a new reference baseline."""
        self._buffer.clear()
        self._baseline = baseline.copy() if baseline is not None else None

    def update(self, board: np.ndarray, confidence: float):
        """
        Returns (is_new_stable, stable_board).
        is_new_stable is True only on the frame stability is first confirmed.
        """
        # ── Bootstrap: no baseline yet ────────────────────────────────────────
        if self._baseline is None:
            if confidence >= self.min_confidence:
                self._baseline = board.copy()   # first clean frame → baseline
            return False, None

        # ── Board returned to baseline (false piece removed / hand gone) ──────
        if np.array_equal(board, self._baseline):
            if self._buffer:
                self._buffer.clear()
            return False, None

        # ── Board differs from baseline — wait for confident stable window ────
        if confidence < self.min_confidence:
            self._buffer.clear()    # hand still present; don't count this frame
            return False, None

        if self._buffer and not np.array_equal(board, self._buffer[-1]):
            self._buffer.clear()    # state still changing

        self._buffer.append(board.copy())

        if len(self._buffer) >= self.required_frames:
            stable = self._buffer[-1].copy()
            self._baseline = stable.copy()      # advance baseline to new state
            self._buffer.clear()
            return True, stable

        return False, None

    @property
    def progress(self) -> int:
        return len(self._buffer)


# ─────────────────────────────────────────────────────────────────────────────
# Game state machine
# ─────────────────────────────────────────────────────────────────────────────

class GamePhase(Enum):
    HUMAN_TURN     = "human_turn"      # Waiting for human to drop a piece
    AI_TURN        = "ai_turn"         # AI chose a column; human places it
    GAME_OVER      = "game_over"       # Win or draw


class GameLoop:
    WINDOW = "Connect Four - Game"

    # Startup first-move guard: after locking an empty board, the human's
    # first piece must remain stable for this many consecutive stable-detector
    # windows before the game accepts it.  Each window requires `stable_frames`
    # (default 5) confident identical frames, so 2 windows ≈ 10 frames ≈ 0.3 s
    # — long enough to outlast a face/hand transient without feeling sluggish.
    # NOTE: after each window we reset the stable baseline to empty so the
    # detector keeps firing; without that reset the counter stalls at 1.
    STARTUP_PIECE_HOLD_FRAMES = 2

    COLOR_NAMES = {1: "Red", 2: "Yellow"}
    PIECE_COLORS = {
        1: (0, 0, 255),       # Red  → BGR
        2: (0, 220, 255),     # Yellow → BGR
        0: (70, 70, 70),      # Empty
    }

    def __init__(self, detector: LockedBoardDetector, ai: AIPlayer, tracker: TurnTracker,
                 human_player: int = 1, stable_frames: int = 5,
                 announcer: Optional["GameAnnouncer"] = None):
        self.detector = detector
        self.ai = ai
        self.tracker = tracker
        self.human_player = human_player
        self.ai_player_num = 3 - human_player
        self.ann = announcer or GameAnnouncer(enabled=False)   # silent no-op if omitted

        self.stable = StableStateDetector(required_frames=stable_frames,
                                          min_confidence=0.75)
        self.phase = GamePhase.HUMAN_TURN
        self.ai_column: Optional[int] = None
        self.ai_policy: Optional[np.ndarray] = None
        self._ai_turn_saved_board:   Optional[np.ndarray] = None  # saved before AI turn
        self._ai_turn_saved_player:  int = 1
        self.status_msg = ""
        self.last_result = None
        self.frozen = False
        self._initialized = False    # True after first stable board observed
        self._prev_locked = False    # track lock→unlock transitions
        self._lock_board_was_empty = False
        self._latest_frame:   Optional[np.ndarray] = None  # raw camera frame
        self._latest_display: Optional[np.ndarray] = None  # drawn overlay
        self._ss_dir:  str = ""   # screenshot directory (set via set_screenshot_dir)
        self._ss_count: int = 0
        self._last_periodic_ss: float = 0.0   # timestamp of last periodic screenshot
        self._session_start:  datetime = datetime.now()   # for manifest

        self.fps_actual = 0.0
        self._fps_t = time.time()
        self._fps_count = 0

        self._winning_cells: list = []    # [(row, col), ...] — set at game over
        self._win_anim_start: float = 0.0
        self._startup_piece_hold: int = 0  # consecutive frames startup piece has been stable

        self._set_status_for_phase()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_screenshot_dir(self, path: str):
        self._ss_dir = path

    def _save_screenshot(self, label: str = "event"):
        """Save event screenshots and a JSON sidecar for replay/testing.

        Writes three files per event:
          NNNN_label.jpg         — overlay/display frame (visual debug)
          NNNN_label_raw.jpg     — raw camera frame (used by replay_test.py)
          NNNN_label.json        — metadata sidecar (raw_board, stable_board, etc.)
        """
        if not self._ss_dir:
            return
        display = self._latest_display if self._latest_display is not None else self._latest_frame
        raw = self._latest_frame
        if display is None and raw is None:
            return
        self._ss_count += 1
        base = os.path.join(self._ss_dir, f"{self._ss_count:04d}_{label}")

        if display is not None:
            cv2.imwrite(base + ".jpg", display)
        if raw is not None:
            cv2.imwrite(base + "_raw.jpg", raw)

        meta = self._build_frame_meta(label)
        with open(base + ".json", "w") as f:
            json.dump(meta, f, indent=2)

    def _build_frame_meta(self, event: str) -> dict:
        """Collect per-frame metadata for the JSON sidecar."""
        result = self.last_result
        raw_board = None
        stable_board = None
        confidence = 0.0
        if result is not None:
            raw_board    = (result.raw_board.tolist()
                            if result.raw_board is not None
                            else result.board.tolist())
            stable_board = result.board.tolist()
            confidence   = float(result.confidence)
        return {
            "seq":          self._ss_count,
            "event":        event,
            "timestamp":    datetime.now().isoformat(),
            "game_phase":   self.phase.name,
            "raw_board":    raw_board,
            "stable_board": stable_board,
            "confidence":   round(confidence, 4),
            "human_color":  self.human_player,
        }

    def _write_manifest(self, quit_reason: str = "unknown"):
        """Write session manifest.json into the screenshot directory."""
        if not self._ss_dir:
            return
        try:
            raw_frames = sorted(
                f for f in os.listdir(self._ss_dir)
                if f.endswith("_raw.jpg")
            )
            manifest = {
                "session":     os.path.basename(self._ss_dir),
                "started":     self._session_start.isoformat(),
                "ended":       datetime.now().isoformat(),
                "complete":    self.phase == GamePhase.GAME_OVER,
                "quit_reason": quit_reason,
                "human_color": self.human_player,
                "raw_frames":  raw_frames,
            }
            with open(os.path.join(self._ss_dir, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)
        except Exception as exc:
            print(f"[WARN] Could not write manifest: {exc}")

    def reset(self):
        self.tracker.reset()
        self.stable.reset()   # no baseline — will bootstrap from first clean frame
        self.phase = GamePhase.HUMAN_TURN
        self.ai_column = None
        self.ai_policy = None
        self._ai_turn_saved_board  = None
        self._ai_turn_saved_player = 1
        self.last_result = None
        self._initialized = False
        self._lock_board_was_empty = False
        self._winning_cells = []
        self._win_anim_start = 0.0
        self._startup_piece_hold = 0
        self._set_status_for_phase()
        self.ann.speak("New game. Your turn.", interrupt=True)
        print("\n" + "=" * 50)
        print("  Game reset — your turn!")
        print("=" * 50 + "\n")

    def process_frame(self, frame: np.ndarray):
        """Run detection + state machine. Call at your desired FPS."""
        self._latest_frame = frame
        now = time.time()
        self._fps_count += 1
        if now - self._fps_t >= 1.0:
            self.fps_actual = self._fps_count / (now - self._fps_t)
            self._fps_count = 0
            self._fps_t = now

        result = self.detector.detect(frame)
        self.last_result = result

        # Track lock transitions
        just_locked = not self._prev_locked and self.detector.is_locked
        just_unlocked = self._prev_locked and not self.detector.is_locked
        self._prev_locked = self.detector.is_locked

        if just_locked:
            # Save a diagnostic screenshot immediately on lock so we can see
            # the grid overlay BEFORE stable detection — useful when phantom
            # pieces prevent stable detection from ever firing.
            self._save_screenshot("board_locked")
            self._last_periodic_ss = time.time()
            # If the board looks empty at lock time, announce "Your turn"
            # immediately — don't wait for stable detection (~2s).  If the
            # human plays quickly and the board has a piece by the time
            # stable detection fires, initialization goes straight to AI
            # turn without re-announcing.
            self._lock_board_was_empty = (
                result.board is not None and int(np.sum(result.board != 0)) == 0
            )
            if self._lock_board_was_empty:
                self.ann.speak("Board detected. Your turn.", interrupt=True)
            else:
                self.ann.speak("Board detected.", interrupt=True)

        # Periodic screenshot every 15s during pre-initialization (diagnostics)
        if (not self._initialized and self.detector.is_locked
                and time.time() - self._last_periodic_ss > 15.0):
            self._save_screenshot("periodic")
            self._last_periodic_ss = time.time()

        if just_unlocked:
            # Board was lost — reset so the next lock re-seeds the tracker.
            self._initialized = False
            self._lock_board_was_empty = False
            self.stable.reset()

        if self.phase == GamePhase.GAME_OVER:
            return

        # Merge YOLO's view with the tracker's confirmed state.
        # Once a piece is accepted by the tracker it is permanent (Connect Four
        # pieces don't disappear mid-game), so we never let a transient YOLO
        # miss or hand occlusion erase it.  YOLO can only ADD to what the
        # tracker already knows.
        detected = result.board
        if self._initialized:
            merged = self.tracker.state.board.copy()
            for r in range(6):
                for c in range(7):
                    if merged[r, c] == 0 and detected[r, c] != 0:
                        merged[r, c] = detected[r, c]
        else:
            merged = detected

        is_stable, stable_board = self.stable.update(merged, result.confidence)
        if not is_stable:
            return

        # ── First stable frame: seed tracker with current board state ─────────
        if not self._initialized:
            red_n    = int(np.sum(stable_board == 1))
            yellow_n = int(np.sum(stable_board == 2))
            piece_n  = red_n + yellow_n

            # If lock-time detection looked empty, a single piece in the very
            # first stable startup board requires extra confirmation before the
            # game accepts it as the human's first move.  ADD_THRESHOLD commits
            # the piece to the stable board quickly (3 frames); we then require
            # STARTUP_PIECE_HOLD_FRAMES MORE consecutive frames of stable presence
            # (~0.5 s) to weed out faces/hands moving past the board.
            if self._lock_board_was_empty and piece_n == 1:
                piece_pos = np.argwhere(stable_board != 0)
                if len(piece_pos) == 1:
                    row, col = map(int, piece_pos[0])
                    piece_player = int(stable_board[row, col])
                    expected_row = 5
                    if piece_player == self.human_player and row == expected_row:
                        self._startup_piece_hold += 1
                        if self._startup_piece_hold >= self.STARTUP_PIECE_HOLD_FRAMES:
                            print("Accepting startup first move after empty lock"
                                  f" — human piece detected in column {col}"
                                  f" (held {self._startup_piece_hold} windows)")
                            self._initialized = True
                            empty_board = np.zeros_like(stable_board)
                            self.tracker.set_board(empty_board)
                            self.stable.reset(empty_board)
                            self._handle_human_turn(stable_board)
                        else:
                            # Piece seen but not yet confirmed long enough.
                            # Reset stable baseline to empty so the detector
                            # fires again on the next stable window — without
                            # this the counter stalls at 1 forever because
                            # StableStateDetector only fires once per state change.
                            print(f"Startup piece hold {self._startup_piece_hold}"
                                  f"/{self.STARTUP_PIECE_HOLD_FRAMES} — waiting...")
                            self.stable.reset(np.zeros_like(stable_board))
                        return

                # Piece present but not matching expected startup conditions —
                # reset to empty and keep waiting.
                self._startup_piece_hold = 0
                print("Ignoring suspicious startup piece after empty lock"
                      " — waiting for clean empty board or a persistent first move")
                self.tracker.reset()
                self.stable.reset(np.zeros_like(stable_board))
                self._set_status_for_phase()
                return
            else:
                # No piece (or wrong piece count) — reset hold counter
                self._startup_piece_hold = 0

            # Reject wildly imbalanced counts before touching the tracker.
            # Red=12 Yellow=0 is impossible in a real game — it means background
            # contamination slipped through (shirt, etc.).  Reset and wait.
            counts_balanced = abs(red_n - yellow_n) <= 1
            if not counts_balanced and piece_n > 0:
                print(f"Ignoring unbalanced initial detection "
                      f"(Red={red_n} Yellow={yellow_n}) — background contamination?")
                self.stable.reset(np.zeros_like(stable_board))
                return

            self._initialized = True
            self.tracker.set_board(stable_board)
            # Update stable baseline so detector knows what "no change" looks like
            self.stable.reset(stable_board)
            print(f"\nInitial board detected: Red={red_n}, Yellow={yellow_n}")
            self._save_screenshot("lock_initial")

            if self.tracker.state.game_over:
                self._end_game({"game_over": True,
                                "winner": self.tracker.state.winner,
                                "winning_cells": self.tracker.state.winning_cells})
                return
            # If it's already the AI's turn when we start, compute immediately.
            if (self.tracker.state.current_player == self.ai_player_num
                    and piece_n > 0):
                print("AI's turn on startup — computing move...")
                self.phase = GamePhase.AI_TURN
                self._run_ai(stable_board)
            else:
                self._set_status_for_phase()
                self.ann.speak("Your turn.", interrupt=True)
            return

        if self.phase == GamePhase.HUMAN_TURN:
            self._handle_human_turn(stable_board)
        elif self.phase == GamePhase.AI_TURN:
            self._handle_ai_placement(stable_board)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Render the full game overlay onto a camera frame."""
        display = frame.copy()
        self._latest_frame = frame
        h, w = display.shape[:2]
        result = self.last_result

        if result is None:
            self._draw_status_bar(display, h, w)
            return display

        # Board contour
        if result.board_contour is not None:
            cv2.drawContours(display, [result.board_contour], -1, (0, 255, 0), 2)

        # Pieces + target column highlight
        if result.grid_centers is not None:
            self._draw_grid(display, result)
            if self.phase == GamePhase.AI_TURN and self.ai_column is not None:
                self._draw_ai_column(display, result.grid_centers)
            if self.ai_policy is not None:
                self._draw_policy_bars(display, result.grid_centers)

        # Top HUD
        conf = result.confidence if result else 0.0
        conf_color = (0, 255, 0) if conf > 0.7 else (0, 165, 255) if conf > 0.4 else (0, 0, 255)
        lock_label = "LOCKED" if self.detector.is_locked else "searching..."
        lock_color = (255, 220, 0) if self.detector.is_locked else (0, 165, 255)
        cv2.putText(display, f"Conf:{conf:.2f}  FPS:{self.fps_actual:.1f}  {lock_label}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, lock_color, 1)

        # Stability progress indicator
        if self.stable.progress > 0:
            pct = self.stable.progress / self.stable.required_frames
            bar_w = int(200 * pct)
            cv2.rectangle(display, (10, 34), (210, 48), (50, 50, 50), -1)
            cv2.rectangle(display, (10, 34), (10 + bar_w, 48), (0, 200, 100), -1)
            cv2.putText(display, "Detecting...", (215, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.putText(display, "r=reset  f=freeze  l=re-lock  q=quit",
                    (w - 360, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        self._draw_status_bar(display, h, w)
        self._latest_display = display
        return display

    # ── State handlers ────────────────────────────────────────────────────────

    def _run_ai(self, board: np.ndarray):
        """Compute AI move and transition to AI_TURN phase."""
        # Save tracker state before AI turn in case wrong column is placed
        self._ai_turn_saved_board  = self.tracker.state.board.copy()
        self._ai_turn_saved_player = self.tracker.state.current_player
        ai_col, info = self.ai.get_move(board, self.ai_player_num)
        self.ai_column = ai_col
        self.ai_policy = info["policy"]
        self.phase = GamePhase.AI_TURN
        ai_color = 'Red' if self.ai_player_num == 1 else 'Yellow'
        print(f"AI ({info['method'].upper()}) → column {ai_col}")
        print(f"  Policy: {np.array2string(info['policy'], precision=3)}")
        col_1 = ai_col + 1  # 1-indexed for display and speech
        self.status_msg = f"AI plays column {col_1} ({ai_color})"
        # Short, natural announcement; say "middle" for column 4
        col_name = "middle column four" if col_1 == 4 else f"column {col_1}"
        self.ann.speak(f"{ai_color} in {col_name}.", interrupt=True)

    def _handle_human_turn(self, board: np.ndarray):
        update = self.tracker.update(board)

        if not update["changed"]:
            if update["error"]:
                print(f"[tracker] {update['error']}")
                # Keep the stable detector anchored to the last accepted board
                # so a bad snapshot doesn't become the new reference state.
                self.stable.reset(self.tracker.state.board)
            return

        col = update["move_col"]
        print(f"\nHuman played column {col}")
        self._save_screenshot(f"human_col{col}")

        if update["game_over"]:
            self._end_game(update)
            return

        # Advance stable baseline to the newly accepted board state so the
        # detector doesn't re-trigger on the same position next turn.
        self.stable.reset(self.tracker.state.board)

        # Trigger AI
        self._run_ai(board)

    def _handle_ai_placement(self, board: np.ndarray):
        preview = self.tracker.analyze_transition(board)

        if not preview["changed"]:
            if preview["error"]:
                print(f"[tracker] AI placement candidate rejected: {preview['error']}")
                # Stay anchored to the last accepted position and wait for a
                # cleaner stable board instead of treating noise as a move.
                self.stable.reset(self.tracker.state.board)
            return

        placed_col = preview["move_col"]

        if placed_col != self.ai_column:
            # Wrong column — restore tracker to pre-AI-turn state and warn
            if self._ai_turn_saved_board is not None:
                self.tracker.state.board          = self._ai_turn_saved_board.copy()
                self.tracker.state.current_player = self._ai_turn_saved_player
            got_1 = placed_col + 1
            want_1 = self.ai_column + 1
            print(f"WRONG COLUMN! Placed column {got_1}, AI wants column {want_1}")
            print(f"   Remove that piece and drop into column {want_1}")
            # Anchor to the wrong-but-stable board so we don't repeat the same
            # warning forever while the misplaced piece still sits there.
            self.stable.reset(board)
            ai_color = 'Red' if self.ai_player_num == 1 else 'Yellow'
            self.status_msg = (f"WRONG COL! AI wants column {want_1} ({ai_color})"
                               " - undo & retry")
            self.ann.speak(f"Wrong column. Use column {want_1}.", interrupt=True)
            return

        # Correct column — let tracker accept the move
        update = self.tracker.update(board)
        if not update["changed"]:
            if update["error"]:
                print(f"[tracker] {update['error']}")
            return

        print(f"AI piece confirmed in column {placed_col} ✓")
        self._save_screenshot(f"ai_col{placed_col}")

        if update["game_over"]:
            self._end_game(update)
            return

        # Advance stable baseline so human's next move is detected fresh
        self.stable.reset(self.tracker.state.board)

        self.ai_column = None
        self.ai_policy = None
        self._ai_turn_saved_board = None
        self.phase = GamePhase.HUMAN_TURN
        self._set_status_for_phase()
        human_color = self.COLOR_NAMES.get(self.human_player, "").lower()
        self.ann.speak(f"Your turn. Drop a {human_color} piece.")

    def _end_game(self, update: dict):
        self.phase = GamePhase.GAME_OVER
        self._save_screenshot("game_over")
        winner = update.get("winner")
        if winner == 0:
            self.status_msg = "DRAW!  Press 'r' to play again"
            print("Game over: DRAW")
            self.ann.speak("It's a draw. Well played. Press R to play again.",
                           interrupt=True)
        elif winner == self.human_player:
            self.status_msg = "YOU WIN!  Press 'r' to play again"
            print("Game over: Human wins!")
            self.ann.speak("Congratulations, you win! Press R to play again.",
                           interrupt=True)
        else:
            self.status_msg = "AI WINS!  Press 'r' to play again"
            print("Game over: AI wins!")
            self.ann.speak("I win! Good game. Press R to play again.",
                           interrupt=True)
        if update.get("winning_cells"):
            self._winning_cells = update["winning_cells"]
            self._win_anim_start = time.time()
            print(f"  Winning cells: {update['winning_cells']}")

    def _set_status_for_phase(self):
        human_color = self.COLOR_NAMES.get(self.human_player, "?")
        self.status_msg = f"Your turn ({human_color}) - drop a piece"

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_grid(self, display: np.ndarray, result):
        for row in range(6):
            for col in range(7):
                cx, cy = result.grid_centers[row, col]
                cell = result.board[row, col]
                color = self.PIECE_COLORS.get(cell, (70, 70, 70))
                label = {1: "R", 2: "Y", 0: "."}.get(cell, "?")
                cv2.circle(display, (cx, cy), 10, color, -1)
                cv2.putText(display, label, (cx - 6, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        if self._winning_cells and result.grid_centers is not None:
            self._draw_win_animation(display, result.grid_centers)

    def _draw_win_animation(self, display: np.ndarray, grid_centers: np.ndarray):
        """Pulsing green rings and connecting line over the four winning pieces only."""
        t = time.time() - self._win_anim_start

        # Radius and brightness oscillate to create the pulse effect (~2.5 Hz)
        pulse      = (np.sin(t * 5.0) + 1.0) / 2.0   # 0 → 1
        radius     = int(14 + pulse * 8)               # 14 → 22 px
        brightness = int(160 + pulse * 95)             # 160 → 255
        ring_color = (0, brightness, 0)                # green, varying brightness

        # Line connecting the four winning cells
        pts = [(int(grid_centers[r, c, 0]), int(grid_centers[r, c, 1]))
               for r, c in self._winning_cells]
        for i in range(len(pts) - 1):
            cv2.line(display, pts[i], pts[i + 1], ring_color, 3, cv2.LINE_AA)

        # Outer pulsing ring + inner glow ring (slightly out of phase) per cell
        for r, c in self._winning_cells:
            cx = int(grid_centers[r, c, 0])
            cy = int(grid_centers[r, c, 1])
            cv2.circle(display, (cx, cy), radius, ring_color, 3, cv2.LINE_AA)
            inner = int(8 + (1.0 - pulse) * 5)
            cv2.circle(display, (cx, cy), inner, (0, 255, 180), 2, cv2.LINE_AA)

    def _draw_ai_column(self, display: np.ndarray, grid_centers: np.ndarray):
        col = self.ai_column
        for row in range(6):
            cx, cy = grid_centers[row, col]
            cv2.circle(display, (cx, cy), 14, (0, 255, 0), 2)
        # Arrow above the column
        cx_top, cy_top = grid_centers[0, col]
        cv2.arrowedLine(display,
                        (cx_top, cy_top - 55),
                        (cx_top, cy_top - 18),
                        (0, 255, 0), 3, tipLength=0.4)
        cv2.putText(display, str(col + 1), (cx_top - 8, cy_top - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def _draw_policy_bars(self, display: np.ndarray, grid_centers: np.ndarray):
        for col in range(7):
            prob = float(self.ai_policy[col])
            cx, cy = grid_centers[0, col]
            bar_h = int(35 * prob)
            is_best = col == self.ai_column
            color = (0, 255, 0) if is_best else (60, 180, 60)
            bar_top = cy - 90
            cv2.rectangle(display,
                          (cx - 9, bar_top),
                          (cx + 9, bar_top - bar_h),
                          color, -1)
            if prob > 0.05:
                cv2.putText(display, f"{prob:.0%}", (cx - 13, bar_top - bar_h - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)

    def _draw_status_bar(self, display: np.ndarray, h: int, w: int):
        bar_h = 56
        overlay = display.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.78, display, 0.22, 0, display)

        phase_colors = {
            GamePhase.HUMAN_TURN: (0, 220, 0),
            GamePhase.AI_TURN:    (0, 200, 255),
            GamePhase.GAME_OVER:  (0, 80, 255),
        }
        color = phase_colors.get(self.phase, (200, 200, 200))
        cv2.putText(display, self.status_msg, (14, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> DetectionConfig:
    with open(path) as f:
        data = json.load(f)
    cfg = DetectionConfig()
    for key, val in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, tuple(val) if isinstance(val, list) else val)
    return cfg


_DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "alphazero-network-model.onnx")


def main():
    parser = argparse.ArgumentParser(description="Connect Four — cooperative game loop")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--config", type=str, help="HSV config JSON (board detection tuning)")
    parser.add_argument("--yolo", type=str, metavar="MODEL",
                        help="Path to a custom YOLO model (.pt or .engine); "
                             "defaults to bundled weights")
    parser.add_argument("--human-color", choices=["red", "yellow"], default="red")
    parser.add_argument("--stable-seconds", type=float, default=1.0,
                        help="Seconds a board state must hold before accepting a move "
                             "(default: 1.0)")
    parser.add_argument("--fps", type=float, default=15.0,
                        help="Frames per second to process (default: 15)")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable text-to-speech announcements")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-mirror", action="store_true",
                        help="Disable mirror flip (mirroring is ON by default for "
                             "selfie/built-in cameras)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose debug logging")
    args = parser.parse_args()

    mirror = not args.no_mirror
    human_player = 1 if args.human_color == "red" else 2
    ai_player_num = 3 - human_player

    print("=" * 50)
    print("  Connect Four Robot — Phase 2 Game Loop")
    print("=" * 50)
    print(f"  You:  {GameLoop.COLOR_NAMES[human_player]} (player {human_player})")
    print(f"  AI:   {GameLoop.COLOR_NAMES[ai_player_num]} (player {ai_player_num})")
    print(f"  Settle time:   {args.stable_seconds}s")
    print()

    cfg = PHYSICAL_CONFIG
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)
        print(f"HSV config: {args.config}")
    from dataclasses import replace
    if args.verbose:
        cfg = replace(cfg, verbose=True)

    yolo_path = args.yolo if args.yolo else None
    detector = YOLOEnhancedBoardDetector(model_path=yolo_path, config=cfg)
    if args.yolo:
        print(f"YOLO model override: {args.yolo}")
    else:
        print("YOLO-enhanced mode: using bundled model for piece classification")

    # Compute stable_frames from seconds × fps (minimum 3)
    stable_frames = max(3, int(round(args.stable_seconds * args.fps)))
    print(f"Stable detection: {args.stable_seconds}s × {args.fps}fps = {stable_frames} frames")

    # Set up logging
    _log_path, _ss_dir, _tee = setup_logging("logs")

    # Log YOLO load status now that the tee is active (the load message printed
    # before setup_logging, so it never appeared in previous log files).
    yolo_active = getattr(detector, '_yolo_model', None) is not None
    if yolo_active:
        print(f"[YOLOEnhanced] YOLO model active — piece classification via YOLO")
    else:
        print("[YOLOEnhanced] YOLO model NOT loaded — falling back to HSV-only mode")

    model_path = _DEFAULT_MODEL if os.path.exists(_DEFAULT_MODEL) else None
    ai       = AIPlayer(model_path=model_path, use_heuristic=(model_path is None))
    tracker  = TurnTracker(robot_player=ai_player_num)
    announcer = GameAnnouncer(enabled=not args.no_tts)
    game = GameLoop(detector, ai, tracker,
                    human_player=human_player,
                    stable_frames=stable_frames,
                    announcer=announcer)
    game.set_screenshot_dir(_ss_dir)

    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {cw}x{ch}  |  AI: {'AlphaZero (ONNX)' if not ai.use_heuristic else 'Heuristic'}")
    print(f"Mirror: {'ON' if mirror else 'OFF'}")

    # Discard the first 30 frames so the camera's auto-exposure and
    # auto-white-balance have time to settle before we attempt board detection.
    print("Camera warming up", end="", flush=True)
    for _ in range(30):
        cap.read()
        print(".", end="", flush=True)
    print(" ready")

    print("\nControls: r=reset  f=freeze  l=re-lock  SPACE=start  q=quit")
    print("Aim the camera at the board, then press SPACE to begin detection.\n")

    cv2.namedWindow(GameLoop.WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(GameLoop.WINDOW, min(cw, 960), min(ch, 560))

    # ── Orientation hold: show live feed until user presses SPACE ────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        orient_display = frame.copy()
        h_o, w_o = orient_display.shape[:2]
        msg1 = "Aim camera at the board"
        msg2 = "Press SPACE to begin detection"
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2
        for i, msg in enumerate([msg1, msg2]):
            (tw, th), _ = cv2.getTextSize(msg, font, scale, thick)
            x = (w_o - tw) // 2
            y = h_o // 2 - 20 + i * 45
            cv2.rectangle(orient_display, (x - 8, y - th - 6), (x + tw + 8, y + 8),
                          (0, 0, 0), -1)
            cv2.putText(orient_display, msg, (x, y), font, scale, (0, 220, 255), thick)
        cv2.imshow(GameLoop.WINDOW, orient_display)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord(' '), ord('q'), 27):
            if key in (ord('q'), 27):
                cap.release()
                cv2.destroyAllWindows()
                announcer.stop()
                sys.stdout = _tee._term
                _tee.close()
                return
            break   # SPACE pressed — begin detection
    print("Starting board detection...")

    frame_interval = 1.0 / args.fps
    last_process = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            now = time.time()
            if not game.frozen and (now - last_process) >= frame_interval:
                last_process = now
                try:
                    game.process_frame(frame)
                except Exception as exc:
                    print(f"[ERROR] process_frame exception: {exc}")
                    traceback.print_exc()
                    game._save_screenshot("error")

            display = game.draw(frame)

            if game.frozen:
                cv2.putText(display, "FROZEN",
                            (display.shape[1] // 2 - 70, display.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 165, 255), 3)

            cv2.imshow(GameLoop.WINDOW, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                game.reset()
            elif key == ord("l"):
                detector.unlock()
                game._initialized = False
                game.stable.reset()   # no baseline — will bootstrap from first clean frame
                print("Re-locking board...")
            elif key == ord("f"):
                game.frozen = not game.frozen
                print("Frozen" if game.frozen else "Resumed")

    finally:
        # Capture final frame if the game didn't finish (error cases, early quit)
        if game.phase != GamePhase.GAME_OVER:
            game._save_screenshot("quit_incomplete")
        quit_reason = "game_over" if game.phase == GamePhase.GAME_OVER else "user_quit"
        game._write_manifest(quit_reason=quit_reason)

        cap.release()
        cv2.destroyAllWindows()
        announcer.stop()
        sys.stdout = _tee._term   # restore real stdout before close
        _tee.close()
        print("Done. Log saved to:", _log_path)


if __name__ == "__main__":
    main()
