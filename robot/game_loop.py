"""
Game Loop - Full cooperative Connect Four game via live camera.

Phase 2.3 of the Connect Four Robot project.

COOPERATIVE MODE: You place both your piece and the AI's piece.
The system detects your move, the AI picks a column, and you drop the AI's
piece there. The camera confirms the placement.

Usage:
    python3 game_loop.py
    python3 game_loop.py --camera 1
    python3 game_loop.py --model ../web/public/alphazero-network-model.onnx
    python3 game_loop.py --config hsv_config.json
    python3 game_loop.py --human-color yellow   # Play as yellow (AI is red)
    python3 game_loop.py --stable-frames 5      # Frames to confirm a move

Controls:
    q / ESC  - Quit
    r        - Reset game
    f        - Freeze / unfreeze frame
"""

import cv2
import numpy as np
import argparse
import json
import time
import os
import sys
from enum import Enum
from typing import Optional

from board_detector import BoardDetector, LockedBoardDetector, DetectionConfig, SCREEN_CONFIG, PHYSICAL_CONFIG
from board_detector_yolo import YOLOBoardDetector
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
        self._initialized = False   # True after first stable board observed

        self.fps_actual = 0.0
        self._fps_t = time.time()
        self._fps_count = 0

        self._set_status_for_phase()

    # ── Public API ────────────────────────────────────────────────────────────

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
        self._set_status_for_phase()
        self.ann.speak("New game. Your turn.", interrupt=True)
        print("\n" + "=" * 50)
        print("  Game reset — your turn!")
        print("=" * 50 + "\n")

    def process_frame(self, frame: np.ndarray):
        """Run detection + state machine. Call at your desired FPS."""
        now = time.time()
        self._fps_count += 1
        if now - self._fps_t >= 1.0:
            self.fps_actual = self._fps_count / (now - self._fps_t)
            self._fps_count = 0
            self._fps_t = now

        result = self.detector.detect(frame)
        self.last_result = result

        if self.phase == GamePhase.GAME_OVER:
            return

        is_stable, stable_board = self.stable.update(result.board, result.confidence)
        if not is_stable:
            return

        # ── First stable frame: seed tracker with current board state ─────────
        if not self._initialized:
            self._initialized = True
            self.tracker.set_board(stable_board)
            # Update stable baseline so detector knows what "no change" looks like
            self.stable.reset(stable_board)
            red_n    = int(np.sum(stable_board == 1))
            yellow_n = int(np.sum(stable_board == 2))
            print(f"\nInitial board detected: Red={red_n}, Yellow={yellow_n}")
            self.ann.speak("Board locked. Ready to play.")
            if self.tracker.state.game_over:
                self._end_game({"game_over": True,
                                "winner": self.tracker.state.winner,
                                "winning_cells": self.tracker.state.winning_cells})
                return
            # If it's already the AI's turn when we start, compute immediately
            if self.tracker.state.current_player == self.ai_player_num and red_n + yellow_n > 0:
                print("AI's turn on startup — computing move...")
                self.phase = GamePhase.AI_TURN
                self._run_ai(stable_board)
            else:
                self._set_status_for_phase()
            return

        if self.phase == GamePhase.HUMAN_TURN:
            self._handle_human_turn(stable_board)
        elif self.phase == GamePhase.AI_TURN:
            self._handle_ai_placement(stable_board)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Render the full game overlay onto a camera frame."""
        display = frame.copy()
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
        self.status_msg = f"AI plays col {ai_col} ({ai_color}) ↑ drop the AI's piece there"
        # Announce with column number (1-indexed feels more natural to say aloud)
        self.ann.speak(f"My move: column {ai_col + 1}. "
                       f"Please drop a {ai_color.lower()} piece there.", interrupt=True)

    def _handle_human_turn(self, board: np.ndarray):
        update = self.tracker.update(board)

        if not update["changed"]:
            if update["error"]:
                print(f"[tracker] {update['error']}")
            return

        col = update["move_col"]
        print(f"\nHuman played column {col}")

        if update["game_over"]:
            self._end_game(update)
            return

        # Advance stable baseline to the newly accepted board state so the
        # detector doesn't re-trigger on the same position next turn.
        self.stable.reset(self.tracker.state.board)

        # Trigger AI
        self._run_ai(board)

    def _handle_ai_placement(self, board: np.ndarray):
        # Peek at what changed before letting tracker advance
        diff = board - self.tracker.state.board
        new_cells = np.argwhere(diff != 0)

        if len(new_cells) == 0:
            return

        placed_col = int(new_cells[0][1]) if len(new_cells) == 1 else -1

        if placed_col != self.ai_column:
            # Wrong column — restore tracker to pre-AI-turn state and warn
            if self._ai_turn_saved_board is not None:
                self.tracker.state.board          = self._ai_turn_saved_board.copy()
                self.tracker.state.current_player = self._ai_turn_saved_player
            print(f"⚠  Wrong column! Placed col {placed_col}, AI wants col {self.ai_column}")
            print(f"   Remove that piece and drop into column {self.ai_column}")
            self.stable.reset()   # discard this stable snapshot
            ai_color = 'Red' if self.ai_player_num == 1 else 'Yellow'
            self.status_msg = (f"⚠ Wrong column! AI wants col {self.ai_column} ({ai_color})"
                               f" — undo & retry")
            self.ann.speak(f"Wrong column. Please undo that and use column "
                           f"{self.ai_column + 1}.", interrupt=True)
            return

        # Correct column — let tracker accept the move
        update = self.tracker.update(board)
        if not update["changed"]:
            if update["error"]:
                print(f"[tracker] {update['error']}")
            return

        print(f"AI piece confirmed in column {placed_col} ✓")

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
            print(f"  Winning cells: {update['winning_cells']}")

    def _set_status_for_phase(self):
        human_color = self.COLOR_NAMES.get(self.human_player, "?")
        self.status_msg = f"Your turn ({human_color}) — drop a piece"

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
        cv2.putText(display, str(col), (cx_top - 8, cy_top - 60),
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


def main():
    parser = argparse.ArgumentParser(description="Connect Four — cooperative game loop")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=str, help="Path to AlphaZero ONNX model")
    parser.add_argument("--config", type=str, help="HSV config JSON")
    parser.add_argument("--screen", action="store_true",
                        help="Use screen-optimised thresholds (phone/monitor display)")
    parser.add_argument("--yolo", type=str, metavar="MODEL",
                        help="Path to YOLOv8 model (.pt or .engine) — uses YOLO detector")
    parser.add_argument("--human-color", choices=["red", "yellow"], default="red")
    parser.add_argument("--stable-frames", type=int, default=3,
                        help="Consecutive frames required to confirm a move (default: 3)")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable text-to-speech announcements")
    parser.add_argument("--tts-rate", type=int, default=155,
                        help="TTS speaking rate in words per minute (default: 155)")
    parser.add_argument("--tts-voice", type=str, default="",
                        help="pyttsx3 voice ID (leave blank for system default)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    human_player = 1 if args.human_color == "red" else 2
    ai_player_num = 3 - human_player

    print("=" * 50)
    print("  Connect Four Robot — Phase 2 Game Loop")
    print("=" * 50)
    print(f"  You:  {GameLoop.COLOR_NAMES[human_player]} (player {human_player})")
    print(f"  AI:   {GameLoop.COLOR_NAMES[ai_player_num]} (player {ai_player_num})")
    print(f"  Stable frames: {args.stable_frames}")
    print()

    if args.yolo:
        detector = YOLOBoardDetector(model_path=args.yolo)
        print(f"YOLO mode: {args.yolo}")
    else:
        if args.screen:
            cfg = SCREEN_CONFIG
            print("Screen mode: using display-optimised HSV thresholds")
            print("NOTE: --screen is tuned for phone/monitor displays. For a real plastic board,")
            print("      omit --screen and use --config hsv_config.json (or run without flags).")
        else:
            cfg = PHYSICAL_CONFIG
        if args.config and os.path.exists(args.config):
            cfg = load_config(args.config)
            print(f"HSV config: {args.config}")
        detector = LockedBoardDetector(cfg)
    ai       = AIPlayer(model_path=args.model, use_heuristic=(args.model is None))
    tracker  = TurnTracker(robot_player=ai_player_num)
    announcer = GameAnnouncer(
        rate=args.tts_rate,
        voice_id=args.tts_voice,
        enabled=not args.no_tts,
    )
    game = GameLoop(detector, ai, tracker,
                    human_player=human_player,
                    stable_frames=args.stable_frames,
                    announcer=announcer)

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

    # Discard the first 30 frames so the camera's auto-exposure and
    # auto-white-balance have time to settle before we attempt board detection.
    print("Camera warming up", end="", flush=True)
    for _ in range(30):
        cap.read()
        print(".", end="", flush=True)
    print(" ready")

    print("\nPoint the camera at your Connect Four board.")
    print("Controls: r=reset  f=freeze  l=re-lock  q=quit\n")

    cv2.namedWindow(GameLoop.WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(GameLoop.WINDOW, min(cw, 960), min(ch, 560))

    frame_interval = 1.0 / args.fps
    last_process = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break

            now = time.time()
            if not game.frozen and (now - last_process) >= frame_interval:
                last_process = now
                game.process_frame(frame)

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
        cap.release()
        cv2.destroyAllWindows()
        announcer.stop()
        print("Done.")


if __name__ == "__main__":
    main()
