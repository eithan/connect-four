"""
Game Loop V2 — Cooperative Connect Four with calibration-first detection.

Changes from game_loop.py
--------------------------
* Uses LockedBoardDetectorV2 (board_detector_v2.py) instead of v1.
* Adds a mandatory CALIBRATING phase between board-lock and game start:
    1. Camera searches for the board (blue frame detection).
    2. Once locked, the user ensures the board is empty, then presses [C].
    3. The detector captures per-cell HSV baselines.
    4. Game starts.  All piece detection now measures *change* from the
       baseline, not absolute colour — static background is ignored.

Usage
-----
    python3 game_loop_v2.py
    python3 game_loop_v2.py --camera 1
    python3 game_loop_v2.py --human-color yellow
    python3 game_loop_v2.py --model path/to/model.onnx
    python3 game_loop_v2.py --no-tts
    python3 game_loop_v2.py --fps 15 --stable-seconds 1.0

Controls
--------
    SPACE    Start detection (after aiming camera)
    C        Calibrate empty board (when board is locked, before game starts)
    R        Reset game (keeps calibration)
    L        Re-lock board (clears lock AND calibration — full restart)
    F        Freeze / unfreeze frame
    Q / ESC  Quit
"""

import argparse
import os
import sys
import time
import traceback
from datetime import datetime

import cv2
import numpy as np

from board_detector_v2 import LockedBoardDetectorV2, DetectionConfigV2
from turn_tracker import TurnTracker
from ai_player import AIPlayer
from tts_announcer import GameAnnouncer

# Reuse stable-state detector and game loop logic from v1 unchanged.
from game_loop import (
    GameLoop,
    StableStateDetector,
    GamePhase,
    setup_logging,
)

WINDOW = "Connect Four V2"


# ─────────────────────────────────────────────────────────────────────────────
# Overlay helpers (pre-game phases)
# ─────────────────────────────────────────────────────────────────────────────

def _centred_text(img: np.ndarray, lines: list, start_y: int,
                  font=cv2.FONT_HERSHEY_SIMPLEX, scale: float = 0.75,
                  color=(255, 255, 255), thickness: int = 2, line_gap: int = 40):
    h, w = img.shape[:2]
    for i, (text, col) in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (w - tw) // 2
        y = start_y + i * line_gap
        cv2.rectangle(img, (x - 8, y - th - 6), (x + tw + 8, y + 8),
                      (0, 0, 0), -1)
        cv2.putText(img, text, (x, y), font, scale, col, thickness)


def draw_searching(frame: np.ndarray, board_found: bool) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    if board_found:
        lines = [("Board visible — locking...", (0, 200, 100))]
    else:
        lines = [("Searching for Connect Four board...", (0, 165, 255)),
                 ("Aim camera so the full blue frame is visible", (180, 180, 180))]
    _centred_text(img, lines, h // 2 - 20)
    return img


def draw_calibrating(frame: np.ndarray, result, fps: float) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]

    # Draw detected contour
    if result is not None and result.board_contour is not None:
        cv2.drawContours(img, [result.board_contour], -1, (0, 255, 80), 2)

    # Draw grid overlay so user can verify alignment
    if result is not None and result.grid_centers is not None:
        for r in range(6):
            for c in range(7):
                cx, cy = result.grid_centers[r, c]
                cv2.circle(img, (int(cx), int(cy)), 12, (0, 200, 100), 1)

    # Status bar at bottom
    bar_h = 75
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    cv2.putText(img,
                "Board locked!  Make sure board is EMPTY, then press [C] to calibrate",
                (12, h - bar_h + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 150), 2)
    cv2.putText(img,
                "Press [L] to re-lock if the grid looks wrong",
                (12, h - bar_h + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1)

    # FPS top-right
    cv2.putText(img, f"FPS:{fps:.1f}  LOCKED",
                (w - 200, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 220, 0), 1)
    return img


def draw_calibrated_flash(frame: np.ndarray) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    _centred_text(img,
                  [("Calibrated!  Starting game...", (0, 255, 120))],
                  h // 2)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Connect Four V2 — calibration-first cooperative game loop"
    )
    parser.add_argument("--camera",  type=int, default=0)
    parser.add_argument("--model",   type=str, help="Path to AlphaZero ONNX model")
    parser.add_argument("--human-color", choices=["red", "yellow"], default="red")
    parser.add_argument("--stable-seconds", type=float, default=1.5,
                        help="Seconds a board state must hold to confirm a move (default: 1.5)")
    parser.add_argument("--fps",     type=float, default=10.0,
                        help="Processing frame rate (default: 10)")
    parser.add_argument("--no-tts",  action="store_true")
    parser.add_argument("--tts-rate", type=int, default=155)
    parser.add_argument("--tts-voice", type=str, default="")
    parser.add_argument("--width",   type=int, default=1280)
    parser.add_argument("--height",  type=int, default=720)
    parser.add_argument("--mirror",  action="store_true",
                        help="Flip frame horizontally (selfie cameras)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    human_player  = 1 if args.human_color == "red" else 2
    ai_player_num = 3 - human_player
    COLOR_NAMES   = {1: "Red", 2: "Yellow"}

    print("=" * 55)
    print("  Connect Four Robot — V2 (calibration-first)")
    print("=" * 55)
    print(f"  You:  {COLOR_NAMES[human_player]}  (player {human_player})")
    print(f"  AI:   {COLOR_NAMES[ai_player_num]}  (player {ai_player_num})")
    print(f"  Settle time: {args.stable_seconds}s  |  FPS: {args.fps}")
    print()

    # Set up logging
    _log_path, _ss_dir, _tee = setup_logging("logs")

    # Build components
    cfg      = DetectionConfigV2()
    detector = LockedBoardDetectorV2(cfg)
    ai       = AIPlayer(model_path=args.model,
                        use_heuristic=(args.model is None))
    tracker  = TurnTracker(robot_player=ai_player_num)
    announcer = GameAnnouncer(rate=args.tts_rate,
                              voice_id=args.tts_voice,
                              enabled=not args.no_tts)

    stable_frames = max(3, int(round(args.stable_seconds * args.fps)))
    print(f"Stable detection: {args.stable_seconds}s × {args.fps}fps = {stable_frames} frames")

    game = GameLoop(
        detector,
        ai,
        tracker,
        human_player=human_player,
        stable_frames=stable_frames,
        announcer=announcer,
    )
    game.set_screenshot_dir(_ss_dir)

    # Open camera
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {cw}×{ch}  |  "
          f"AI: {'AlphaZero (ONNX)' if not ai.use_heuristic else 'Heuristic'}")

    # Warm up auto-exposure / AWB
    print("Camera warming up", end="", flush=True)
    for _ in range(30):
        cap.read()
        print(".", end="", flush=True)
    print(" ready")

    print("\nControls: SPACE=start  C=calibrate  R=reset  L=re-lock  F=freeze  Q=quit")
    print("Point the camera at the full board, then press SPACE.\n")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(cw, 960), min(ch, 560))

    # ── Orientation hold ──────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)
        disp = frame.copy()
        h_o, w_o = disp.shape[:2]
        for i, (msg, col) in enumerate([
            ("Aim camera at the full Connect Four board", (255, 255, 255)),
            ("Press SPACE to begin", (0, 220, 255)),
        ]):
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            x = (w_o - tw) // 2
            y = h_o // 2 - 20 + i * 50
            cv2.rectangle(disp, (x - 8, y - th - 6), (x + tw + 8, y + 8), (0, 0, 0), -1)
            cv2.putText(disp, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        cv2.imshow(WINDOW, disp)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            cap.release()
            cv2.destroyAllWindows()
            announcer.stop()
            sys.stdout = _tee._term
            _tee.close()
            return
        if key == ord(' '):
            break
    print("Starting board detection...")

    # ── Main loop ─────────────────────────────────────────────────────────────

    # outer_state drives the pre-game lifecycle; GameLoop handles the game itself.
    outer_state = "SEARCHING"   # SEARCHING | CALIBRATING | PLAYING

    frame_interval = 1.0 / args.fps
    last_process   = 0.0
    last_result    = None
    fps_actual     = 0.0
    fps_t          = time.time()
    fps_count      = 0

    # Show a brief flash after calibration succeeds
    cal_flash_until = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            now = time.time()

            # FPS counter
            fps_count += 1
            if now - fps_t >= 1.0:
                fps_actual = fps_count / (now - fps_t)
                fps_count  = 0
                fps_t      = now

            # ── Processing ────────────────────────────────────────────────
            if (now - last_process) >= frame_interval:
                last_process = now

                if outer_state == "SEARCHING":
                    last_result = detector.detect(frame)
                    if detector.is_locked:
                        outer_state = "CALIBRATING"
                        print("Board locked. Make the board empty, then press [C].")
                        announcer.speak("Board detected. Clear the board, then press C to calibrate.",
                                        interrupt=True)

                elif outer_state == "CALIBRATING":
                    # Keep running detection so the grid overlay stays fresh
                    last_result = detector.detect(frame)

                elif outer_state == "PLAYING":
                    if not game.frozen:
                        try:
                            game.process_frame(frame)
                        except Exception as exc:
                            print(f"[ERROR] process_frame: {exc}")
                            traceback.print_exc()
                            game._save_screenshot("error")

            # ── Rendering ─────────────────────────────────────────────────
            if outer_state == "SEARCHING":
                board_found = (last_result is not None and
                               last_result.confidence > 0.1)
                display = draw_searching(frame, board_found)

            elif outer_state == "CALIBRATING":
                if now < cal_flash_until:
                    display = draw_calibrated_flash(frame)
                else:
                    display = draw_calibrating(frame, last_result, fps_actual)

            else:  # PLAYING
                if now < cal_flash_until:
                    display = draw_calibrated_flash(frame)
                else:
                    display = game.draw(frame)
                    if game.frozen:
                        h_d, w_d = display.shape[:2]
                        cv2.putText(display, "FROZEN",
                                    (w_d // 2 - 70, h_d // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 165, 255), 3)

            cv2.imshow(WINDOW, display)

            # ── Key handling ──────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            elif key == ord('c'):
                if outer_state == "CALIBRATING":
                    print("Calibrating...")
                    if detector.calibrate(frame):
                        print("Calibration successful — starting game.")
                        announcer.speak("Calibrated. Your turn.", interrupt=True)
                        cal_flash_until = now + 1.5
                        outer_state = "PLAYING"
                        # Reset game stable detector so it bootstraps cleanly
                        game._initialized = False
                        game._prev_locked  = False
                        game.stable.reset()
                    else:
                        print("Calibration failed — board not locked?")
                else:
                    print("[C] only active during calibration phase.")

            elif key == ord('l'):
                print("Re-locking board...")
                detector.unlock()
                outer_state       = "SEARCHING"
                last_result       = None
                game._initialized = False
                game._prev_locked = False
                game.stable.reset()
                # Reset game state too so it doesn't resume mid-game with stale data
                game.tracker.reset()
                game.phase     = GamePhase.HUMAN_TURN
                game.ai_column = None
                game.status_msg = "Re-locking..."

            elif key == ord('r'):
                if outer_state == "PLAYING":
                    game.reset()
                    print("Game reset.")

            elif key == ord('f'):
                if outer_state == "PLAYING":
                    game.frozen = not game.frozen
                    print("Frozen" if game.frozen else "Resumed")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        announcer.stop()
        sys.stdout = _tee._term
        _tee.close()
        print(f"Done. Log saved to: {_log_path}")


if __name__ == "__main__":
    main()
