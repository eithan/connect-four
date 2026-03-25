"""
Camera Feed - Live board detection viewer with optional HSV tuning.

Phase 2.1 / 2.2 of the Connect Four Robot project.

Usage:
    python3 camera_feed.py                          # Basic live view
    python3 camera_feed.py --camera 1               # Use camera index 1
    python3 camera_feed.py --tune                   # Interactive HSV tuning
    python3 camera_feed.py --config my_config.json  # Load saved HSV config
    python3 camera_feed.py --save my_config.json    # Save config on 's' keypress

Controls:
    q / ESC  - Quit
    s        - Save current HSV config to file
    f        - Freeze / unfreeze frame
"""

import cv2
import numpy as np
import argparse
import json
import time
import os
import sys

from board_detector import BoardDetector, DetectionConfig, board_to_string

WINDOW_MAIN = "Connect Four - Camera Feed"
WINDOW_TUNE = "Connect Four - HSV Tuning"
DEFAULT_FPS = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> DetectionConfig:
    with open(path) as f:
        data = json.load(f)
    cfg = DetectionConfig()
    for key, val in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, tuple(val) if isinstance(val, list) else val)
    return cfg


def save_config(cfg: DetectionConfig, path: str):
    data = {
        "board_hsv_low":   list(cfg.board_hsv_low),
        "board_hsv_high":  list(cfg.board_hsv_high),
        "red_hsv_low1":    list(cfg.red_hsv_low1),
        "red_hsv_high1":   list(cfg.red_hsv_high1),
        "red_hsv_low2":    list(cfg.red_hsv_low2),
        "red_hsv_high2":   list(cfg.red_hsv_high2),
        "yellow_hsv_low":  list(cfg.yellow_hsv_low),
        "yellow_hsv_high": list(cfg.yellow_hsv_high),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[camera_feed] Config saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# HSV tuning trackbars
# ─────────────────────────────────────────────────────────────────────────────

def setup_tuning_trackbars(cfg: DetectionConfig):
    cv2.namedWindow(WINDOW_TUNE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TUNE, 520, 640)

    params = [
        # Board (blue frame)
        ("Board H low",   cfg.board_hsv_low[0],  180),
        ("Board H high",  cfg.board_hsv_high[0], 180),
        ("Board S low",   cfg.board_hsv_low[1],  255),
        ("Board S high",  cfg.board_hsv_high[1], 255),
        ("Board V low",   cfg.board_hsv_low[2],  255),
        ("Board V high",  cfg.board_hsv_high[2], 255),
        # Red pieces (two HSV ranges to handle wrap-around)
        ("Red1 H low",    cfg.red_hsv_low1[0],   180),
        ("Red1 H high",   cfg.red_hsv_high1[0],  180),
        ("Red S low",     cfg.red_hsv_low1[1],   255),
        ("Red S high",    cfg.red_hsv_high1[1],  255),
        ("Red V low",     cfg.red_hsv_low1[2],   255),
        ("Red2 H low",    cfg.red_hsv_low2[0],   180),
        ("Red2 H high",   cfg.red_hsv_high2[0],  180),
        # Yellow pieces
        ("Yellow H low",  cfg.yellow_hsv_low[0],  180),
        ("Yellow H high", cfg.yellow_hsv_high[0], 180),
        ("Yellow S low",  cfg.yellow_hsv_low[1],  255),
        ("Yellow S high", cfg.yellow_hsv_high[1], 255),
        ("Yellow V low",  cfg.yellow_hsv_low[2],  255),
    ]

    def noop(_): pass
    for name, val, max_val in params:
        cv2.createTrackbar(name, WINDOW_TUNE, int(val), int(max_val), noop)

    cv2.imshow(WINDOW_TUNE, np.zeros((40, 520, 3), dtype=np.uint8))


def read_trackbar_config() -> DetectionConfig:
    def tb(name): return cv2.getTrackbarPos(name, WINDOW_TUNE)

    sv_low  = tb("Red S low")
    sv_high = tb("Red S high")
    rv_low  = tb("Red V low")

    return DetectionConfig(
        board_hsv_low=(tb("Board H low"),  tb("Board S low"),  tb("Board V low")),
        board_hsv_high=(tb("Board H high"), tb("Board S high"), tb("Board V high")),
        red_hsv_low1=(tb("Red1 H low"),  sv_low,  rv_low),
        red_hsv_high1=(tb("Red1 H high"), sv_high, 255),
        red_hsv_low2=(tb("Red2 H low"),  sv_low,  rv_low),
        red_hsv_high2=(tb("Red2 H high"), sv_high, 255),
        yellow_hsv_low=(tb("Yellow H low"),  tb("Yellow S low"),  tb("Yellow V low")),
        yellow_hsv_high=(tb("Yellow H high"), tb("Yellow S high"), 255),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Overlay rendering
# ─────────────────────────────────────────────────────────────────────────────

def draw_overlay(frame: np.ndarray, result, fps: float) -> np.ndarray:
    display = frame.copy()
    h, w = display.shape[:2]

    if result.board_contour is not None:
        cv2.drawContours(display, [result.board_contour], -1, (0, 255, 0), 2)

    if result.grid_centers is not None:
        for row in range(6):
            for col in range(7):
                cx, cy = result.grid_centers[row, col]
                cell = result.board[row, col]
                if cell == 1:
                    color, label = (0, 0, 255), "R"
                elif cell == 2:
                    color, label = (0, 220, 255), "Y"
                else:
                    color, label = (90, 90, 90), "."
                cv2.circle(display, (cx, cy), 9, color, -1)
                cv2.putText(display, label, (cx - 6, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    conf = result.confidence
    conf_color = (0, 255, 0) if conf > 0.7 else (0, 165, 255) if conf > 0.4 else (0, 0, 255)
    cv2.putText(display, f"Conf: {conf:.2f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
    cv2.putText(display, f"FPS: {fps:.1f}", (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    for i, err in enumerate(result.errors):
        cv2.putText(display, f"ERR: {err}", (10, 82 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

    red_n   = int(np.sum(result.board == 1))
    yellow_n = int(np.sum(result.board == 2))
    cv2.putText(display, f"R:{red_n}  Y:{yellow_n}", (w - 130, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    cv2.putText(display, "q=quit  f=freeze  s=save", (w - 260, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    return display


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live Connect Four board detection viewer")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--tune", action="store_true", help="Enable interactive HSV tuning")
    parser.add_argument("--config", type=str, help="Load HSV config from JSON")
    parser.add_argument("--save", type=str, default="hsv_config.json",
                        help="Path to save HSV config (default: hsv_config.json)")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    cfg = DetectionConfig()
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)
        print(f"Loaded config: {args.config}")

    detector = BoardDetector(cfg)

    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {args.camera} opened: {aw}x{ah}")
    if args.tune:
        print("Tuning mode active — adjust trackbars to match your lighting, press 's' to save")
    print("Controls: q/ESC=quit  f=freeze  s=save config")

    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_MAIN, min(aw, 960), min(ah, 540))

    if args.tune:
        setup_tuning_trackbars(cfg)

    frame_interval = 1.0 / args.fps
    last_process = 0.0
    last_result = None
    frozen = False
    fps_actual = 0.0
    fps_t = time.time()
    fps_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed — check connection")
                break

            now = time.time()

            if args.tune:
                cfg = read_trackbar_config()
                detector = BoardDetector(cfg)

            if not frozen and (now - last_process) >= frame_interval:
                last_process = now
                last_result = detector.detect(frame)
                fps_count += 1
                if now - fps_t >= 1.0:
                    fps_actual = fps_count / (now - fps_t)
                    fps_count = 0
                    fps_t = now

            if last_result is not None:
                display = draw_overlay(frame, last_result, fps_actual)
            else:
                display = frame.copy()

            if frozen:
                h = display.shape[0]
                cv2.putText(display, "FROZEN", (20, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 165, 255), 3)

            cv2.imshow(WINDOW_MAIN, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("f"):
                frozen = not frozen
                print("Frozen" if frozen else "Resumed")
            elif key == ord("s"):
                save_config(detector.config, args.save)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed.")


if __name__ == "__main__":
    main()
