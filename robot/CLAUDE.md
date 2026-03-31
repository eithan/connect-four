# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A robot arm system that plays Connect Four using computer vision. Phase 2.3 is active: a cooperative live-camera game loop where a human physically places both their own piece and the AI's suggested piece.

## Common Commands

```bash
# Install dependencies
pip install opencv-python>=4.6.0 numpy>=1.21.0 onnxruntime>=1.14.0 Pillow>=9.0.0

# Run end-to-end test pipeline (uses test_images/ with ground truth .json)
python test_pipeline.py
python test_pipeline.py --image test_images/board_mid_game.png
python test_pipeline.py --sequence

# Live camera viewer with interactive HSV tuning
python camera_feed.py --tune

# Run cooperative game loop
python game_loop.py
python game_loop.py --camera 1 --human-color yellow --model path/to/model.onnx

# Regenerate synthetic test images (after changing generate_test_images.py)
python generate_test_images.py
```

## Architecture

**Data flow for every frame:**

```
Camera frame
  → BoardDetector._find_board_bbox()   # Detect blue frame via HSV + morphology
  → BoardDetector._compute_warp()      # Perspective rectify to top-down view
  → BoardDetector._find_holes()        # Find circular slots via contour analysis
  → BoardDetector._fit_grid()          # Cluster centers into 6×7 grid
  → BoardDetector._classify_cells()    # HSV sampling: red / yellow / empty
  → BoardDetector._apply_gravity_filter()
  → DetectionResult(board: 6×7 ndarray, confidence: float)
  → TurnTracker.update()               # Diff boards, validate move, detect wins
  → AIPlayer.get_move()                # ONNX inference or heuristic fallback
  → game_loop.py overlay + TTS
```

**Key classes:**
- `BoardDetector` (`board_detector.py`) — core vision pipeline; `LockedBoardDetector` wraps it with two-stage locking for stability across frames.
- `DetectionConfig` (`board_detector.py:55`) — all HSV thresholds, grid parameters, adaptive-mode gates. Two presets: `PHYSICAL_CONFIG` (default) and `SCREEN_CONFIG`.
- `TurnTracker` (`turn_tracker.py`) — stateful; compares consecutive boards, validates gravity + player alternation, detects wins.
- `AIPlayer` (`ai_player.py`) — loads ONNX AlphaZero model (3-channel: current player / empty / opponent); falls back to heuristic (win > block > center).
- `StableStateDetector` (`game_loop.py`) — requires N identical consecutive frames before committing a move (default 5 frames).
- `YOLOBoardDetector` (`board_detector_yolo.py`) — drop-in replacement for `BoardDetector` using YOLOv8; not the default pipeline.

## Color Detection (Critical)

The codebase uses **adaptive color mode** by default (`DetectionConfig.adaptive_color = True`):
- At board-lock time, samples empty-cell baseline HSV values.
- Detects pieces by ΔSaturation increase above baseline, not fixed ranges.
- Hard gates prevent false positives: red requires S ≥ 150; yellow requires S ≥ 100 AND V ≥ 100.
- `ADD_THRESHOLD = 5` (frames a new piece must persist before being accepted).

Fixed HSV ranges (fallback / SCREEN_CONFIG):
- Board (blue): H 90–140, S 80–255, V 50–255
- Red pieces: H < 12 or > 158 (wraps hue), S ≥ 130
- Yellow pieces: H 20–92, S 100–255, V 80–255

## Test Data

`test_images/` holds 8 scenarios (empty → nearly full → win states), each with a clean `.png` and a `_noisy.png`. Each has a `.json` ground truth (6×7 board array). The pipeline must score 8/8 on clean images; noisy variants are a secondary target.

## Phase Plan

- **Phase 1** ✅ Static image vision pipeline
- **Phase 2** 🔄 Live camera + cooperative game loop (current)
- **Phases 3–6** ROS2 integration, real arm motion, full autonomy
