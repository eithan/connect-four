# Connect Four Robot - Phase 1: Vision Pipeline

Computer vision + AI pipeline for a robot arm that plays Connect Four.

## Quick Start

```bash
# Install dependencies
pip install opencv-python-headless numpy onnxruntime Pillow

# Generate synthetic test board images
python generate_test_images.py

# Run the full test pipeline
python test_pipeline.py
```

## What This Does

Phase 1 implements the "brain" of the robot — everything except the physical arm:

1. **Board Detection** (`board_detector.py`): Takes a camera image of a Connect Four board and extracts the 6x7 board state using OpenCV color segmentation
2. **Turn Tracking** (`turn_tracker.py`): Compares consecutive board observations to detect moves, track turns, and detect wins/draws
3. **AI Player** (`ai_player.py`): Loads your AlphaZero ONNX model and selects the best move. Falls back to a simple heuristic if no model is available.
4. **Test Pipeline** (`test_pipeline.py`): End-to-end test harness that ties it all together

## Using Your AlphaZero Model

Copy your trained ONNX model files into the `models/` directory:

```bash
cp /path/to/connect-four/ai/src/connect_four_ai/models/alphazero-network-model.onnx models/
cp /path/to/connect-four/ai/src/connect_four_ai/models/alphazero-network-model.onnx.data models/
```

Then run: `python test_pipeline.py --model models/alphazero-network-model.onnx`

## Using Real Photos

Replace the synthetic images in `test_images/` with photos of your physical Connect Four board. You may need to tune the HSV color thresholds in `board_detector.py` for your lighting conditions.

## Project Structure

```
connect-four-robot/
├── README.md
├── generate_test_images.py   # Creates synthetic test images
├── board_detector.py         # OpenCV board state extraction
├── turn_tracker.py           # Game state tracking
├── ai_player.py              # ONNX model inference + heuristic
├── test_pipeline.py          # End-to-end test harness
├── test_images/              # Generated test images + ground truth
├── test_output/              # Debug images from test runs
└── models/                   # Place your ONNX model here
```

## Next Phases

- **Phase 2**: Live camera feed on Mac
- **Phase 3**: ROS2 nodes + Gazebo simulation
- **Phase 4**: Arm motion planning (pick up piece, drop in column)
- **Phase 5**: Real hardware integration on Jetson Orin Nano
- **Phase 6**: Natural language commands, voice interaction
