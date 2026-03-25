"""
Test Pipeline - End-to-end test of the Connect Four vision + AI system.

Usage:
    python test_pipeline.py              # Test all images + game sequence
    python test_pipeline.py --image X    # Test single image
    python test_pipeline.py --sequence   # Run game sequence only
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np

from board_detector import BoardDetector, board_to_string
from turn_tracker import TurnTracker
from ai_player import AIPlayer


def test_single_image(image_path, detector, ai, output_dir):
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(image_path)}")
    print(f"{'='*60}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"  ERROR: Could not load image")
        return {'error': 'load failed'}

    result = detector.detect(image, debug=True)

    print(f"\n  Detection confidence: {result.confidence:.2f}")
    for err in result.errors:
        print(f"  ERROR: {err}")
    print(f"\n  Detected board:\n  " + board_to_string(result.board).replace("\n", "\n  "))

    # Check ground truth
    json_path = image_path.rsplit('.', 1)[0] + '.json'
    accuracy = None
    if os.path.exists(json_path):
        with open(json_path) as f:
            gt = json.load(f)
        gt_board = np.array(gt['board'], dtype=np.int8)
        matches = np.sum(result.board == gt_board)
        accuracy = matches / 42
        print(f"\n  Ground truth:\n  " + board_to_string(gt_board).replace("\n", "\n  "))
        print(f"\n  Accuracy: {matches}/42 ({accuracy*100:.1f}%)")
        if accuracy < 1.0:
            for r, c in np.argwhere(result.board != gt_board):
                print(f"    Mismatch ({r},{c}): got={result.board[r,c]} want={gt_board[r,c]}")

    # Turn and AI analysis
    red = int(np.sum(result.board == 1))
    yellow = int(np.sum(result.board == 2))
    current = 1 if red == yellow else (2 if red == yellow + 1 else -1)
    names = {1: "Red", 2: "Yellow", -1: "Invalid"}
    print(f"\n  Pieces: Red={red}, Yellow={yellow}")
    print(f"  Next to play: {names.get(current, '?')}")

    tracker = TurnTracker(robot_player=2)
    tracker.set_board(result.board)

    if tracker.state.game_over:
        w = tracker.state.winner
        print(f"  Game status: {'DRAW' if w == 0 else names[w] + ' WINS!'}")
        if tracker.state.winning_cells:
            print(f"  Winning cells: {tracker.state.winning_cells}")
    elif current > 0:
        move, info = ai.get_move(result.board, current)
        print(f"\n  AI recommendation ({info['method']}): column {move}")
        print(f"    Policy: {np.array2string(info['policy'], precision=3)}")

    if result.debug_image is not None:
        path = os.path.join(output_dir, "debug_" + os.path.basename(image_path))
        cv2.imwrite(path, result.debug_image)
        print(f"\n  Debug image: {path}")

    return {'image': os.path.basename(image_path), 'confidence': result.confidence,
            'accuracy': accuracy, 'errors': result.errors}


def test_game_sequence(detector, ai, output_dir):
    print(f"\n{'='*60}")
    print("Simulating game sequence (turn tracking)")
    print(f"{'='*60}")

    from generate_test_images import board_to_image

    tracker = TurnTracker(robot_player=2)
    moves = [3, 4, 3, 2, 3, 5]
    board = np.zeros((6, 7), dtype=np.int8)

    for i, col in enumerate(moves):
        player = 1 if i % 2 == 0 else 2
        name = "Red" if player == 1 else "Yellow"
        for row in range(5, -1, -1):
            if board[row, col] == 0:
                board[row, col] = player
                break

        print(f"\n  Move {i+1}: {name} plays column {col}")
        img = board_to_image(board)
        result = detector.detect(img)
        update = tracker.update(result.board)
        print(f"    Changed: {update['changed']}, Col: {update['move_col']}, "
              f"Robot's turn: {update['is_robot_turn']}")
        if update['error']:
            print(f"    ERROR: {update['error']}")
        if update['is_robot_turn']:
            move, info = ai.get_move(result.board, tracker.robot_player)
            print(f"    >>> Robot would play: column {move} ({info['method']})")
        if update['game_over']:
            w = {0: "Draw", 1: "Red", 2: "Yellow"}.get(update['winner'], "?")
            print(f"    GAME OVER - Winner: {w}")
            break


def main():
    parser = argparse.ArgumentParser(description="Test Connect Four vision pipeline")
    parser.add_argument("--image", type=str, help="Test a single image")
    parser.add_argument("--model", type=str, help="Path to ONNX model")
    parser.add_argument("--sequence", action="store_true", help="Run game sequence only")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(base_dir, "test_images")
    out_dir = os.path.join(base_dir, "test_output")
    os.makedirs(out_dir, exist_ok=True)

    detector = BoardDetector()
    ai = AIPlayer(model_path=args.model, use_heuristic=(args.model is None))

    print("Connect Four Robot - Vision Pipeline Test")
    print(f"  AI mode: {'ONNX' if not ai.use_heuristic else 'Heuristic (no model)'}")

    if args.image:
        test_single_image(args.image, detector, ai, out_dir)
    elif args.sequence:
        test_game_sequence(detector, ai, out_dir)
    else:
        files = sorted(f for f in os.listdir(test_dir)
                       if f.endswith('.png') and 'noisy' not in f)
        if not files:
            print(f"\nNo images found. Run: python generate_test_images.py")
            sys.exit(1)

        results = []
        for f in files:
            r = test_single_image(os.path.join(test_dir, f), detector, ai, out_dir)
            results.append(r)

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        total = len(results)
        perfect = sum(1 for r in results if r.get('accuracy') == 1.0)
        print(f"  Images: {total}, Perfect accuracy: {perfect}/{total}")

        test_game_sequence(detector, ai, out_dir)


if __name__ == "__main__":
    main()
