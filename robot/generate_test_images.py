"""
Generate synthetic Connect Four board images for testing the vision pipeline.

Usage:
    python generate_test_images.py

Output:
    test_images/board_*.png  - Rendered board images
    test_images/board_*.json - Ground truth board states
"""

import cv2
import numpy as np
import json
import os

ROWS = 6
COLS = 7

# Colors (BGR)
BOARD_BLUE = (180, 100, 20)
EMPTY_SLOT = (220, 220, 220)
RED_PIECE = (40, 40, 200)
YELLOW_PIECE = (30, 210, 230)
BACKGROUND = (240, 240, 235)

CELL_SIZE = 80
PIECE_RADIUS = 30
BORDER = 40
BOARD_PADDING = 10


def board_to_image(board, add_noise=False, angle_deg=0, brightness_shift=0):
    board_w = COLS * CELL_SIZE + 2 * BOARD_PADDING
    board_h = ROWS * CELL_SIZE + 2 * BOARD_PADDING
    img_w = board_w + 2 * BORDER
    img_h = board_h + 2 * BORDER + 50

    img = np.full((img_h, img_w, 3), BACKGROUND, dtype=np.uint8)
    bx, by = BORDER, BORDER + 50
    cv2.rectangle(img, (bx, by), (bx + board_w, by + board_h), BOARD_BLUE, -1)

    for row in range(ROWS):
        for col in range(COLS):
            cx = bx + BOARD_PADDING + col * CELL_SIZE + CELL_SIZE // 2
            cy = by + BOARD_PADDING + row * CELL_SIZE + CELL_SIZE // 2
            cell_val = board[row, col]
            color = EMPTY_SLOT if cell_val == 0 else (RED_PIECE if cell_val == 1 else YELLOW_PIECE)
            cv2.circle(img, (cx, cy), PIECE_RADIUS, color, -1)
            if cell_val != 0:
                cv2.circle(img, (cx, cy), PIECE_RADIUS, (0, 0, 0), 2)
                cv2.circle(img, (cx - PIECE_RADIUS // 3, cy - PIECE_RADIUS // 3),
                          PIECE_RADIUS // 5, (255, 255, 255), -1)
            else:
                cv2.circle(img, (cx, cy), PIECE_RADIUS, (180, 180, 180), 2)

    for col in range(COLS):
        cx = bx + BOARD_PADDING + col * CELL_SIZE + CELL_SIZE // 2
        cv2.putText(img, str(col), (cx - 8, by - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    if brightness_shift != 0:
        img = np.clip(img.astype(np.int16) + brightness_shift, 0, 255).astype(np.uint8)
    if add_noise:
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if angle_deg != 0:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=BACKGROUND)
    return img


def create_test_scenarios():
    scenarios = []

    board = np.zeros((ROWS, COLS), dtype=np.int8)
    scenarios.append(("empty", board.copy(), "Empty board"))

    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[5, 3] = 1; board[5, 4] = 2; board[4, 3] = 1
    board[5, 2] = 2; board[5, 5] = 1; board[3, 3] = 2
    scenarios.append(("early_game", board.copy(), "Early game - 3 moves each"))

    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[5, 3] = 1; board[5, 4] = 2; board[5, 2] = 1; board[5, 5] = 2
    board[4, 3] = 1; board[4, 4] = 2; board[5, 1] = 1; board[4, 2] = 2
    board[3, 3] = 1; board[5, 6] = 2; board[5, 0] = 1; board[4, 5] = 2
    scenarios.append(("mid_game", board.copy(), "Mid game - 6 moves each"))

    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[5, 1] = 1; board[5, 2] = 1; board[5, 3] = 1; board[5, 4] = 1
    board[4, 1] = 2; board[4, 2] = 2; board[4, 3] = 2
    scenarios.append(("red_wins_horizontal", board.copy(), "Red wins horizontally"))

    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[5, 0] = 2; board[4, 0] = 2; board[3, 0] = 2; board[2, 0] = 2
    board[5, 1] = 1; board[5, 2] = 1; board[5, 3] = 1; board[4, 1] = 1
    scenarios.append(("yellow_wins_vertical", board.copy(), "Yellow wins vertically"))

    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[5, 0] = 1; board[5, 1] = 2; board[4, 1] = 1
    board[5, 2] = 2; board[4, 2] = 2; board[3, 2] = 1
    board[5, 3] = 2; board[4, 3] = 1; board[3, 3] = 2; board[2, 3] = 1
    board[5, 4] = 1
    scenarios.append(("red_wins_diagonal", board.copy(), "Red wins diagonally"))

    board = np.zeros((ROWS, COLS), dtype=np.int8)
    piece = 1
    for col in range(COLS):
        for row in range(ROWS - 1, -1, -1):
            if col == 6 and row < 2:
                break
            board[row, col] = piece
            piece = 3 - piece
    scenarios.append(("nearly_full", board.copy(), "Nearly full board"))

    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[5, 3] = 1
    scenarios.append(("first_move", board.copy(), "First move - Red center"))

    return scenarios


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")
    os.makedirs(output_dir, exist_ok=True)
    scenarios = create_test_scenarios()
    print(f"Generating {len(scenarios)} test board images...\n")

    for name, board, description in scenarios:
        print(f"[{name}] {description}")
        img = board_to_image(board)
        img_path = os.path.join(output_dir, f"board_{name}.png")
        json_path = os.path.join(output_dir, f"board_{name}.json")
        cv2.imwrite(img_path, img)
        red_count = int(np.sum(board == 1))
        yellow_count = int(np.sum(board == 2))
        gt = {"name": name, "description": description, "board": board.tolist(),
              "red_count": red_count, "yellow_count": yellow_count,
              "next_player": 1 if red_count == yellow_count else 2}
        with open(json_path, 'w') as f:
            json.dump(gt, f, indent=2)
        print(f"  Saved: {img_path}")

        img_noisy = board_to_image(board, add_noise=True, angle_deg=2, brightness_shift=-15)
        noisy_path = os.path.join(output_dir, f"board_{name}_noisy.png")
        cv2.imwrite(noisy_path, img_noisy)
        noisy_json = os.path.join(output_dir, f"board_{name}_noisy.json")
        gt_noisy = {**gt, "name": f"{name}_noisy", "description": f"{description} (noisy)"}
        with open(noisy_json, 'w') as f:
            json.dump(gt_noisy, f, indent=2)

    print(f"\nDone! Generated {len(scenarios) * 2} images in {output_dir}/")


if __name__ == "__main__":
    main()
