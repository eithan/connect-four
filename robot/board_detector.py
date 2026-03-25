"""
Board Detector - Extracts Connect Four board state from an image.

Uses OpenCV color segmentation to detect the blue board frame,
red pieces, yellow pieces, and empty slots.

Returns a 6x7 numpy array: 0=empty, 1=red, 2=yellow
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class DetectionConfig:
    """HSV color thresholds. Tune these for your lighting conditions."""
    board_hsv_low: Tuple[int, int, int] = (95, 50, 40)
    board_hsv_high: Tuple[int, int, int] = (130, 255, 200)
    red_hsv_low1: Tuple[int, int, int] = (0, 80, 80)
    red_hsv_high1: Tuple[int, int, int] = (10, 255, 255)
    red_hsv_low2: Tuple[int, int, int] = (165, 80, 80)
    red_hsv_high2: Tuple[int, int, int] = (180, 255, 255)
    yellow_hsv_low: Tuple[int, int, int] = (18, 80, 80)
    yellow_hsv_high: Tuple[int, int, int] = (38, 255, 255)
    min_board_area_ratio: float = 0.05


@dataclass
class DetectionResult:
    """Result of board detection."""
    board: np.ndarray
    confidence: float
    grid_centers: Optional[np.ndarray] = None
    board_contour: Optional[np.ndarray] = None
    debug_image: Optional[np.ndarray] = None
    errors: List[str] = field(default_factory=list)


class BoardDetector:
    """Detects Connect Four board state from images."""

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()

    def detect(self, image: np.ndarray, debug: bool = False) -> DetectionResult:
        errors = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        board_mask = self._detect_board_region(hsv)
        board_contour = self._find_board_contour(board_mask, image.shape)

        if board_contour is None:
            errors.append("Could not find board region")
            return DetectionResult(board=np.zeros((6, 7), dtype=np.int8),
                                   confidence=0.0, errors=errors)

        grid_centers = self._compute_grid_centers(board_contour)
        board = self._classify_cells(hsv, grid_centers)
        confidence = self._compute_confidence(board)
        debug_img = self._draw_debug(image, board, grid_centers, board_contour, confidence) if debug else None

        return DetectionResult(board=board, confidence=confidence, grid_centers=grid_centers,
                               board_contour=board_contour, debug_image=debug_img, errors=errors)

    def _detect_board_region(self, hsv):
        cfg = self.config
        mask = cv2.inRange(hsv, np.array(cfg.board_hsv_low), np.array(cfg.board_hsv_high))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def _find_board_contour(self, mask, img_shape):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < img_shape[0] * img_shape[1] * self.config.min_board_area_ratio:
            return None
        return largest

    def _compute_grid_centers(self, board_contour):
        x, y, w, h = cv2.boundingRect(board_contour)
        pad_x, pad_y = int(w * 0.02), int(h * 0.02)
        x += pad_x; y += pad_y; w -= 2 * pad_x; h -= 2 * pad_y
        cell_w, cell_h = w / 7, h / 6
        centers = np.zeros((6, 7, 2), dtype=np.int32)
        for row in range(6):
            for col in range(7):
                centers[row, col] = [int(x + col * cell_w + cell_w / 2),
                                     int(y + row * cell_h + cell_h / 2)]
        return centers

    def _classify_cells(self, hsv, grid_centers):
        cfg = self.config
        board = np.zeros((6, 7), dtype=np.int8)
        cell_spacing = abs(grid_centers[0, 1, 0] - grid_centers[0, 0, 0])
        sample_radius = max(int(cell_spacing * 0.25), 5)

        red_mask = (cv2.inRange(hsv, np.array(cfg.red_hsv_low1), np.array(cfg.red_hsv_high1)) |
                    cv2.inRange(hsv, np.array(cfg.red_hsv_low2), np.array(cfg.red_hsv_high2)))
        yellow_mask = cv2.inRange(hsv, np.array(cfg.yellow_hsv_low), np.array(cfg.yellow_hsv_high))

        for row in range(6):
            for col in range(7):
                cx, cy = grid_centers[row, col]
                roi = np.zeros(hsv.shape[:2], dtype=np.uint8)
                cv2.circle(roi, (cx, cy), sample_radius, 255, -1)
                total = cv2.countNonZero(roi)
                red_r = cv2.countNonZero(red_mask & roi) / max(total, 1)
                yel_r = cv2.countNonZero(yellow_mask & roi) / max(total, 1)
                if red_r > 0.15 and red_r > yel_r:
                    board[row, col] = 1
                elif yel_r > 0.15 and yel_r > red_r:
                    board[row, col] = 2
        return board

    def _compute_confidence(self, board):
        confidence = 1.0
        for col in range(7):
            found_empty = False
            for row in range(6):
                if board[row, col] == 0:
                    found_empty = True
                elif found_empty:
                    confidence -= 0.15
        red_count = int(np.sum(board == 1))
        yellow_count = int(np.sum(board == 2))
        if abs(red_count - yellow_count) > 1:
            confidence -= 0.2
        return max(0.0, min(1.0, confidence))

    def _draw_debug(self, image, board, grid_centers, board_contour, confidence):
        debug = image.copy()
        cv2.drawContours(debug, [board_contour], -1, (0, 255, 0), 2)
        for row in range(6):
            for col in range(7):
                cx, cy = grid_centers[row, col]
                cell = board[row, col]
                color = (200, 200, 200) if cell == 0 else ((0, 0, 255) if cell == 1 else (0, 255, 255))
                label = "." if cell == 0 else ("R" if cell == 1 else "Y")
                cv2.circle(debug, (cx, cy), 5, color, -1)
                cv2.putText(debug, label, (cx - 6, cy + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(debug, f"Confidence: {confidence:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0, 255, 0) if confidence > 0.8 else (0, 0, 255), 2)
        return debug


def board_to_string(board):
    """Pretty-print a board state."""
    symbols = {0: ".", 1: "R", 2: "Y"}
    lines = ["  " + " ".join(str(i) for i in range(7)), "  " + "-" * 13]
    for row in range(6):
        lines.append(f"{row}|" + " ".join(symbols[board[row, col]] for col in range(7)) + "|")
    lines.append("  " + "-" * 13)
    return "\n".join(lines)
