"""
Board Detector - Extracts Connect Four board state from an image.

Uses OpenCV color segmentation to detect the blue board frame,
red pieces, yellow pieces, and empty slots.

Returns a 6x7 numpy array: 0=empty, 1=red, 2=yellow

Two built-in configs:
  DetectionConfig()        — physical board defaults
  SCREEN_CONFIG            — digital screen / phone display

Fallback detection: when the board frame isn't found via color,
the detector tries to infer the grid from piece positions directly.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class DetectionConfig:
    """HSV color thresholds. Tune these for your lighting / display conditions."""

    # Blue board frame
    board_hsv_low:  Tuple[int, int, int] = (90,  50,  40)
    board_hsv_high: Tuple[int, int, int] = (140, 255, 255)   # V→255: covers bright screens

    # Red pieces (two ranges to wrap around the 0/180 boundary)
    red_hsv_low1:  Tuple[int, int, int] = (0,   80, 80)
    red_hsv_high1: Tuple[int, int, int] = (12,  255, 255)
    red_hsv_low2:  Tuple[int, int, int] = (163, 80, 80)
    red_hsv_high2: Tuple[int, int, int] = (180, 255, 255)

    # Yellow pieces
    yellow_hsv_low:  Tuple[int, int, int] = (15, 80, 80)
    yellow_hsv_high: Tuple[int, int, int] = (40, 255, 255)

    # Detection behaviour
    min_board_area_ratio: float = 0.02   # Minimum board area as fraction of image
    piece_fallback: bool = True          # Try piece-based detection if board frame not found


# ── Presets ──────────────────────────────────────────────────────────────────

# Optimised for phone/monitor screens (emitted light, high saturation + brightness)
SCREEN_CONFIG = DetectionConfig(
    board_hsv_low=(85,  80,  80),
    board_hsv_high=(140, 255, 255),
    red_hsv_low1=(0,   120, 100),
    red_hsv_high1=(12,  255, 255),
    red_hsv_low2=(163, 120, 100),
    red_hsv_high2=(180, 255, 255),
    yellow_hsv_low=(15, 120, 100),
    yellow_hsv_high=(40, 255, 255),
    min_board_area_ratio=0.01,
    piece_fallback=True,
)


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """Result of board detection."""
    board:         np.ndarray
    confidence:    float
    grid_centers:  Optional[np.ndarray] = None
    board_contour: Optional[np.ndarray] = None
    debug_image:   Optional[np.ndarray] = None
    fallback_used: bool = False
    errors:        List[str] = field(default_factory=list)


class BoardDetector:
    """Detects Connect Four board state from images."""

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()

    def detect(self, image: np.ndarray, debug: bool = False) -> DetectionResult:
        errors: List[str] = []
        fallback_used = False
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # ── Primary: find board frame by color ───────────────────────────────
        board_mask = self._detect_board_region(hsv)
        board_contour = self._find_board_contour(board_mask, image.shape)

        # ── Fallback: infer board bounds from piece positions ─────────────────
        if board_contour is None and self.config.piece_fallback:
            board_contour = self._piece_fallback_contour(hsv, image.shape)
            if board_contour is not None:
                fallback_used = True
                errors.append("Board frame not found — using piece-fallback detection")

        if board_contour is None:
            errors.append("Board not found (no blue frame, no visible pieces)")
            return DetectionResult(
                board=np.zeros((6, 7), dtype=np.int8),
                confidence=0.0,
                errors=errors,
            )

        grid_centers = self._compute_grid_centers(board_contour)
        board        = self._classify_cells(hsv, grid_centers)
        confidence   = self._compute_confidence(board)

        # Penalise fallback detections slightly — they may be less accurate
        if fallback_used:
            confidence = min(confidence, 0.65)

        debug_img = (
            self._draw_debug(image, board, grid_centers, board_contour,
                             confidence, fallback_used)
            if debug else None
        )

        return DetectionResult(
            board=board,
            confidence=confidence,
            grid_centers=grid_centers,
            board_contour=board_contour,
            debug_image=debug_img,
            fallback_used=fallback_used,
            errors=errors,
        )

    # ── Board frame detection ─────────────────────────────────────────────────

    def _detect_board_region(self, hsv: np.ndarray) -> np.ndarray:
        cfg = self.config
        mask = cv2.inRange(hsv,
                           np.array(cfg.board_hsv_low),
                           np.array(cfg.board_hsv_high))
        # Large close kernel fills the white/empty circles punched through the board
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        return mask

    def _find_board_contour(self, mask: np.ndarray, img_shape) -> Optional[np.ndarray]:
        """
        Find the board boundary. Rather than just taking the single largest contour,
        we take the bounding rect of *all* significant blue clusters. This handles
        cases where the board's white empty circles break up the blue mask.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        img_area = img_shape[0] * img_shape[1]
        min_cluster = img_area * 0.0005   # Ignore tiny noise blobs

        # Collect all points from meaningful blue clusters
        all_pts: List = []
        for cnt in contours:
            if cv2.contourArea(cnt) > min_cluster:
                all_pts.extend(cnt.reshape(-1, 2).tolist())

        if not all_pts:
            return None

        pts = np.array(all_pts, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)

        if w * h < img_area * self.config.min_board_area_ratio:
            return None

        # Return as a rectangular contour
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                        dtype=np.int32).reshape(-1, 1, 2)

    # ── Piece-first fallback ──────────────────────────────────────────────────

    def _piece_fallback_contour(self, hsv: np.ndarray,
                                 img_shape) -> Optional[np.ndarray]:
        """
        When the board frame can't be found by color, try to locate visible pieces
        and infer the grid bounding box from their positions.
        Works even if the board color doesn't match (e.g. non-blue boards).
        """
        cfg = self.config
        h, w = img_shape[:2]

        red = (cv2.inRange(hsv, np.array(cfg.red_hsv_low1), np.array(cfg.red_hsv_high1)) |
               cv2.inRange(hsv, np.array(cfg.red_hsv_low2), np.array(cfg.red_hsv_high2)))
        yellow = cv2.inRange(hsv,
                             np.array(cfg.yellow_hsv_low),
                             np.array(cfg.yellow_hsv_high))
        mask = red | yellow

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_area = h * w
        blobs: List[Tuple[int, int, int]] = []   # (cx, cy, radius)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (img_area * 0.0005 < area < img_area * 0.04):
                continue
            perim = cv2.arcLength(cnt, True)
            if perim > 0 and 4 * np.pi * area / perim ** 2 > 0.35:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    blobs.append((
                        int(M["m10"] / M["m00"]),
                        int(M["m01"] / M["m00"]),
                        int(np.sqrt(area / np.pi)),
                    ))

        if len(blobs) < 2:
            return None

        xs = [b[0] for b in blobs]
        ys = [b[1] for b in blobs]
        med_r = float(np.median([b[2] for b in blobs]))
        cell_size = med_r * 2.6   # Typical piece-radius to cell-size ratio

        # Pad outward to cover empty cells beyond the detected pieces
        pad = int(cell_size * 1.5)
        x1 = max(0, min(xs) - pad)
        y1 = max(0, min(ys) - pad)
        x2 = min(w - 1, max(xs) + pad)
        y2 = min(h - 1, max(ys) + pad)

        if (x2 - x1) < cell_size * 2 or (y2 - y1) < cell_size * 2:
            return None

        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        dtype=np.int32).reshape(-1, 1, 2)

    # ── Grid + classification ────────────────────────────────────────────────

    def _compute_grid_centers(self, board_contour: np.ndarray) -> np.ndarray:
        x, y, w, h = cv2.boundingRect(board_contour)
        pad_x = int(w * 0.02)
        pad_y = int(h * 0.02)
        x += pad_x; y += pad_y; w -= 2 * pad_x; h -= 2 * pad_y
        cell_w, cell_h = w / 7, h / 6
        centers = np.zeros((6, 7, 2), dtype=np.int32)
        for row in range(6):
            for col in range(7):
                centers[row, col] = [
                    int(x + col * cell_w + cell_w / 2),
                    int(y + row * cell_h + cell_h / 2),
                ]
        return centers

    def _classify_cells(self, hsv: np.ndarray,
                        grid_centers: np.ndarray) -> np.ndarray:
        cfg = self.config
        board = np.zeros((6, 7), dtype=np.int8)
        cell_spacing = abs(int(grid_centers[0, 1, 0]) - int(grid_centers[0, 0, 0]))
        sample_radius = max(int(cell_spacing * 0.30), 6)   # Slightly larger than before

        red_mask = (cv2.inRange(hsv, np.array(cfg.red_hsv_low1), np.array(cfg.red_hsv_high1)) |
                    cv2.inRange(hsv, np.array(cfg.red_hsv_low2), np.array(cfg.red_hsv_high2)))
        yellow_mask = cv2.inRange(hsv,
                                  np.array(cfg.yellow_hsv_low),
                                  np.array(cfg.yellow_hsv_high))

        for row in range(6):
            for col in range(7):
                cx, cy = int(grid_centers[row, col, 0]), int(grid_centers[row, col, 1])
                roi = np.zeros(hsv.shape[:2], dtype=np.uint8)
                cv2.circle(roi, (cx, cy), sample_radius, 255, -1)
                total   = cv2.countNonZero(roi)
                red_r   = cv2.countNonZero(red_mask   & roi) / max(total, 1)
                yel_r   = cv2.countNonZero(yellow_mask & roi) / max(total, 1)
                if red_r > 0.12 and red_r > yel_r:
                    board[row, col] = 1
                elif yel_r > 0.12 and yel_r > red_r:
                    board[row, col] = 2
        return board

    def _compute_confidence(self, board: np.ndarray) -> float:
        confidence = 1.0
        for col in range(7):
            found_empty = False
            for row in range(6):
                if board[row, col] == 0:
                    found_empty = True
                elif found_empty:
                    confidence -= 0.15   # Floating piece — gravity violation
        red_count    = int(np.sum(board == 1))
        yellow_count = int(np.sum(board == 2))
        if abs(red_count - yellow_count) > 1:
            confidence -= 0.2
        return max(0.0, min(1.0, confidence))

    # ── Debug overlay ────────────────────────────────────────────────────────

    def _draw_debug(self, image: np.ndarray, board: np.ndarray,
                    grid_centers: np.ndarray, board_contour: np.ndarray,
                    confidence: float, fallback_used: bool) -> np.ndarray:
        debug = image.copy()
        color = (0, 165, 255) if fallback_used else (0, 255, 0)
        cv2.drawContours(debug, [board_contour], -1, color, 2)

        for row in range(6):
            for col in range(7):
                cx, cy = int(grid_centers[row, col, 0]), int(grid_centers[row, col, 1])
                cell = board[row, col]
                c = (200, 200, 200) if cell == 0 else ((0, 0, 255) if cell == 1 else (0, 255, 255))
                label = "." if cell == 0 else ("R" if cell == 1 else "Y")
                cv2.circle(debug, (cx, cy), 5, c, -1)
                cv2.putText(debug, label, (cx - 6, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

        status = f"{'FALLBACK ' if fallback_used else ''}Confidence: {confidence:.2f}"
        cv2.putText(debug, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if confidence > 0.8 else (0, 0, 255), 2)
        return debug


def board_to_string(board: np.ndarray) -> str:
    """Pretty-print a board state."""
    symbols = {0: ".", 1: "R", 2: "Y"}
    lines = ["  " + " ".join(str(i) for i in range(7)), "  " + "-" * 13]
    for row in range(6):
        lines.append(f"{row}|" + " ".join(symbols[board[row, col]] for col in range(7)) + "|")
    lines.append("  " + "-" * 13)
    return "\n".join(lines)
