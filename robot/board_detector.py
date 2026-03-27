"""
Board Detector - Extracts Connect Four board state from an image.

Uses OpenCV color segmentation to detect the blue board frame,
red pieces, yellow pieces, and empty slots.

Returns a 6x7 numpy array: 0=empty, 1=red, 2=yellow

Three built-in configs:
  DetectionConfig()        — physical board defaults (real plastic board)
  PHYSICAL_CONFIG          — alias for DetectionConfig() with explicit name
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

    # Blue board frame — physical plastic board under typical indoor lighting.
    # Increase S_low if background blue objects bleed in; decrease if board looks dark.
    board_hsv_low:  Tuple[int, int, int] = (90,  80,  50)
    board_hsv_high: Tuple[int, int, int] = (140, 255, 255)

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

# Explicit alias for the physical-board defaults (same as DetectionConfig()).
# Use this when you want to be unambiguous in code, e.g. config=PHYSICAL_CONFIG.
PHYSICAL_CONFIG = DetectionConfig()

# Optimised for phone/monitor screens (emitted light, high saturation + brightness).
# Do NOT use this for a real plastic board — use PHYSICAL_CONFIG or DetectionConfig().
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

        grid_centers = self._compute_grid_centers(board_contour, image)
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
        Find the board boundary from the blue-frame mask.

        Strategy: pick the LARGEST single blue blob whose bounding-rect aspect
        ratio is plausible for a Connect Four board (~7:6 ≈ 1.17, allowing
        perspective tilt). We do NOT merge all blue blobs — doing so causes any
        unrelated blue object in the scene (furniture, clothing, walls) to
        inflate the bounding box wildly.

        The large MORPH_CLOSE in _detect_board_region fills the white hole-circles
        so the board usually appears as one solid blue blob. If multiple blobs
        remain, we take the largest one that looks board-shaped.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        img_area = img_shape[0] * img_shape[1]
        noise_threshold = img_area * 0.001   # 0.1% — discard tiny noise blobs

        # Sort candidates largest-first, drop noise
        candidates = sorted(
            [c for c in contours if cv2.contourArea(c) > noise_threshold],
            key=cv2.contourArea,
            reverse=True,
        )
        if not candidates:
            return None

        # Connect Four board is 7 cols × 6 rows ≈ 1.17:1.
        # Allow 0.6–2.8 to tolerate tilt/perspective.
        ASPECT_MIN, ASPECT_MAX = 0.6, 2.8

        for cnt in candidates:
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue
            if ASPECT_MIN <= w / h <= ASPECT_MAX:
                if w * h >= img_area * self.config.min_board_area_ratio:
                    return np.array(
                        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                        dtype=np.int32,
                    ).reshape(-1, 1, 2)

        # Nothing passed the aspect check — fall back to the largest blob.
        x, y, w, h = cv2.boundingRect(candidates[0])
        if w * h >= img_area * self.config.min_board_area_ratio:
            return np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
        return None

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

    def _compute_grid_centers(self, board_contour: np.ndarray,
                               image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the 6×7 grid cell centres.

        Primary path: detect the dark hole-circles within the board region
        (HoughCircles) and fit a grid to their actual positions. This is immune
        to the bounding-box being taller/wider than just the hole area (due to
        the board's guide rail, legs, etc.).

        Fallback: pad the bounding rect with a generous margin that accounts for
        the typical Connect Four board structure (15% top guide, 7% bottom base,
        7% side borders).
        """
        bx, by, bw, bh = cv2.boundingRect(board_contour)

        if image is not None:
            centers = self._grid_from_holes(image, bx, by, bw, bh)
            if centers is not None:
                return centers

        # Fallback: padding-based estimation.
        # Real boards: holes fill ~85% of width, ~68% of height;
        # the top guide rail adds ~15% and the base/legs add ~7%.
        pad_x     = int(bw * 0.07)
        pad_y_top = int(bh * 0.15)
        pad_y_bot = int(bh * 0.07)
        x = bx + pad_x
        y = by + pad_y_top
        w = bw - 2 * pad_x
        h = bh - pad_y_top - pad_y_bot
        cell_w, cell_h = w / 7, h / 6
        centers = np.zeros((6, 7, 2), dtype=np.int32)
        for row in range(6):
            for col in range(7):
                centers[row, col] = [
                    int(x + col * cell_w + cell_w / 2),
                    int(y + row * cell_h + cell_h / 2),
                ]
        return centers

    def _grid_from_holes(self, image: np.ndarray,
                          bx: int, by: int, bw: int, bh: int) -> Optional[np.ndarray]:
        """
        Find dark hole-circles inside the board region using HoughCircles,
        then cluster them into a 6×7 grid.

        Returns None if too few circles are found or clustering fails.
        """
        buf = max(10, int(min(bw, bh) * 0.03))
        x1 = max(0, bx - buf);  y1 = max(0, by - buf)
        x2 = min(image.shape[1], bx + bw + buf)
        y2 = min(image.shape[0], by + bh + buf)
        roi = image[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.5)

        # Expected size: cells are ~board_width/7; holes are ~30-42% of cell_size
        cell_est  = bw / 7.0
        min_r     = max(int(cell_est * 0.22), 8)
        max_r     = int(cell_est * 0.46)
        min_dist  = int(cell_est * 0.65)

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=min_dist,
            param1=60, param2=22,
            minRadius=min_r, maxRadius=max_r,
        )
        if circles is None or len(circles[0]) < 12:
            return None

        pts = np.round(circles[0]).astype(int)
        cx = (pts[:, 0] + x1).astype(float)
        cy = (pts[:, 1] + y1).astype(float)

        return self._fit_6x7_grid(cx, cy, cell_est)

    def _fit_6x7_grid(self, xs: np.ndarray, ys: np.ndarray,
                       cell_est: float) -> Optional[np.ndarray]:
        """
        Cluster hole centres into 6 rows × 7 columns and return a (6,7,2) grid.
        """
        row_means = self._cluster_positions(ys, 6, cell_est)
        col_means = self._cluster_positions(xs, 7, cell_est)
        if row_means is None or col_means is None:
            return None
        centers = np.zeros((6, 7, 2), dtype=np.int32)
        for r, y in enumerate(row_means):
            for c, x in enumerate(col_means):
                centers[r, c] = [int(x), int(y)]
        return centers

    @staticmethod
    def _cluster_positions(values: np.ndarray, n_target: int,
                            cell_size: float) -> Optional[List[float]]:
        """
        Gap-cluster 1D hole positions into n_target groups.
        Consecutive sorted values separated by > 50% of cell_size start a new group.
        Missing groups are extrapolated from the median spacing.
        Returns sorted group means, or None if grouping is incoherent.
        """
        if len(values) < max(n_target - 2, 3):
            return None

        sv = np.sort(values)
        groups: List[List[float]] = [[float(sv[0])]]
        for v in sv[1:]:
            if v - groups[-1][-1] > cell_size * 0.5:
                groups.append([])
            groups[-1].append(float(v))

        n_found = len(groups)
        if n_found > n_target + 1 or n_found < n_target - 2:
            return None   # Too noisy or too many holes missing

        means: List[float] = sorted(float(np.mean(g)) for g in groups)

        # Extrapolate missing positions
        if len(means) < n_target:
            spacing = (
                float(np.median([means[i + 1] - means[i]
                                  for i in range(len(means) - 1)]))
                if len(means) > 1 else cell_size
            )
            while len(means) < n_target:
                expected_span = n_target * spacing
                current_span  = means[-1] - means[0]
                if current_span + spacing <= expected_span:
                    means.append(means[-1] + spacing)
                else:
                    means.insert(0, means[0] - spacing)
            means = sorted(means)[:n_target]

        return means

    def _classify_cells(self, hsv: np.ndarray,
                        grid_centers: np.ndarray) -> np.ndarray:
        cfg = self.config
        board = np.zeros((6, 7), dtype=np.int8)
        cell_spacing = abs(int(grid_centers[0, 1, 0]) - int(grid_centers[0, 0, 0]))
        # 0.25 keeps sampling well inside each cell, avoiding bleed from neighbours
        sample_radius = max(int(cell_spacing * 0.25), 5)

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
                if red_r > 0.18 and red_r > yel_r:
                    board[row, col] = 1
                elif yel_r > 0.18 and yel_r > red_r:
                    board[row, col] = 2

        return self._apply_gravity_filter(board)

    @staticmethod
    def _apply_gravity_filter(board: np.ndarray) -> np.ndarray:
        """
        Remove physically impossible (floating) pieces.
        In a real game, pieces stack from the bottom of each column.
        Any piece above an empty slot is a misdetection — strip it out.
        """
        filtered = board.copy()
        for col in range(7):
            found_empty = False
            for row in range(5, -1, -1):   # scan bottom → top
                if filtered[row, col] == 0:
                    found_empty = True
                elif found_empty:
                    filtered[row, col] = 0  # floating piece — remove
        return filtered

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


class LockedBoardDetector:
    """
    Wraps BoardDetector with grid locking and pre-lock smoothing.

    Problem: as the phone/camera moves, the board bounding box shifts each
    frame, causing grid centers to drift and misdetections to accumulate.
    Background motion (fans, shadows, lighting) can also intermittently break
    detection for a single frame, preventing the streak from ever reaching the
    lock threshold.

    Solution:
    - Smooth the bounding box over a small rolling window of recent detections
      so the grid centers don't jump around frame-to-frame.
    - Allow up to SKIP_TOLERANCE bad frames within the lock streak without
      resetting it — a single shadow flicker won't kill the streak.
    - Once locked, grid centers are fixed; only piece classification runs.

    Auto-unlock: classification confidence stays below 0.2 for UNLOCK_FRAMES
    consecutive frames → the board has probably left the view.
    Press 'l' in the UI to force an immediate re-lock.
    """

    LOCK_MIN_CONF  = 0.45   # Confidence needed for a frame to count as "good"
    LOCK_FRAMES    = 4      # Good frames (within window) required to lock
    UNLOCK_FRAMES  = 20     # Bad frames before auto-unlock (~10 s @ 2 fps)
    SKIP_TOLERANCE = 1      # Bad frames tolerated inside the lock streak
    SMOOTH_WINDOW  = 4      # Bounding-box frames to average (reduces jitter)

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.detector = BoardDetector(config)
        self._locked_centers: Optional[np.ndarray] = None
        self._locked_contour: Optional[np.ndarray] = None
        self._good_streak  = 0
        self._bad_in_streak = 0   # Consecutive bad frames inside current streak
        self._bad_streak   = 0    # Consecutive bad frames while locked (for unlock)
        self.is_locked     = False
        # Rolling buffer of recent bounding rects (x, y, w, h) for smoothing
        self._smooth_buf: List[Tuple[int, int, int, int]] = []

    @property
    def config(self) -> DetectionConfig:
        return self.detector.config

    @config.setter
    def config(self, value: DetectionConfig):
        self.detector.config = value
        self.unlock()   # Reset lock when config changes (e.g. tuning mode)

    def unlock(self):
        """Force the detector to re-find the board frame on the next frame."""
        self._locked_centers = None
        self._locked_contour = None
        self._good_streak   = 0
        self._bad_in_streak = 0
        self._bad_streak    = 0
        self.is_locked      = False
        self._smooth_buf.clear()

    def detect(self, image: np.ndarray, debug: bool = False) -> DetectionResult:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if self.is_locked:
            return self._detect_locked(image, hsv, debug)
        return self._detect_full(image, hsv, debug)

    # ── Locked fast-path ─────────────────────────────────────────────────────

    def _detect_locked(self, image: np.ndarray, hsv: np.ndarray,
                       debug: bool) -> DetectionResult:
        board      = self.detector._classify_cells(hsv, self._locked_centers)
        confidence = self.detector._compute_confidence(board)

        if confidence < 0.2:
            self._bad_streak += 1
            if self._bad_streak >= self.UNLOCK_FRAMES:
                print("[Board] Lock released — board lost from view")
                self.unlock()
        else:
            self._bad_streak = 0

        debug_img = None
        if debug:
            debug_img = self.detector._draw_debug(
                image, board, self._locked_centers,
                self._locked_contour, confidence, False)
            cv2.drawContours(debug_img, [self._locked_contour], -1, (255, 220, 0), 3)
            cv2.putText(debug_img, "LOCKED", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)

        return DetectionResult(
            board=board,
            confidence=confidence,
            grid_centers=self._locked_centers,
            board_contour=self._locked_contour,
            debug_image=debug_img,
        )

    # ── Full detection (pre-lock) ─────────────────────────────────────────────

    def _smooth_contour(self, contour: np.ndarray) -> np.ndarray:
        """
        Average the bounding rect over the last SMOOTH_WINDOW good detections.
        This prevents the grid from jumping around due to per-frame noise.
        """
        x, y, w, h = cv2.boundingRect(contour)
        self._smooth_buf.append((x, y, w, h))
        if len(self._smooth_buf) > self.SMOOTH_WINDOW:
            self._smooth_buf.pop(0)
        ax = int(np.mean([b[0] for b in self._smooth_buf]))
        ay = int(np.mean([b[1] for b in self._smooth_buf]))
        aw = int(np.mean([b[2] for b in self._smooth_buf]))
        ah = int(np.mean([b[3] for b in self._smooth_buf]))
        return np.array(
            [[ax, ay], [ax + aw, ay], [ax + aw, ay + ah], [ax, ay + ah]],
            dtype=np.int32,
        ).reshape(-1, 1, 2)

    def _detect_full(self, image: np.ndarray, hsv: np.ndarray,
                     debug: bool) -> DetectionResult:
        result = self.detector.detect(image, debug=debug)

        is_good = (result.board_contour is not None
                   and result.confidence >= self.LOCK_MIN_CONF)

        if is_good:
            self._bad_in_streak = 0
            # Smooth the bounding box and recompute grid/board from it so the
            # display is stable even before the lock is achieved.
            smoothed = self._smooth_contour(result.board_contour)
            result.grid_centers = self.detector._compute_grid_centers(smoothed, image)
            result.board = self.detector._classify_cells(hsv, result.grid_centers)
            result.board_contour = smoothed
            result.confidence = self.detector._compute_confidence(result.board)

            self._good_streak += 1
            if self._good_streak >= self.LOCK_FRAMES:
                self._locked_centers = result.grid_centers.copy()
                self._locked_contour = result.board_contour.copy()
                self.is_locked = True
                self._good_streak   = 0
                self._bad_in_streak = 0
                self._smooth_buf.clear()
                print("[Board] Locked ✓")
        else:
            self._bad_in_streak += 1
            if self._bad_in_streak > self.SKIP_TOLERANCE:
                # Too many consecutive bad frames — abandon this streak
                if self._good_streak > 0:
                    print(f"[Board] Streak reset after {self._good_streak} good frames "
                          f"({self._bad_in_streak} bad in a row)")
                self._good_streak   = 0
                self._bad_in_streak = 0
                self._smooth_buf.clear()
            # else: tolerate this bad frame; streak lives on

        return result


def board_to_string(board: np.ndarray) -> str:
    """Pretty-print a board state."""
    symbols = {0: ".", 1: "R", 2: "Y"}
    lines = ["  " + " ".join(str(i) for i in range(7)), "  " + "-" * 13]
    for row in range(6):
        lines.append(f"{row}|" + " ".join(symbols[board[row, col]] for col in range(7)) + "|")
    lines.append("  " + "-" * 13)
    return "\n".join(lines)
