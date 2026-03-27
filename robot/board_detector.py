"""
Board Detector — Connect Four
==============================

Pipeline
--------
1. Locate the blue board frame (HSV mask → largest well-shaped blue blob).
2. Within the board, find hole circles by detecting non-blue circular regions
   (contour analysis — more reliable than HoughCircles).
3. Cluster hole centres into a 6×7 grid (gap-based 1D clustering with
   bound-constrained extrapolation for partially-visible holes).
4. Classify each cell as red / yellow-green / empty by sampling HSV at
   the hole centre.

Key design notes
----------------
- "Yellow" range covers H 15–85 to capture both standard yellow AND the
  lime-green / chartreuse pieces common on cheaper boards.
- Empty holes show through to background (gray, not white) — we classify
  by positive evidence of red or yellow-green, not by absence of background.
- Contour-based hole detection works inside a blue mask inversion; no
  HoughCircles parameter tuning required.
- LockedBoardDetector wraps BoardDetector with two-stage locking:
    Stage 1 (Candidate) — first good detection shown immediately.
    Stage 2 (Locked)    — confirmed after LOCK_FRAMES good detections.
  Bad frames hold the last candidate; only 15+ consecutive bad frames
  (board truly gone) clear the state.

External API (unchanged from previous versions)
-----------------------------------------------
  DetectionConfig, DetectionResult, BoardDetector, LockedBoardDetector,
  PHYSICAL_CONFIG, SCREEN_CONFIG, board_to_string
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionConfig:
    """HSV colour thresholds and detection parameters."""

    # Blue board frame
    board_hsv_low:  Tuple[int, int, int] = (90,  80, 50)
    board_hsv_high: Tuple[int, int, int] = (140, 255, 255)

    # Red pieces (two ranges to wrap the 0/180 hue boundary)
    red_hsv_low1:  Tuple[int, int, int] = (0,   80, 80)
    red_hsv_high1: Tuple[int, int, int] = (12,  255, 255)
    red_hsv_low2:  Tuple[int, int, int] = (163, 80, 80)
    red_hsv_high2: Tuple[int, int, int] = (180, 255, 255)

    # Yellow / lime-green pieces
    # H 15–85 covers standard yellow (H≈20-35) AND lime/chartreuse (H≈40-75)
    # S≥100 prevents background colours seen through empty holes from matching
    yellow_hsv_low:  Tuple[int, int, int] = (15, 100, 80)
    yellow_hsv_high: Tuple[int, int, int] = (85, 255, 255)

    # Minimum board area as fraction of image
    min_board_area_ratio: float = 0.02

    # Fraction of a cell that must be piece-colour to count as a piece
    piece_threshold: float = 0.18

    # Hole circularity minimum (0–1; 1 = perfect circle)
    min_circularity: float = 0.40


# Physical board preset (default)
PHYSICAL_CONFIG = DetectionConfig()

# Screen/phone display preset (emitted light — higher saturation)
SCREEN_CONFIG = DetectionConfig(
    board_hsv_low=(85,  80, 80),
    board_hsv_high=(140, 255, 255),
    red_hsv_low1=(0,   120, 100),
    red_hsv_high1=(12,  255, 255),
    red_hsv_low2=(163, 120, 100),
    red_hsv_high2=(180, 255, 255),
    yellow_hsv_low=(15, 120, 100),
    yellow_hsv_high=(85, 255, 255),
    min_board_area_ratio=0.01,
)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    board:          np.ndarray                    # (6, 7) int8: 0=empty 1=red 2=yellow
    confidence:     float                         # 0.0 – 1.0
    grid_centers:   Optional[np.ndarray] = None   # (6, 7, 2) int32
    board_contour:  Optional[np.ndarray] = None   # for display
    debug_image:    Optional[np.ndarray] = None
    fallback_used:  bool = False
    errors:         List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Core detector
# ─────────────────────────────────────────────────────────────────────────────

class BoardDetector:
    """Single-frame Connect Four board state extractor."""

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()

    # ── Public entry point ────────────────────────────────────────────────────

    def detect(self, image: np.ndarray, debug: bool = False) -> DetectionResult:
        errors: List[str] = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # ── 1. Locate board ───────────────────────────────────────────────────
        bbox = self._find_board_bbox(hsv, image.shape)
        if bbox is None:
            return DetectionResult(
                board=np.zeros((6, 7), dtype=np.int8),
                confidence=0.0,
                errors=["Board not found — no blue frame detected"],
            )
        bx, by, bw, bh = bbox

        # ── 2. Detect holes ───────────────────────────────────────────────────
        holes = self._find_holes(hsv, bx, by, bw, bh)

        # ── 3. Fit 6×7 grid ───────────────────────────────────────────────────
        cell_est = bw / 7.0
        fallback_used = False
        grid_centers = None

        if len(holes) >= 8:
            grid_centers = self._fit_grid(holes, cell_est, bx, bx + bw, by, by + bh)

        if grid_centers is None:
            grid_centers = self._grid_from_bounds(bx, by, bw, bh)
            fallback_used = True
            errors.append(f"Hole detection found {len(holes)} circles — using padded-bounds grid")

        # ── 4. Classify cells ─────────────────────────────────────────────────
        # Compute confidence on the raw (unfiltered) board — floating pieces
        # in the raw detection mean the grid is misaligned, which is the
        # signal we want.  Apply the gravity filter afterwards for the result.
        board_raw  = self._classify_cells(hsv, grid_centers)
        confidence = self._compute_confidence(board_raw)
        if fallback_used:
            confidence = min(confidence, 0.65)
        board = self._apply_gravity_filter(board_raw)

        board_contour = np.array(
            [[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]],
            dtype=np.int32,
        ).reshape(-1, 1, 2)

        debug_img = None
        if debug:
            debug_img = self._draw_debug(
                image, board, grid_centers, board_contour, holes,
                confidence, fallback_used,
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

    # ── Step 1: Board bounding box ────────────────────────────────────────────

    def _find_board_bbox(self, hsv: np.ndarray,
                          img_shape: Tuple) -> Optional[Tuple[int, int, int, int]]:
        """
        Find the blue board frame and return (x, y, w, h).
        Uses a large MORPH_CLOSE to fill the hole grid, then picks the
        largest blob with a plausible Connect Four aspect ratio (0.6–2.8).
        """
        cfg = self.config
        mask = cv2.inRange(hsv, np.array(cfg.board_hsv_low), np.array(cfg.board_hsv_high))

        # Fill the circular holes so the board reads as a solid blue rectangle
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        img_area     = img_shape[0] * img_shape[1]
        noise_thresh = img_area * 0.001

        # Sort by area largest-first, ignore noise
        candidates = sorted(
            [c for c in contours if cv2.contourArea(c) > noise_thresh],
            key=cv2.contourArea, reverse=True,
        )
        if not candidates:
            return None

        # Prefer the largest blob with a board-like aspect ratio
        for cnt in candidates:
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue
            aspect = w / h
            if 0.6 <= aspect <= 2.8 and w * h >= img_area * cfg.min_board_area_ratio:
                return x, y, w, h

        # Fallback: just the largest blob
        x, y, w, h = cv2.boundingRect(candidates[0])
        if w * h >= img_area * cfg.min_board_area_ratio:
            return x, y, w, h
        return None

    # ── Step 2: Hole detection ────────────────────────────────────────────────

    def _find_holes(self, hsv: np.ndarray,
                     bx: int, by: int, bw: int, bh: int) -> List[Tuple[float, float]]:
        """
        Find hole circles within the board region.

        The board frame is blue.  Holes (empty or containing a piece) appear as
        non-blue circular regions.  We detect them by:
          a) Creating a RAW blue mask (no morphology) inside the board ROI so
             the holes aren't filled in.
          b) Inverting → non-blue mask.
          c) Finding contours, filtering by area (expected hole size) and
             circularity (≥ min_circularity).

        Both empty holes (showing gray background) and filled holes (red/yellow
        pieces) are captured — piece colour is determined in Step 4.
        """
        cfg = self.config
        hsv_roi  = hsv[by:by + bh, bx:bx + bw]

        # Raw blue mask — NOT morphed so hole gaps remain visible
        raw_blue = cv2.inRange(
            hsv_roi, np.array(cfg.board_hsv_low), np.array(cfg.board_hsv_high)
        )

        # Invert: non-blue = holes/pieces
        non_blue = cv2.bitwise_not(raw_blue)

        # Small MORPH_OPEN to remove single-pixel noise without closing holes
        k_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        non_blue = cv2.morphologyEx(non_blue, cv2.MORPH_OPEN, k_noise)

        # Expected hole area based on board width / 7 columns
        cell_est  = bw / 7.0
        min_area  = np.pi * (cell_est * 0.18) ** 2
        max_area  = np.pi * (cell_est * 0.52) ** 2

        contours, _ = cv2.findContours(non_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        holes: List[Tuple[float, float]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area < area < max_area):
                continue
            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue
            circularity = 4 * np.pi * area / (perim ** 2)
            if circularity < cfg.min_circularity:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"] + bx
            cy = M["m01"] / M["m00"] + by
            holes.append((float(cx), float(cy)))

        return holes

    # ── Step 3: Grid fitting ──────────────────────────────────────────────────

    def _fit_grid(self, holes: List[Tuple[float, float]],
                   cell_est: float,
                   min_x: float, max_x: float,
                   min_y: float, max_y: float) -> Optional[np.ndarray]:
        """Cluster hole centres into 6 rows × 7 columns."""
        xs = np.array([h[0] for h in holes])
        ys = np.array([h[1] for h in holes])

        col_means = self._cluster_1d(xs, 7, cell_est, min_x, max_x)
        row_means = self._cluster_1d(ys, 6, cell_est, min_y, max_y)

        if col_means is None or row_means is None:
            return None
        if len(col_means) != 7 or len(row_means) != 6:
            return None

        centers = np.zeros((6, 7, 2), dtype=np.int32)
        for r, y in enumerate(row_means):
            for c, x in enumerate(col_means):
                centers[r, c] = [int(x), int(y)]
        return centers

    @staticmethod
    def _cluster_1d(values: np.ndarray, n_target: int,
                     cell_size: float,
                     lo: Optional[float] = None,
                     hi: Optional[float] = None) -> Optional[List[float]]:
        """
        Gap-based 1D clustering into n_target groups.

        Consecutive sorted values separated by > 50% of cell_size start a new
        group.  Missing groups are extrapolated from median spacing, clamped to
        [lo, hi] so the grid never extends outside the board boundary.
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
            return None

        means: List[float] = sorted(float(np.mean(g)) for g in groups)

        # Merge if one extra group (noise split)
        while len(means) > n_target:
            diffs = [means[i + 1] - means[i] for i in range(len(means) - 1)]
            idx   = int(np.argmin(diffs))
            means = means[:idx] + [(means[idx] + means[idx + 1]) / 2] + means[idx + 2:]

        # Extrapolate missing positions, constrained to board bounds
        if len(means) < n_target:
            spacing = (
                float(np.median([means[i + 1] - means[i]
                                  for i in range(len(means) - 1)]))
                if len(means) > 1 else cell_size
            )
            while len(means) < n_target:
                right = means[-1] + spacing
                left  = means[0]  - spacing
                go_right = (hi is None or right <= hi + cell_size * 0.4)
                go_left  = (lo is None or left  >= lo - cell_size * 0.4)
                span = means[-1] - means[0]
                if span + spacing <= n_target * spacing and go_right:
                    means.append(right)
                elif go_left:
                    means.insert(0, left)
                elif go_right:
                    means.append(right)
                else:
                    break
            means = sorted(means)[:n_target]

        if len(means) != n_target:
            return None
        return means

    def _grid_from_bounds(self, bx: int, by: int,
                           bw: int, bh: int) -> np.ndarray:
        """
        Fallback grid when hole detection fails.
        Pads the bounding box to account for the board's guide rail (top),
        base legs (bottom), and side borders.
        """
        pad_x     = int(bw * 0.07)
        pad_y_top = int(bh * 0.15)
        pad_y_bot = int(bh * 0.07)
        x = bx + pad_x
        y = by + pad_y_top
        w = bw - 2 * pad_x
        h = bh - pad_y_top - pad_y_bot
        cw, ch = w / 7, h / 6
        centers = np.zeros((6, 7, 2), dtype=np.int32)
        for r in range(6):
            for c in range(7):
                centers[r, c] = [
                    int(x + c * cw + cw / 2),
                    int(y + r * ch + ch / 2),
                ]
        return centers

    # ── Step 4: Cell classification ───────────────────────────────────────────

    def _classify_cells(self, hsv: np.ndarray,
                         grid_centers: np.ndarray) -> np.ndarray:
        """
        For each grid cell, sample a circle around the centre and measure the
        fraction of red and yellow-green pixels.  The dominant colour wins if
        it exceeds piece_threshold.
        """
        cfg = self.config
        board = np.zeros((6, 7), dtype=np.int8)

        cell_spacing  = abs(int(grid_centers[0, 1, 0]) - int(grid_centers[0, 0, 0]))
        sample_radius = max(int(cell_spacing * 0.28), 6)

        h_img, w_img = hsv.shape[:2]

        red_mask = (
            cv2.inRange(hsv, np.array(cfg.red_hsv_low1),  np.array(cfg.red_hsv_high1)) |
            cv2.inRange(hsv, np.array(cfg.red_hsv_low2),  np.array(cfg.red_hsv_high2))
        )
        yellow_mask = cv2.inRange(
            hsv, np.array(cfg.yellow_hsv_low), np.array(cfg.yellow_hsv_high)
        )

        for r in range(6):
            for c in range(7):
                cx = int(grid_centers[r, c, 0])
                cy = int(grid_centers[r, c, 1])
                if not (0 <= cx < w_img and 0 <= cy < h_img):
                    continue

                roi_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                cv2.circle(roi_mask, (cx, cy), sample_radius, 255, -1)
                total = cv2.countNonZero(roi_mask)
                if total == 0:
                    continue

                red_r = cv2.countNonZero(red_mask    & roi_mask) / total
                yel_r = cv2.countNonZero(yellow_mask & roi_mask) / total

                if red_r > cfg.piece_threshold and red_r > yel_r:
                    board[r, c] = 1
                elif yel_r > cfg.piece_threshold and yel_r > red_r:
                    board[r, c] = 2

        return board   # raw, un-filtered — caller applies gravity filter

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_gravity_filter(board: np.ndarray) -> np.ndarray:
        """Remove physically impossible floating pieces."""
        b = board.copy()
        for col in range(7):
            found_empty = False
            for row in range(5, -1, -1):
                if b[row, col] == 0:
                    found_empty = True
                elif found_empty:
                    b[row, col] = 0
        return b

    def _compute_confidence(self, board: np.ndarray) -> float:
        """
        Score the raw (un-gravity-filtered) board.

        Scan each column BOTTOM→TOP.  A piece sitting above an empty cell is
        physically impossible (floating) and indicates a misdetection.  Each
        such violation subtracts 0.15.  Piece-count imbalance > 1 subtracts 0.2.

        Note: call this on the RAW board before applying _apply_gravity_filter.
        After filtering there are no floating pieces, so violations would always
        be zero and the check would be meaningless.
        """
        confidence = 1.0
        for col in range(7):
            found_empty = False
            for row in range(5, -1, -1):   # bottom → top
                if board[row, col] == 0:
                    found_empty = True      # found empty below
                elif found_empty:
                    confidence -= 0.15      # piece above an empty = floating
        red    = int(np.sum(board == 1))
        yellow = int(np.sum(board == 2))
        if abs(red - yellow) > 1:
            confidence -= 0.2
        return max(0.0, min(1.0, confidence))

    def _draw_debug(self, image: np.ndarray, board: np.ndarray,
                     grid_centers: np.ndarray, board_contour: np.ndarray,
                     holes: List, confidence: float,
                     fallback_used: bool) -> np.ndarray:
        dbg = image.copy()
        color = (0, 165, 255) if fallback_used else (0, 255, 0)
        cv2.drawContours(dbg, [board_contour], -1, color, 2)

        # Detected holes in cyan (before grid fitting)
        for hx, hy in holes:
            cv2.circle(dbg, (int(hx), int(hy)), 4, (255, 255, 0), 1)

        # Grid cells
        for r in range(6):
            for c in range(7):
                cx, cy = int(grid_centers[r, c, 0]), int(grid_centers[r, c, 1])
                cell   = board[r, c]
                col    = (90, 90, 90) if cell == 0 else ((0, 0, 255) if cell == 1 else (0, 220, 255))
                label  = "." if cell == 0 else ("R" if cell == 1 else "Y")
                cv2.circle(dbg, (cx, cy), 6, col, -1)
                cv2.putText(dbg, label, (cx - 6, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)

        tag = "FALLBACK " if fallback_used else ""
        cv2.putText(dbg,
                    f"{tag}Conf:{confidence:.2f}  holes:{len(holes)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if confidence > 0.8 else (0, 165, 255), 2)
        return dbg


# ─────────────────────────────────────────────────────────────────────────────
# Locking wrapper
# ─────────────────────────────────────────────────────────────────────────────

class LockedBoardDetector:
    """
    Two-stage locking wrapper around BoardDetector.

    Stage 1 — Candidate: first board detection stores a candidate grid.
              Subsequent frames classify pieces at the *candidate* position
              regardless of whether the blue frame is re-detected.  A "good
              frame" is defined by classification quality (conf ≥ LOCK_MIN_CONF),
              not by whether the frame was found again.  This makes locking
              robust to an unstable/flickering bounding box.

    Stage 2 — Locked: after LOCK_FRAMES consecutive good classification frames
              the grid is confirmed frozen.  Only classification runs each frame.

    CANDIDATE_RESET consecutive bad classification frames (board truly gone or
    covered) drop the candidate and restart.

    Console output: board grid is printed on lock and whenever the board state
    changes while locked.
    """

    LOCK_MIN_CONF   = 0.45   # Classification confidence needed for a "good" frame
    LOCK_FRAMES     = 3      # Good frames needed to confirm lock
    CANDIDATE_RESET = 20     # Consecutive bad frames to drop candidate entirely
    UNLOCK_FRAMES   = 20     # Consecutive bad locked frames → unlock

    # Piece removal requires this many consecutive absent frames.
    # Fan/lighting flicker typically lasts 1–3 frames; real removals are
    # permanent.  5 frames ≈ ~0.5 s at 10 fps — fast enough to track gameplay
    # while ignoring shadows.
    REMOVE_THRESHOLD = 5

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.detector = BoardDetector(config)
        self._locked_centers: Optional[np.ndarray] = None
        self._locked_contour: Optional[np.ndarray] = None
        self._cand_centers:   Optional[np.ndarray] = None
        self._cand_contour:   Optional[np.ndarray] = None
        self._good_count  = 0
        self._bad_streak  = 0
        self._lock_bad    = 0
        self.is_locked    = False
        self._last_board:    Optional[np.ndarray] = None  # for change detection
        self._stable_board:  Optional[np.ndarray] = None  # temporally-smoothed state
        self._absent_count:  np.ndarray = np.zeros((6, 7), dtype=np.int32)

    @property
    def config(self) -> DetectionConfig:
        return self.detector.config

    @config.setter
    def config(self, value: DetectionConfig):
        self.detector.config = value
        self.unlock()

    def unlock(self):
        """Force re-detection on the next frame."""
        self._locked_centers = None
        self._locked_contour = None
        self._cand_centers   = None
        self._cand_contour   = None
        self._good_count  = 0
        self._bad_streak  = 0
        self._lock_bad    = 0
        self.is_locked    = False
        self._last_board   = None
        self._stable_board = None
        self._absent_count = np.zeros((6, 7), dtype=np.int32)

    def detect(self, image: np.ndarray, debug: bool = False) -> DetectionResult:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if self.is_locked:
            return self._detect_locked(image, hsv, debug)
        return self._detect_unlocked(image, hsv, debug)

    # ── Locked fast path ─────────────────────────────────────────────────────

    def _detect_locked(self, image: np.ndarray, hsv: np.ndarray,
                        debug: bool) -> DetectionResult:
        board_raw  = self.detector._classify_cells(hsv, self._locked_centers)
        confidence = self.detector._compute_confidence(board_raw)
        board_filt = self.detector._apply_gravity_filter(board_raw)
        board      = self._temporal_smooth(board_filt)

        if confidence < 0.2:
            self._lock_bad += 1
            if self._lock_bad >= self.UNLOCK_FRAMES:
                print("[Board] Lock released — board lost from view")
                self.unlock()
        else:
            self._lock_bad = 0
            # Print board only when the stable state actually changes
            if self._last_board is None or not np.array_equal(board, self._last_board):
                self._last_board = board.copy()
                self._print_board(board)

        debug_img = None
        if debug:
            debug_img = self.detector._draw_debug(
                image, board, self._locked_centers,
                self._locked_contour, [], confidence, False,
            )
            cv2.drawContours(debug_img, [self._locked_contour], -1, (255, 220, 0), 3)
            cv2.putText(debug_img, "LOCKED", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)

        return DetectionResult(
            board=board, confidence=confidence,
            grid_centers=self._locked_centers,
            board_contour=self._locked_contour,
            debug_image=debug_img,
        )

    # ── Pre-lock detection ────────────────────────────────────────────────────

    def _detect_unlocked(self, image: np.ndarray, hsv: np.ndarray,
                          debug: bool) -> DetectionResult:
        result = self.detector.detect(image, debug=debug)

        # Only update the candidate when the fresh detection is itself verified
        # good — this prevents a slightly-wrong bounding box on the next frame
        # from overwriting a good candidate and tanking confidence.
        if result.board_contour is not None and result.grid_centers is not None:
            fresh_raw  = self.detector._classify_cells(hsv, result.grid_centers)
            fresh_conf = self.detector._compute_confidence(fresh_raw)
            if fresh_conf >= self.LOCK_MIN_CONF:
                self._cand_centers = result.grid_centers.copy()
                self._cand_contour = result.board_contour.copy()

        # If we have a candidate, classify at that position regardless of
        # whether the blue frame was detected this frame.
        if self._cand_centers is not None:
            cand_raw  = self.detector._classify_cells(hsv, self._cand_centers)
            cand_conf = self.detector._compute_confidence(cand_raw)
            cand_board = self.detector._apply_gravity_filter(cand_raw)

            # Override result with the candidate-based classification
            result.grid_centers  = self._cand_centers
            result.board_contour = self._cand_contour
            result.board         = cand_board     # gravity-filtered
            result.confidence    = cand_conf

            is_good = cand_conf >= self.LOCK_MIN_CONF

            if is_good:
                self._bad_streak = 0
                self._good_count += 1
                if self._good_count >= self.LOCK_FRAMES:
                    self._locked_centers = self._cand_centers.copy()
                    self._locked_contour = self._cand_contour.copy()
                    self.is_locked    = True
                    self._good_count  = 0
                    self._last_board  = cand_board.copy()
                    print("[Board] Locked ✓")
                    self._print_board(cand_board)
                else:
                    print(f"[Board] Good frame {self._good_count}/{self.LOCK_FRAMES} "
                          f"— conf {cand_conf:.2f}")
            else:
                self._bad_streak += 1
                if self._bad_streak >= self.CANDIDATE_RESET:
                    print(f"[Board] Candidate dropped ({self._bad_streak} bad frames)")
                    self._cand_centers = None
                    self._cand_contour = None
                    self._good_count   = 0
                    self._bad_streak   = 0

        return result

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _temporal_smooth(self, board: np.ndarray) -> np.ndarray:
        """
        Debounce piece removal.

        New pieces are accepted immediately (1 frame).
        A piece is only removed from the stable state after it has been
        absent for REMOVE_THRESHOLD consecutive frames — this filters out
        1–3 frame shadows from a spinning fan or momentary lighting changes.
        """
        if self._stable_board is None:
            self._stable_board = board.copy()
            self._absent_count = np.zeros((6, 7), dtype=np.int32)
            return self._stable_board.copy()

        for r in range(6):
            for c in range(7):
                detected = board[r, c]
                stable   = self._stable_board[r, c]

                if detected != 0:
                    # Piece visible this frame — accept immediately
                    self._stable_board[r, c] = detected
                    self._absent_count[r, c] = 0
                elif stable != 0:
                    # Piece was there but not seen this frame
                    self._absent_count[r, c] += 1
                    if self._absent_count[r, c] >= self.REMOVE_THRESHOLD:
                        self._stable_board[r, c] = 0   # confirmed gone
                    # else: keep the piece in the stable state
                # else: was empty, still empty — nothing to do

        return self._stable_board.copy()

    @staticmethod
    def _print_board(board: np.ndarray):
        """Print the board grid to the console."""
        print(board_to_string(board))


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def board_to_string(board: np.ndarray) -> str:
    symbols = {0: ".", 1: "R", 2: "Y"}
    lines = ["  " + " ".join(str(i) for i in range(7)), "  " + "-" * 13]
    for row in range(6):
        lines.append(f"{row}|" + " ".join(symbols[board[row, c]] for c in range(7)) + "|")
    lines.append("  " + "-" * 13)
    return "\n".join(lines)
