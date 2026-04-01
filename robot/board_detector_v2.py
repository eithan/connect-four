"""
Board Detector V2 — Calibration-first, delta-based piece detection.

Core insight
------------
The transparent holes in a Connect Four board let background objects (skin,
clothing, furniture) appear inside cells.  Fixed HSV thresholds cannot
reliably tell a yellow piece from a yellow shirt seen through a hole.

V2 solution: measure *change*, not absolute colour.

  1. Lock board geometry once (perspective warp → canonical top-down view).
  2. Capture a per-cell HSV baseline from the EMPTY board (explicit calibrate
     step, once per game).
  3. Classify each cell:
       • ΔSaturation ≥ threshold (something new appeared)   [delta gate]
       • Absolute hue / saturation gate                     [colour gate]
       • Low saturation variance in sample area             [uniformity gate]
  4. Gravity filter + lightweight temporal smoothing (3 add / 8 remove frames).

Drop-in interface — same detect() / is_locked / unlock() API as
LockedBoardDetector.  game_loop_v2.py adds the calibration lifecycle via
detector.calibrate(frame).
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Reuse DetectionResult so GameLoop.draw() works without changes.
from board_detector import DetectionResult

EMPTY, RED, YELLOW = 0, 1, 2
ROWS, COLS = 6, 7

# Canonical warped-board dimensions — each cell is CELL_PX × CELL_PX.
CELL_PX = 100
WARP_W  = COLS * CELL_PX   # 700
WARP_H  = ROWS * CELL_PX   # 600


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionConfigV2:
    """
    Thresholds for V2 detection.  Defaults work under typical indoor
    lighting with a physical Connect Four board.  You rarely need to
    change these — calibration handles lighting variation automatically.
    """

    # ── Board (blue frame) detection ──────────────────────────────────────
    # S ≥ 80 targets the vivid blue plastic of the board while rejecting
    # gray-blue shirts (S ≈ 20-60) and other muted blue background objects.
    # The detection loop relaxes these incrementally if the board isn't found.
    board_hsv_low:  Tuple[int, int, int] = (90,  80, 40)
    board_hsv_high: Tuple[int, int, int] = (140, 255, 255)

    # Minimum ratio of contour area to its convex-hull area.
    # A rectangular board ≈ 0.85+.  A human body / irregular blob ≈ 0.3-0.5.
    board_min_solidity: float = 0.60

    # Minimum ratio of contour area to its bounding-rect area.
    # Guards against thin L-shapes or very concave blobs passing the aspect check.
    board_min_fill: float = 0.40

    # ── Absolute piece colour gates ───────────────────────────────────────
    # Red wraps around the 0/180 hue boundary (OpenCV 0-179 scale).
    red_h_max:    int = 12     # H < red_h_max  OR  H > red_h_wrap → red hue
    red_h_wrap:   int = 158
    red_s_min:    int = 120    # minimum saturation for a confirmed red piece

    yellow_h_min: int = 15
    yellow_h_max: int = 42     # standard yellow / golden plastic
    yellow_s_min: int = 90
    yellow_v_min: int = 80

    # ── Delta gate ────────────────────────────────────────────────────────
    # Minimum median-saturation increase above the empty-board baseline
    # before colour gates are applied.
    #   Lighting drift   (empty → empty):  ΔS  0 – 12
    #   Piece placed     (empty → piece):  ΔS  35 – 100+
    # Set to 0 to disable (pure absolute mode; useful without calibration).
    delta_s_min: int = 18

    # ── Uniformity gate ───────────────────────────────────────────────────
    # Solid plastic pieces have low S variance in the sample circle.
    # Patterned background visible through empty holes has high variance.
    # Pieces near board edges can have std_S ≈ 60-70 due to slight warp
    # distortion at corners; backgrounds through holes rarely exceed 80.
    uniformity_s_max: float = 80.0

    # ── Coverage threshold ────────────────────────────────────────────────
    # Fraction of sample-circle pixels whose hue falls in the piece range.
    piece_coverage_min: float = 0.28

    # ── Sampling ──────────────────────────────────────────────────────────
    # Sample-circle radius as a fraction of the half-cell size (CELL_PX/2).
    # 0.55 → radius ≈ 27 px inside a 100 px cell; samples the hole interior
    # while avoiding the cell wall / lens distortion at the edges.
    sample_radius_frac: float = 0.55

    # ── Temporal smoothing ────────────────────────────────────────────────
    add_threshold:    int = 3   # consecutive frames a new piece must appear
    remove_threshold: int = 8   # consecutive absent frames before removing

    # ── Locking ───────────────────────────────────────────────────────────
    lock_frames: int = 3   # board must be visible this many frames to lock

    # ── Gravity filter ────────────────────────────────────────────────────
    gravity_filter: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell calibration snapshot
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CellBaseline:
    """HSV statistics from one cell sampled on the empty board."""
    median_h: float
    median_s: float
    median_v: float
    std_s:    float
    # If a piece was already present at calibration, record its colour so we
    # always report it (it can never go EMPTY mid-game from our perspective).
    sentinel: int = EMPTY


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Return 4 corners in [TL, TR, BR, BL] order."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _apply_gravity(board: np.ndarray) -> np.ndarray:
    """Remove floating pieces (no piece may have an empty slot beneath it)."""
    out = board.copy()
    for c in range(COLS):
        found_empty = False
        for r in range(ROWS - 1, -1, -1):   # bottom → top
            if board[r, c] == EMPTY:
                found_empty = True
            elif found_empty:
                out[r, c] = EMPTY
    return out


def _compute_confidence(board: np.ndarray) -> float:
    """Heuristic confidence in [0, 1] based on gravity compliance and balance."""
    total = int(np.sum(board != EMPTY))
    if total == 0:
        return 1.0
    viol = 0
    for c in range(COLS):
        fe = False
        for r in range(ROWS - 1, -1, -1):
            if board[r, c] == EMPTY:
                fe = True
            elif fe:
                viol += 1
    grav  = 1.0 - viol / max(total, 1)
    red_n = int(np.sum(board == RED))
    yel_n = int(np.sum(board == YELLOW))
    bal   = 1.0 if abs(red_n - yel_n) <= 2 else 0.75
    return float(np.clip(grav * bal, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Core single-frame detector
# ─────────────────────────────────────────────────────────────────────────────

class BoardDetectorV2:
    """
    Single-frame board detector.

    Lifecycle (managed by LockedBoardDetectorV2)
    ---------------------------------------------
    1. find_board_geometry(frame) → bool   — try without committing
    2. commit_lock()                        — lock the geometry
    3. calibrate(frame)                     — capture empty-board baseline
    4. detect(frame, debug)  → DetectionResult
    """

    def __init__(self, config: Optional[DetectionConfigV2] = None):
        self.config = config or DetectionConfigV2()

        # Set at lock time
        self._warp_M:        Optional[np.ndarray] = None
        self._warp_M_inv:    Optional[np.ndarray] = None
        self._cell_centers:  Optional[np.ndarray] = None  # (ROWS, COLS, 2) warped px
        self._board_contour: Optional[np.ndarray] = None

        # Candidate geometry (found but not yet committed)
        self._candidate: Optional[tuple] = None

        # Set at calibration time
        self._baselines: Optional[List[List[Optional[CellBaseline]]]] = None

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def is_locked(self) -> bool:
        return self._warp_M is not None

    @property
    def is_calibrated(self) -> bool:
        return self._baselines is not None

    # ── Locking ───────────────────────────────────────────────────────────────

    def find_board_geometry(self, frame: np.ndarray) -> bool:
        """
        Try to find the board and compute the warp, without committing.
        Returns True if found (geometry stored as candidate).
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        result = self._compute_geometry(frame, hsv)
        if result is None:
            self._candidate = None
            return False
        self._candidate = result
        return True

    def commit_lock(self) -> bool:
        """Commit the last candidate geometry as the locked board."""
        if self._candidate is None:
            return False
        self._warp_M, self._warp_M_inv, self._cell_centers, self._board_contour = self._candidate
        self._candidate = None
        return True

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(self, frame: np.ndarray) -> bool:
        """
        Sample each cell from *frame* (should show an empty board) and store
        the HSV baseline.  Must be called after commit_lock().
        """
        if not self.is_locked:
            return False
        warped_hsv = self._warp_to_hsv(frame)
        baselines: List[List[Optional[CellBaseline]]] = []
        for r in range(ROWS):
            row: List[Optional[CellBaseline]] = []
            for c in range(COLS):
                px = self._sample_cell(warped_hsv, r, c)
                if px is None:
                    row.append(None)
                    continue
                h = px[:, 0].astype(float)
                s = px[:, 1].astype(float)
                v = px[:, 2].astype(float)
                # Detect sentinel: piece already present at calibration time
                sentinel = self._classify_absolute(h, s, v)
                row.append(CellBaseline(
                    median_h=float(np.median(h)),
                    median_s=float(np.median(s)),
                    median_v=float(np.median(v)),
                    std_s=float(np.std(s)),
                    sentinel=sentinel,
                ))
            baselines.append(row)
        self._baselines = baselines
        return True

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, debug: bool = False) -> DetectionResult:
        if not self.is_locked:
            return DetectionResult(
                board=np.zeros((ROWS, COLS), dtype=np.int8),
                confidence=0.0,
                errors=["Board not locked"],
            )

        warped_hsv = self._warp_to_hsv(frame)
        raw = np.zeros((ROWS, COLS), dtype=np.int8)

        for r in range(ROWS):
            for c in range(COLS):
                px = self._sample_cell(warped_hsv, r, c)
                if px is None:
                    continue
                h = px[:, 0].astype(float)
                s = px[:, 1].astype(float)
                v = px[:, 2].astype(float)
                baseline = (self._baselines[r][c]
                            if self._baselines is not None else None)

                # Sentinel: piece present at calibration — always report it
                if baseline is not None and baseline.sentinel != EMPTY:
                    raw[r, c] = baseline.sentinel
                    continue

                raw[r, c] = self._classify(h, s, v, baseline)

        if self.config.gravity_filter:
            raw = _apply_gravity(raw)

        grid_centers = self._warp_centers_to_image()
        conf = _compute_confidence(raw)

        dbg = None
        if debug:
            wbgr = cv2.warpPerspective(frame, self._warp_M, (WARP_W, WARP_H))
            dbg  = self._draw_debug(wbgr, raw)

        return DetectionResult(
            board=raw,
            confidence=conf,
            grid_centers=grid_centers,
            board_contour=self._board_contour,
            debug_image=dbg,
            errors=[],
        )

    def reset(self):
        self._warp_M        = None
        self._warp_M_inv    = None
        self._cell_centers  = None
        self._board_contour = None
        self._candidate     = None
        self._baselines     = None

    # unlock is the name game_loop.py uses
    unlock = reset

    # ── Board detection ───────────────────────────────────────────────────────

    def _compute_geometry(self, frame: np.ndarray, hsv: np.ndarray) -> Optional[tuple]:
        """Find board, compute warp + grid. Returns (M, M_inv, centers, contour)."""
        contour = self._find_board_contour(hsv)
        if contour is None:
            return None

        peri   = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            src = _order_corners(approx.reshape(4, 2))
        else:
            x, y, w, h = cv2.boundingRect(contour)
            src = _order_corners(np.array(
                [[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32))

        dst = np.array([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]],
                       dtype=np.float32)
        M     = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)

        warped_hsv   = cv2.warpPerspective(hsv, M, (WARP_W, WARP_H))
        cell_centers = self._fit_grid(warped_hsv)

        return M, M_inv, cell_centers, contour

    def _find_board_contour(self, hsv: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the bright-blue board frame.

        Tries progressively looser thresholds (V first, then S) to cope with
        dim lighting, but shape quality checks (aspect ratio, solidity, fill)
        are enforced at every tier to reject human bodies, shirts, and other
        false matches that pass the colour gate.

        Quality checks
        --------------
        Aspect ratio  0.65 – 2.0   Connect Four ≈ 1.17; generous for angle
        Solidity      > 0.60        board ≈ 0.85; body with arms ≈ 0.30-0.45
        Fill          > 0.40        rectangular objects fill their bounding box
        """
        cfg = self.config
        h_img, w_img = hsv.shape[:2]
        img_area = h_img * w_img

        base_lo = list(cfg.board_hsv_low)   # (H, S, V)

        # Tier 1: primary thresholds
        # Tier 2: relax V by -15 (mild dim lighting)
        # Tier 3: relax V by -25, S by -15 (dim + slightly less vivid board)
        tiers = [
            (base_lo[0], base_lo[1],            max(base_lo[2],       8)),
            (base_lo[0], base_lo[1],            max(base_lo[2] - 15,  8)),
            (base_lo[0], max(base_lo[1] - 15, 40), max(base_lo[2] - 25, 8)),
        ]

        for (lo_h, lo_s, lo_v) in tiers:
            lo_arr = np.array([lo_h, lo_s, lo_v], dtype=np.uint8)
            hi_arr = np.array(cfg.board_hsv_high, dtype=np.uint8)
            mask = cv2.inRange(hsv, lo_arr, hi_arr)

            # Fill circular holes so the board reads as a solid blue blob
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
            k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            best, best_area = None, 0.0
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < img_area * 0.02:       # too small
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                ar = bw / max(bh, 1)
                if not (0.65 < ar < 2.0):        # wrong aspect ratio
                    continue

                # Solidity: board is compact; human body with spread arms is not
                hull_area = cv2.contourArea(cv2.convexHull(cnt))
                solidity  = area / max(hull_area, 1.0)
                if solidity < cfg.board_min_solidity:
                    continue

                # Fill: rectangular objects fill their bounding box well
                fill = area / max(bw * bh, 1)
                if fill < cfg.board_min_fill:
                    continue

                if area > best_area:
                    best_area, best = area, cnt

            if best is not None:
                return best

        return None

    def _fit_grid(self, warped_hsv: np.ndarray) -> np.ndarray:
        """
        Return (ROWS, COLS, 2) cell-centre coordinates in warped-image px.

        Detects actual hole centres and computes their median offset from the
        ideal uniform grid; applies that correction so the sample circles land
        on the holes even if the warp corners were slightly imprecise.
        """
        holes = self._find_holes(warped_hsv)

        ideal_cx = np.array([CELL_PX // 2 + c * CELL_PX for c in range(COLS)], dtype=float)
        ideal_cy = np.array([CELL_PX // 2 + r * CELL_PX for r in range(ROWS)], dtype=float)

        dx, dy = 0.0, 0.0
        if len(holes) >= 4:
            offsets_x, offsets_y = [], []
            for (hx, hy) in holes:
                ci = int(np.clip(round((hx - CELL_PX // 2) / CELL_PX), 0, COLS - 1))
                ri = int(np.clip(round((hy - CELL_PX // 2) / CELL_PX), 0, ROWS - 1))
                offsets_x.append(hx - ideal_cx[ci])
                offsets_y.append(hy - ideal_cy[ri])
            dx = float(np.median(offsets_x))
            dy = float(np.median(offsets_y))

        centers = np.zeros((ROWS, COLS, 2), dtype=np.float32)
        for r in range(ROWS):
            for c in range(COLS):
                centers[r, c, 0] = ideal_cx[c] + dx
                centers[r, c, 1] = ideal_cy[r] + dy
        return centers

    def _find_holes(self, warped_hsv: np.ndarray) -> List[Tuple[float, float]]:
        """Find circular non-blue regions (holes) in the warped board image."""
        cfg = self.config
        blue = cv2.inRange(warped_hsv,
                           np.array(cfg.board_hsv_low, dtype=np.uint8),
                           np.array(cfg.board_hsv_high, dtype=np.uint8))
        hole_mask = cv2.bitwise_not(blue)

        # Mask out image edges where the board frame wraps
        b = int(CELL_PX * 0.15)
        hole_mask[:b, :] = hole_mask[-b:, :] = 0
        hole_mask[:, :b] = hole_mask[:, -b:] = 0

        cnts, _ = cv2.findContours(hole_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        out: List[Tuple[float, float]] = []
        min_a = (CELL_PX * 0.20) ** 2
        max_a = (CELL_PX * 0.75) ** 2
        for cnt in cnts:
            a = cv2.contourArea(cnt)
            if not (min_a < a < max_a):
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            if 4 * np.pi * a / (peri * peri) < 0.30:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            out.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
        return out

    # ── Sampling ──────────────────────────────────────────────────────────────

    def _warp_to_hsv(self, frame: np.ndarray) -> np.ndarray:
        warped = cv2.warpPerspective(frame, self._warp_M, (WARP_W, WARP_H))
        return cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    def _sample_cell(self, warped_hsv: np.ndarray, r: int, c: int) -> Optional[np.ndarray]:
        """Return HSV pixel array (N, 3) sampled within the hole circle, or None."""
        cx = int(round(float(self._cell_centers[r, c, 0])))
        cy = int(round(float(self._cell_centers[r, c, 1])))
        radius = max(4, int(CELL_PX * self.config.sample_radius_frac / 2))

        h_img, w_img = warped_hsv.shape[:2]
        x0 = max(0, cx - radius)
        x1 = min(w_img, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(h_img, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return None

        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        px = warped_hsv[y0:y1, x0:x1][mask]
        return px if len(px) >= 12 else None

    # ── Classification ────────────────────────────────────────────────────────

    def _classify(self, h, s, v, baseline: Optional[CellBaseline]) -> int:
        """
        Gate 1 (delta): require saturation increase above baseline.
        Gate 2 (uniformity): reject textured backgrounds.
        Gate 3 (colour): confirm red / yellow hue + absolute saturation.
        """
        cfg   = self.config
        med_s = float(np.median(s))
        std_s = float(np.std(s))

        # ── Gate 1: delta or absolute floor ──────────────────────────────
        if baseline is not None and cfg.delta_s_min > 0:
            if med_s - baseline.median_s < cfg.delta_s_min:
                return EMPTY
        else:
            # No baseline — use a higher absolute floor
            if med_s < 55:
                return EMPTY

        # ── Gate 2: uniformity ────────────────────────────────────────────
        if std_s > cfg.uniformity_s_max:
            return EMPTY

        # ── Gate 3: colour ────────────────────────────────────────────────
        return self._classify_absolute(h, s, v)

    def _classify_absolute(self, h, s, v) -> int:
        """Pure colour classification (no baseline required)."""
        cfg   = self.config
        med_s = float(np.median(s))
        med_v = float(np.median(v))

        if med_s >= cfg.red_s_min:
            red_mask = (h <= cfg.red_h_max) | (h >= cfg.red_h_wrap)
            if float(np.mean(red_mask)) >= cfg.piece_coverage_min:
                return RED

        if med_s >= cfg.yellow_s_min and med_v >= cfg.yellow_v_min:
            yel_mask = (h >= cfg.yellow_h_min) & (h <= cfg.yellow_h_max)
            if float(np.mean(yel_mask)) >= cfg.piece_coverage_min:
                return YELLOW

        return EMPTY

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _warp_centers_to_image(self) -> Optional[np.ndarray]:
        """Inverse-warp cell centres from canonical space to original image coords."""
        if self._cell_centers is None or self._warp_M_inv is None:
            return None
        pts  = self._cell_centers.reshape(-1, 1, 2).astype(np.float32)
        orig = cv2.perspectiveTransform(pts, self._warp_M_inv)
        return orig.reshape(ROWS, COLS, 2).astype(np.int32)

    def _draw_debug(self, warped_bgr: np.ndarray, board: np.ndarray) -> np.ndarray:
        img    = warped_bgr.copy()
        radius = max(4, int(CELL_PX * self.config.sample_radius_frac / 2))
        cmap   = {EMPTY: (80, 80, 80), RED: (0, 0, 220), YELLOW: (0, 200, 220)}
        if self._cell_centers is None:
            return img
        for r in range(ROWS):
            for c in range(COLS):
                cx  = int(round(float(self._cell_centers[r, c, 0])))
                cy  = int(round(float(self._cell_centers[r, c, 1])))
                val = int(board[r, c])
                cv2.circle(img, (cx, cy), radius, cmap.get(val, (80, 80, 80)), 2)
                if val != EMPTY:
                    cv2.putText(img, "R" if val == RED else "Y",
                                (cx - 8, cy + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                cmap[val], 2)
        return img


# ─────────────────────────────────────────────────────────────────────────────
# Stateful wrapper — locking + temporal smoothing
# ─────────────────────────────────────────────────────────────────────────────

class LockedBoardDetectorV2:
    """
    Drop-in replacement for LockedBoardDetector.

    Phases
    ------
    SEARCHING   find_board_geometry called each frame; lock after N successes
    LOCKED      geometry committed; calibrate(frame) may be called
    CALIBRATED  delta detection + temporal smoothing active
    """

    def __init__(self, config: Optional[DetectionConfigV2] = None):
        self.config   = config or DetectionConfigV2()
        self._inner   = BoardDetectorV2(self.config)
        self._lock_ctr = 0

        # Temporal state arrays
        self._state   = np.zeros((ROWS, COLS), dtype=np.int8)
        self._pending = np.zeros((ROWS, COLS), dtype=np.int8)
        self._add_ctr = np.zeros((ROWS, COLS), dtype=np.int8)
        self._rm_ctr  = np.zeros((ROWS, COLS), dtype=np.int8)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_locked(self) -> bool:
        return self._inner.is_locked

    @property
    def is_calibrated(self) -> bool:
        return self._inner.is_calibrated

    def calibrate(self, frame: np.ndarray) -> bool:
        """Capture empty-board baseline.  Call once after is_locked is True."""
        return self._inner.calibrate(frame)

    def unlock(self):
        """Reset to searching state (clears lock AND calibration)."""
        self._inner.reset()
        self._lock_ctr = 0
        self._state[:]   = 0
        self._pending[:] = 0
        self._add_ctr[:] = 0
        self._rm_ctr[:]  = 0

    # alias used by game_loop.py
    reset = unlock

    def detect(self, frame: np.ndarray, debug: bool = False) -> DetectionResult:
        if not self.is_locked:
            return self._seek_lock(frame)
        return self._detect_locked(frame, debug)

    # ── Locking phase ─────────────────────────────────────────────────────────

    def _seek_lock(self, frame: np.ndarray) -> DetectionResult:
        found = self._inner.find_board_geometry(frame)
        if found:
            self._lock_ctr += 1
        else:
            self._lock_ctr = 0

        if self._lock_ctr >= self.config.lock_frames:
            self._inner.commit_lock()
            self._lock_ctr = 0

        return DetectionResult(
            board=np.zeros((ROWS, COLS), dtype=np.int8),
            confidence=0.2 if found else 0.0,
            errors=["Locking..." if found else "Searching for board..."],
        )

    # ── Locked phase ──────────────────────────────────────────────────────────

    def _detect_locked(self, frame: np.ndarray, debug: bool) -> DetectionResult:
        raw    = self._inner.detect(frame, debug)
        rb     = raw.board
        cfg    = self.config

        for r in range(ROWS):
            for c in range(COLS):
                rv = int(rb[r, c])
                cv = int(self._state[r, c])

                if rv == cv:
                    # Consistent — bleed off counters
                    self._add_ctr[r, c] = 0
                    self._rm_ctr[r, c]  = 0

                elif rv != EMPTY:
                    # Raw detects a piece not yet in smoothed state
                    if rv == int(self._pending[r, c]):
                        self._add_ctr[r, c] = min(self._add_ctr[r, c] + 1, 127)
                    else:
                        self._pending[r, c] = rv
                        self._add_ctr[r, c] = 1
                    self._rm_ctr[r, c] = 0
                    if self._add_ctr[r, c] >= cfg.add_threshold:
                        self._state[r, c]   = rv
                        self._add_ctr[r, c] = 0

                else:
                    # Raw says empty; smoothed state has a piece
                    self._rm_ctr[r, c]  = min(self._rm_ctr[r, c] + 1, 127)
                    self._add_ctr[r, c] = 0
                    if self._rm_ctr[r, c] >= cfg.remove_threshold:
                        self._state[r, c]   = EMPTY
                        self._rm_ctr[r, c]  = 0
                        self._pending[r, c] = EMPTY

        return DetectionResult(
            board=self._state.copy(),
            confidence=raw.confidence,
            grid_centers=raw.grid_centers,
            board_contour=raw.board_contour,
            debug_image=raw.debug_image,
            errors=raw.errors,
        )
