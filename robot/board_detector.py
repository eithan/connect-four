"""
Board Detector — Connect Four
==============================

Pipeline
--------
1. Locate the blue board frame (HSV mask → largest well-shaped blue blob).
1b. (Optional) Perspective-warp the board region to a canonical top-down
    rectangle, removing camera angle distortion.  Enable with
    ``DetectionConfig.perspective_warp = True`` or ``--perspective-warp``.
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
- Perspective warp (when enabled) warps the detected board contour to a
  clean top-down rectangle before hole detection and grid fitting.  Grid
  centres are then inverse-warped back to original image coordinates for
  cell classification (which runs on the original, un-warped HSV image to
  preserve true colour values).  This fixes non-uniform row spacing caused
  by camera angle and improves grid overlay alignment.

External API (unchanged from previous versions)
-----------------------------------------------
  DetectionConfig, DetectionResult, BoardDetector, LockedBoardDetector,
  YOLOEnhancedBoardDetector, PHYSICAL_CONFIG, SCREEN_CONFIG, board_to_string
"""

import os
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
    # S_min raised from 80->110: the plastic board frame is highly saturated
    # (S~180-240); clothing fabric blues are much less saturated (S~80-140).
    # Raising S_min excludes most shirt pixels from the blue mask, which both
    # prevents the board+shirt blob from merging and restores hole visibility
    # (shirt seen through transparent holes is also excluded, so holes appear
    # non-blue again, enabling reliable hole-count metrics).
    board_hsv_low:  Tuple[int, int, int] = (90,  110, 50)
    board_hsv_high: Tuple[int, int, int] = (140, 255, 255)

    # Red pieces (two ranges to wrap the 0/180 hue boundary)
    #
    # S lowered 165→130: under dim indoor lighting, camera auto-exposure
    # causes red plastic S to fluctuate 143-175.  S=130 gives a 13-unit
    # buffer below worst-case dips (S≈143).
    # Empty holes: dim rooms S≈60-110, kitchen background peaks S≈155.
    # At S=130, some kitchen background pixels (H≈12, S≈140-155) will
    # enter the red mask, but piece_threshold (0.38) blocks false
    # positives — background covers <20% of the sample circle.
    red_hsv_low1:  Tuple[int, int, int] = (0,   130, 80)
    red_hsv_high1: Tuple[int, int, int] = (12,  255, 255)
    red_hsv_low2:  Tuple[int, int, int] = (158, 130, 80)
    red_hsv_high2: Tuple[int, int, int] = (180, 255, 255)

    # Yellow / amber / lime-green pieces
    # H 20–92: standard Connect Four yellow plastic reads H≈21-22 under
    #           kitchen/indoor light.  Lower bound 20 gives 1-hue margin while
    #           blocking the persistent warm-orange false-positive at H≈14-15
    #           visible through empty holes when background is slightly warm.
    # S≥100: yellow plastic reads S≈121-170 depending on lighting.  Under dim
    #         indoor light with auto-exposure, S dips to ~121.  S=100 gives a
    #         21-unit buffer.  Empty holes with H in yellow range (20-92) max
    #         out at S≈66 (dim rooms) to S≈91 (kitchen) — safe gap above 100.
    yellow_hsv_low:  Tuple[int, int, int] = (20, 100, 80)
    yellow_hsv_high: Tuple[int, int, int] = (92, 255, 255)

    # Perspective warp — rectify board to top-down before grid fitting.
    # Produces significantly better grid alignment than the non-warped
    # path, especially with angled cameras.  On by default; disable with
    # --no-perspective-warp if needed.
    perspective_warp: bool = True

    # Minimum board area as fraction of image
    min_board_area_ratio: float = 0.02

    # Fraction of the sample circle that must be piece-colour to call it a piece.
    # 0.18 → too loose (background through empty holes easily hits 18%)
    # 0.60 → too strict (piece viewed at a slight angle or with minor grid
    #          offset may only cover 50-55% of the sample circle → missed)
    # 0.38 → catches real pieces (≥40% even with grid jitter or dim lighting)
    #          while blocking patchy background (≤30%).  Lowered from 0.45
    #          because dim/different lighting reduces sample coverage to ~35-45%.
    piece_threshold: float = 0.38

    # Hole circularity minimum (0–1; 1 = perfect circle)
    min_circularity: float = 0.40

    # Verbose logging: per-cell HSV, hole counts, cluster decisions
    verbose: bool = False

    # Adaptive colour detection: classify pieces by measuring HSV change
    # from the empty-board baseline rather than fixed HSV ranges.
    # This eliminates lighting-dependent S_min tuning entirely.
    adaptive: bool = True

    # Adaptive mode thresholds
    # Minimum saturation increase from empty baseline to consider a cell
    # "has a piece" (replaces fixed S_min).  Empty→piece typically shows
    # ΔS ≥ 40–80; empty→empty drift is usually < 15.
    # Lowered 25→20: top-row pieces viewed at angle through the board's
    # transparent face show ΔS≈18-22 in stable state; empty drift stays
    # below 15, so 20 preserves the safety margin.
    adaptive_min_delta_s: int = 20

    # Hue boundary between red and yellow (in OpenCV 0-180 scale).
    # Red wraps around 0/180, so: hue < boundary OR hue > (180 - boundary) → red.
    # Real yellows in this setup often land around H=20-30, so keep the red
    # low-end boundary tight enough that those warm yellows stay yellow.
    adaptive_hue_boundary: int = 15

    # Maximum hue for yellow pieces (OpenCV 0-180 scale).
    # Real yellow pieces sit at H≈20-50.  H=60-140 is green/cyan/blue —
    # that's the board frame, shadows, or background bleeding through holes.
    # Anything between adaptive_hue_boundary and adaptive_yellow_hue_max is
    # yellow; outside that window (but not red) → treated as empty.
    adaptive_yellow_hue_max: int = 60

    # Absolute colour floors used by adaptive classification to reject
    # skin / clothing / warm background seen through transparent slots.
    # Lowered 140→115: top-row red pieces are viewed at an angle through
    # the board's curved slot opening, washing out saturation to S≈115-130.
    # Skin S is typically 20-80, so 115 still safely rejects false positives.
    adaptive_red_min_s: int = 115
    adaptive_red_min_v: int = 90
    adaptive_yellow_min_s: int = 100
    adaptive_yellow_min_v: int = 100


# Physical board preset (default).
# perspective_warp is OFF: the non-warp path fits the grid directly from
# detected hole centroids.  Columns are regularised (even spacing) and missing
# columns are extrapolated from median spacing — this handles partial board
# visibility (e.g. monitor reflection washing out rightmost columns) far better
# than the warp path, which computes a homography from the clipped contour
# corners and then shifts the entire inverse-warped grid.
# Row spacing is left as raw cluster positions (not regularised), which matches
# the actual row layout even when perspective makes rows uneven.
# The looser GRID_GAP_MIN_FRAC (0.30) handles angled-camera row spacing without
# the warp path being needed.
PHYSICAL_CONFIG = DetectionConfig(perspective_warp=False)

# Screen/phone display preset (emitted light — higher saturation)
SCREEN_CONFIG = DetectionConfig(
    board_hsv_low=(85,  80, 80),
    board_hsv_high=(140, 255, 255),
    red_hsv_low1=(0,   140, 100),
    red_hsv_high1=(12,  255, 255),
    red_hsv_low2=(163, 140, 100),
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
    board:          np.ndarray                    # (6, 7) int8: 0=empty 1=red 2=yellow (post temporal-smooth)
    confidence:     float                         # 0.0 – 1.0
    grid_centers:   Optional[np.ndarray] = None   # (6, 7, 2) int32
    board_contour:  Optional[np.ndarray] = None   # for display
    debug_image:    Optional[np.ndarray] = None
    fallback_used:  bool = False
    errors:         List[str] = field(default_factory=list)
    raw_board:      Optional[np.ndarray] = None   # (6, 7) pre-temporal-smooth per-frame classification


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
        bbox_result = self._find_board_bbox(hsv, image.shape)
        if bbox_result is None:
            return DetectionResult(
                board=np.zeros((6, 7), dtype=np.int8),
                confidence=0.0,
                errors=["Board not found — no blue frame detected"],
            )
        bx, by, bw, bh, board_cnt = bbox_result
        self._board_bbox = (bx, by, bw, bh)

        # ── 1b. Perspective warp (optional) ───────────────────────────────────
        warp_info = None
        if self.config.perspective_warp:
            warp_info = self._compute_warp(board_cnt, bx, by, bw, bh)

        # Two cells of padding on each side of the blue bbox for hole search.
        # Purpose: detect when the blue bbox has been clipped (outer columns
        # sit outside the detected frame).  Two cells (instead of one) handles
        # the case where a nearby blue object (e.g. the Connect Four game box)
        # merges with the board contour, shifting bx left and making bw too
        # narrow — the rightmost 1–2 columns end up beyond bx+bw+half_cell and
        # would never be found with only 1 cell of padding.  When holes are
        # found beyond that threshold, bbox_is_clipped fires and the grid is
        # fitted from the full padded hole set, forcing both missing columns to
        # extrapolate rightward instead of leftward.
        img_h_px, img_w_px = hsv.shape[:2]
        cell_pad  = int(bw / 7.0 * 2)     # ≈ 2 cell widths
        half_cell = (bw / 7.0) * 0.5
        hbx  = max(0,           bx - cell_pad)
        hbw  = min(img_w_px - hbx, bw + 2 * cell_pad)

        # Search the padded region to find holes that might be outside the bbox.
        holes_padded = self._find_holes(hsv, hbx, by, hbw, bh, cell_est_override=bw / 7.0)

        # When holes appear outside the blue bbox the contour is clipped and the
        # warp matrix (derived from those narrow corners) would mis-place the
        # outermost column(s) after inverse-warping.  Fall back to the non-warp
        # path so the grid is fitted directly in image coordinates where the hole
        # positions are exact.
        bbox_is_clipped = False
        if warp_info is not None and len(holes_padded) > 0:
            xs = [h[0] for h in holes_padded]
            if max(xs) > bx + bw + half_cell or min(xs) < bx - half_cell:
                warp_info = None   # contour was clipped — skip warp this frame
                bbox_is_clipped = True

        # For grid fitting, restrict holes to within the actual board bounds
        # (with half-cell tolerance) so false-positive circles in the padded
        # area (face features, clothing beside the board) don't contaminate
        # _fit_grid.  Only when the bbox was genuinely clipped do we keep the
        # full padded set — those holes include columns that fell outside the
        # detected blue frame and are needed for an accurate non-warp fit.
        if bbox_is_clipped:
            holes = holes_padded
        else:
            holes = [h for h in holes_padded
                     if bx - half_cell <= h[0] <= bx + bw + half_cell]

        if warp_info is not None:
            M_fwd, M_inv, warp_w, warp_h = warp_info

            # ── 2–3. Detect holes in ORIGINAL image, regularise in warp space ─
            #
            # Key insight: hole detection is reliable on the original image
            # (proven margins, no warp artifacts).  The perspective warp is
            # only needed for REGULARISATION — in the warped (top-down) space,
            # both row and column spacing are uniform, so we can fit a clean
            # evenly-spaced grid.  No actual image warping required.
            #
            # Pipeline:
            #   a) Find holes in original image (filtered to board bounds)
            #   b) Forward-warp hole positions into the canonical space
            #   c) Fit + regularise the grid in canonical space (both axes)
            #   d) Inverse-warp the regularised grid back to original coords
            #
            # `holes` already computed above (board-bounds-filtered, not clipped).

            cell_est_w = warp_w / 7.0
            fallback_used = False
            grid_centers_w = None

            if len(holes) >= 8:
                # Forward-warp hole positions to canonical space
                holes_w = self._forward_warp_points(holes, M_fwd)
                grid_centers_w = self._fit_grid(
                    holes_w, cell_est_w, 0, warp_w, 0, warp_h,
                )

            if grid_centers_w is None:
                grid_centers_w = self._grid_from_bounds(0, 0, warp_w, warp_h)
                fallback_used = True
                errors.append(f"Perspective warp: hole detection found "
                              f"{len(holes)} circles — using padded-bounds grid")

            # In canonical space both axes are uniform → regularise both
            if not fallback_used and grid_centers_w is not None:
                grid_centers_w = self._regularize_grid_both_axes(grid_centers_w)

            # ── Inverse-warp grid centres back to original image coords ───────
            grid_centers = self._inverse_warp_grid(grid_centers_w, M_inv)

        else:
            # ── Standard (non-warped) pipeline ────────────────────────────────
            # `holes` already computed above (before the clipped-bbox check).

            # ── 3. Fit 6×7 grid ──────────────────────────────────────────────
            cell_est = bw / 7.0
            fallback_used = False
            grid_centers = None

            if len(holes) >= 8:
                # When bbox was clipped, holes span the padded region → use
                # padded bounds so extrapolation reaches clipped columns.
                # Otherwise use original bbox bounds to prevent false positives
                # from the padded area from pulling the grid off-center.
                lo_x = hbx if bbox_is_clipped else bx
                hi_x = (hbx + hbw) if bbox_is_clipped else (bx + bw)
                grid_centers = self._fit_grid(holes, cell_est, lo_x, hi_x, by, by + bh)

            if grid_centers is None:
                grid_centers = self._grid_from_bounds(bx, by, bw, bh)
                fallback_used = True
                errors.append(f"Hole detection found {len(holes)} circles — using padded-bounds grid")

        # ── 4. Classify cells (always on original HSV for true colours) ───────
        # Compute confidence on the raw (unfiltered) board — floating pieces
        # in the raw detection mean the grid is misaligned, which is the
        # signal we want.  Apply the gravity filter afterwards for the result.
        board_raw  = self._classify_cells(hsv, grid_centers)
        confidence = self._compute_confidence(board_raw)
        if fallback_used:
            confidence = min(confidence, 0.65)
        board = self._apply_gravity_filter(board_raw)

        # Build board_contour from actual grid-center extent so the overlay
        # rectangle matches the fitted grid, not the (potentially clipped) blue
        # bounding box.  Add ~0.6-cell padding to include the physical frame
        # around the outermost holes.
        if grid_centers is not None:
            gc_x = grid_centers[:, :, 0]
            gc_y = grid_centers[:, :, 1]
            c_spacing = max(int(np.median(np.diff(grid_centers[0, :, 0]))), 1)
            r_spacing = max(int(np.median(np.diff(grid_centers[:, 0, 1]))), 1)
            pad_x = int(c_spacing * 0.6)
            pad_y = int(r_spacing * 0.6)
            gx0 = max(0,           int(gc_x.min()) - pad_x)
            gy0 = max(0,           int(gc_y.min()) - pad_y)
            gx1 = min(img_w_px - 1, int(gc_x.max()) + pad_x)
            gy1 = min(img_h_px - 1, int(gc_y.max()) + pad_y)
            board_contour = np.array(
                [[gx0, gy0], [gx1, gy0], [gx1, gy1], [gx0, gy1]],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
        else:
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
                          img_shape: Tuple) -> Optional[Tuple[int, int, int, int, np.ndarray]]:
        """
        Find the blue board frame and return (x, y, w, h, contour).
        Uses a large MORPH_CLOSE to fill the hole grid, then picks the
        largest blob with a plausible Connect Four aspect ratio (0.6–2.8).

        The raw contour is returned alongside the bounding rect so the
        perspective-warp step can extract true corner points (not just the
        axis-aligned bounding box).
        """
        cfg = self.config
        mask = cv2.inRange(hsv, np.array(cfg.board_hsv_low), np.array(cfg.board_hsv_high))

        # Fill the circular holes so the board reads as a solid blue rectangle
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Secondary pass: break thin blue connections between the board and
        # nearby clothing/background.  This is especially helpful when the
        # player is close to the camera wearing a blue shirt.
        k_detach = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        detached_mask = cv2.erode(mask, k_detach, iterations=1)
        detached_mask = cv2.dilate(detached_mask, k_detach, iterations=1)
        detached_contours, _ = cv2.findContours(
            detached_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        all_contours = contours + detached_contours
        if not all_contours:
            return None

        img_area     = img_shape[0] * img_shape[1]
        noise_thresh = img_area * 0.001

        # Sort by area largest-first, ignore noise
        deduped: List[np.ndarray] = []
        seen_rects = set()
        for cnt in sorted(all_contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(cnt) <= noise_thresh:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            rect_key = (
                int(round(x / 12.0)),
                int(round(y / 12.0)),
                int(round(w / 12.0)),
                int(round(h / 12.0)),
            )
            if rect_key in seen_rects:
                continue
            seen_rects.add(rect_key)
            deduped.append(cnt)
        candidates = deduped
        if not candidates:
            return None

        img_h, img_w = img_shape[:2]

        # Prefer candidates that not only look like a blue board-sized blob,
        # but also contain a board-like distribution of hole circles inside.
        # When the player's blue shirt is close behind the transparent board,
        # the hole signal gets weaker because the holes are no longer strongly
        # non-blue, so shape cues still need to be able to win on their own.
        best = None
        best_key = None
        best_support = None
        for cnt in candidates[:20]:
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue
            aspect = w / h
            if not (0.6 <= aspect <= 2.8 and w * h >= img_area * cfg.min_board_area_ratio):
                continue
            hole_count, span_x_ratio, span_y_ratio, hole_bounds = self._hole_support_metrics(
                hsv, x, y, w, h
            )
            contour_area = cv2.contourArea(cnt)
            extent = contour_area / max(float(w * h), 1.0)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = contour_area / max(float(hull_area), 1.0)
            perim = cv2.arcLength(cnt, True)
            approx_vertices = 99
            if perim > 0:
                approx = cv2.approxPolyDP(cnt, 0.03 * perim, True)
                approx_vertices = len(approx)

            strong_support = hole_count >= 10 and span_x_ratio >= 0.58 and span_y_ratio >= 0.48
            decent_support = hole_count >= 8 and span_x_ratio >= 0.55 and span_y_ratio >= 0.42
            board_like_shape = (
                extent >= 0.58
                and solidity >= 0.82
                and 0.78 <= aspect <= 1.45
                and approx_vertices <= 8
            )
            not_frame_hog = (
                y > int(img_h * 0.02)
                and x > int(img_w * 0.01)
                and (x + w) < int(img_w * 0.99)
                and h < img_h * 0.92
                and w < img_w * 0.96
            )
            key = (
                strong_support,
                decent_support,
                board_like_shape,
                hole_count,
                round(span_y_ratio * 1000),
                round(span_x_ratio * 1000),
                not_frame_hog,
                round(solidity * 1000),
                round(extent * 1000),
                -abs(aspect - 1.18),
                -abs(approx_vertices - 4),
                -h,
                contour_area,
            )
            if best_key is None or key > best_key:
                best_key = key
                best = (x, y, w, h, cnt)
                best_support = (hole_count, span_x_ratio, span_y_ratio, hole_bounds)

        if best is not None:
            x, y, w, h, cnt = best
            if best_support is not None:
                hole_count, span_x_ratio, span_y_ratio, hole_bounds = best_support
                if hole_count >= 14 and hole_bounds is not None:
                    refined = self._refine_bbox_from_holes(hole_bounds, img_shape)
                    rx, ry, rw, rh, _ = refined
                    bbox_area = max(float(w * h), 1.0)
                    refined_area = max(float(rw * rh), 1.0)
                    center_dx = abs((x + w / 2.0) - (rx + rw / 2.0))
                    center_dy = abs((y + h / 2.0) - (ry + rh / 2.0))
                    size_mismatch = bbox_area / refined_area
                    if (
                        size_mismatch >= 1.22
                        or center_dx > rw * 0.18
                        or center_dy > rh * 0.18
                    ):
                        return refined
                suspicious_bbox = (
                    y <= int(img_h * 0.02)
                    or h >= img_h * 0.88
                    or w >= img_w * 0.94
                    or span_y_ratio < 0.55
                    or span_x_ratio < 0.68
                )
                if hole_count >= 10 and hole_bounds is not None and suspicious_bbox:
                    return self._refine_bbox_from_holes(hole_bounds, img_shape)
            return best

        # Fallback: just the largest blob
        cnt = candidates[0]
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= img_area * cfg.min_board_area_ratio:
            return x, y, w, h, cnt
        return None

    def _hole_support_metrics(self, hsv: np.ndarray, bx: int, by: int,
                               bw: int, bh: int
                               ) -> Tuple[int, float, float, Optional[Tuple[float, float, float, float]]]:
        """
        Estimate whether a board bbox contains a plausible spread of hole circles.

        Returns (count, x_span_ratio, y_span_ratio, hole_bounds), where span
        ratios are the detected-hole span relative to bbox width/height and
        hole_bounds is (min_x, min_y, max_x, max_y) in image coordinates.
        """
        cfg = self.config
        hsv_roi = hsv[by:by + bh, bx:bx + bw]
        if hsv_roi.size == 0:
            return 0, 0.0, 0.0, None

        raw_blue = cv2.inRange(
            hsv_roi, np.array(cfg.board_hsv_low), np.array(cfg.board_hsv_high)
        )
        non_blue = cv2.bitwise_not(raw_blue)
        k_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        non_blue = cv2.morphologyEx(non_blue, cv2.MORPH_OPEN, k_noise)

        cell_est  = bw / 7.0
        min_area  = np.pi * (cell_est * 0.18) ** 2
        max_area  = np.pi * (cell_est * 0.52) ** 2
        margin_top  = cell_est * 0.1
        margin_bot  = cell_est * 0.1
        margin_side = bw * 0.03
        y_lo = by + margin_top
        y_hi = by + bh - margin_bot
        x_lo = bx + margin_side
        x_hi = bx + bw - margin_side

        contours, _ = cv2.findContours(non_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pts: List[Tuple[float, float]] = []
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
            if not (x_lo < cx < x_hi and y_lo < cy < y_hi):
                continue
            pts.append((float(cx), float(cy)))

        if not pts:
            return 0, 0.0, 0.0, None

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        span_x = (max(xs) - min(xs)) / max(float(bw), 1.0)
        span_y = (max(ys) - min(ys)) / max(float(bh), 1.0)
        hole_bounds = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
        return len(pts), float(span_x), float(span_y), hole_bounds

    def _refine_bbox_from_holes(
        self,
        hole_bounds: Tuple[float, float, float, float],
        img_shape: Tuple[int, ...],
    ) -> Tuple[int, int, int, int, np.ndarray]:
        """
        Tighten an oversized blue contour back around the detected hole grid.

        This is mainly a rescue path for cases where nearby blue clothing
        merges with the board and inflates the raw contour.
        """
        img_h, img_w = img_shape[:2]
        min_x, min_y, max_x, max_y = hole_bounds

        hole_w = max(max_x - min_x, 1.0)
        hole_h = max(max_y - min_y, 1.0)
        cell_w = max(hole_w / 6.0, 1.0)
        cell_h = max(hole_h / 5.0, 1.0)

        pad_x = cell_w * 0.82
        pad_y = cell_h * 0.88

        x0 = int(round(max(0.0, min_x - pad_x)))
        y0 = int(round(max(0.0, min_y - pad_y)))
        x1 = int(round(min(float(img_w - 1), max_x + pad_x)))
        y1 = int(round(min(float(img_h - 1), max_y + pad_y)))

        contour = np.array(
            [[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]],
            dtype=np.int32,
        )
        return x0, y0, max(1, x1 - x0 + 1), max(1, y1 - y0 + 1), contour

    # ── Step 1b: Perspective warp ────────────────────────────────────────────

    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        """
        Order 4 points as [top-left, top-right, bottom-right, bottom-left].

        Uses the sum (x+y) to find TL (smallest) and BR (largest), and the
        difference (y-x) to find TR (smallest diff) and BL (largest diff).
        """
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(s)]   # top-left
        ordered[2] = pts[np.argmax(s)]   # bottom-right
        ordered[1] = pts[np.argmin(d)]   # top-right
        ordered[3] = pts[np.argmax(d)]   # bottom-left
        return ordered

    def _compute_warp(self, board_contour: np.ndarray,
                       bx: int, by: int, bw: int, bh: int
                       ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]:
        """
        Compute a perspective transform that maps the detected board region
        to a canonical top-down rectangle.

        Returns (warped_bgr, warped_hsv, M_inverse, dst_w, dst_h) or None
        if the contour doesn't yield a usable quadrilateral.

        Strategy:
          1. Approximate the board contour to a polygon.
          2. If it reduces to exactly 4 vertices, use those as source corners.
          3. Otherwise fall back to a rotatedRect (minimum-area bounding rect),
             which still captures camera rotation and mild perspective.
          4. Destination is a 7:6 aspect rectangle (Connect Four proportions)
             scaled to roughly match the bounding box width for consistent
             cell sizing downstream.
        """
        # Try to approximate the contour to a quadrilateral
        peri = cv2.arcLength(board_contour, True)
        approx = cv2.approxPolyDP(board_contour, 0.02 * peri, True)

        if len(approx) == 4:
            src_pts = approx.reshape(4, 2).astype(np.float32)
        else:
            # Fall back to minimum-area rotated rectangle
            rect = cv2.minAreaRect(board_contour)
            box = cv2.boxPoints(rect)
            src_pts = box.astype(np.float32)

        src_ordered = self._order_corners(src_pts)

        # Destination: canonical top-down rectangle.
        # Preserve the source bounding-box aspect ratio so the board's
        # true proportions are maintained (the full frame includes guide
        # rail + tray, which is taller than the 7:6 playing grid).
        # Forcing 7:6 compressed the board vertically, pushing the grid
        # overlay down by ~1 row.
        dst_w = max(bw, 350)
        dst_h = max(int(dst_w * bh / bw), 300)   # match source aspect ratio
        dst_pts = np.array([
            [0,     0],
            [dst_w, 0],
            [dst_w, dst_h],
            [0,     dst_h],
        ], dtype=np.float32)

        # Sanity check: source quad shouldn't be degenerate
        src_area = cv2.contourArea(src_ordered.reshape(-1, 1, 2).astype(np.int32))
        if src_area < bw * bh * 0.3:
            # Quad is too small relative to bbox — approxPolyDP collapsed
            return None

        M = cv2.getPerspectiveTransform(src_ordered, dst_pts)
        M_inv = cv2.getPerspectiveTransform(dst_pts, src_ordered)

        # Need the original BGR image for warping (we only have HSV here).
        # Store M and let the caller warp.  Actually — since detect() has
        # the image, we reconstruct BGR from HSV.  This avoids changing the
        # method signature of _find_board_bbox.
        # NOTE: BGR→HSV→BGR round-trip is lossy, so we warp the original
        # image in detect() instead.  Return M and M_inv; detect() does
        # the actual cv2.warpPerspective.
        return M, M_inv, dst_w, dst_h

    def _warp_image(self, image: np.ndarray, M: np.ndarray,
                     dst_w: int, dst_h: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply perspective warp and return (warped_bgr, warped_hsv)."""
        warped_bgr = cv2.warpPerspective(image, M, (dst_w, dst_h))
        warped_hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
        return warped_bgr, warped_hsv

    @staticmethod
    def _inverse_warp_grid(grid_w: np.ndarray, M_inv: np.ndarray) -> np.ndarray:
        """
        Map a (6, 7, 2) grid of warped-space centres back to original
        image coordinates using the inverse perspective matrix.
        """
        pts = grid_w.reshape(-1, 2).astype(np.float64)
        # cv2.perspectiveTransform needs shape (N, 1, 2) float32/64
        pts_h = pts.reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts_h, M_inv)
        return mapped.reshape(6, 7, 2).astype(np.int32)

    @staticmethod
    def _forward_warp_points(points: List[Tuple[float, float]],
                              M_fwd: np.ndarray) -> List[Tuple[float, float]]:
        """Map a list of (x, y) points from original image to warped space."""
        if not points:
            return []
        pts = np.array(points, dtype=np.float64).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, M_fwd)
        return [(float(p[0][0]), float(p[0][1])) for p in mapped]

    @staticmethod
    def _inverse_warp_points(points: List[Tuple[float, float]],
                              M_inv: np.ndarray) -> List[Tuple[float, float]]:
        """Map a list of (x, y) points from warped space back to original."""
        if not points:
            return []
        pts = np.array(points, dtype=np.float64).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, M_inv)
        return [(float(p[0][0]), float(p[0][1])) for p in mapped]

    @staticmethod
    def _regularize_grid_both_axes(grid: np.ndarray) -> np.ndarray:
        """
        In a warped (top-down) image, both column AND row spacing should be
        uniform.  Fit straight lines through both axes and return the
        perfectly-evenly-spaced grid.
        """
        rows, cols = grid.shape[:2]

        # Average column x-positions across all rows, then regularise
        col_xs = [float(np.mean(grid[:, c, 0])) for c in range(cols)]
        idx_c = np.arange(cols, dtype=float)
        slope_c, intercept_c = np.polyfit(idx_c, col_xs, 1)
        reg_col_xs = [slope_c * i + intercept_c for i in range(cols)]

        # Average row y-positions across all columns, then regularise
        row_ys = [float(np.mean(grid[r, :, 1])) for r in range(rows)]
        idx_r = np.arange(rows, dtype=float)
        slope_r, intercept_r = np.polyfit(idx_r, row_ys, 1)
        reg_row_ys = [slope_r * i + intercept_r for i in range(rows)]

        out = np.zeros_like(grid)
        for r in range(rows):
            for c in range(cols):
                out[r, c] = [int(round(reg_col_xs[c])), int(round(reg_row_ys[r]))]
        return out

    # ── Step 2: Hole detection ────────────────────────────────────────────────

    def _find_holes(self, hsv: np.ndarray,
                     bx: int, by: int, bw: int, bh: int,
                     cell_est_override: Optional[float] = None) -> List[Tuple[float, float]]:
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

        # Expected hole area based on board width / 7 columns.
        # Use override when the search bbox is wider than the actual board
        # (e.g. padded to catch clipped outer columns) so area limits stay correct.
        cell_est  = cell_est_override if cell_est_override is not None else bw / 7.0
        min_area  = np.pi * (cell_est * 0.18) ** 2
        max_area  = np.pi * (cell_est * 0.52) ** 2

        # Edge exclusion zones — as small as possible.
        #
        # On budget boards (NICETY etc.) the first row of holes is only
        # ~22px below the top of the blue frame.  Any margin above ~20px
        # will exclude it, shifting the grid down by 1 row.  So we use
        # a tiny margin (0.1 cells ≈ 10-12px) — just enough to skip the
        # frame border itself — and rely on _cluster_1d's median-deviation
        # outlier dropping to handle any guide-rail or tray circles that
        # leak through.
        margin_top  = cell_est * 0.1
        margin_bot  = cell_est * 0.1
        margin_side = bw * 0.03
        y_lo = by + margin_top
        y_hi = by + bh - margin_bot
        x_lo = bx + margin_side
        x_hi = bx + bw - margin_side

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
            # Discard circles in the edge exclusion zones
            if not (x_lo < cx < x_hi and y_lo < cy < y_hi):
                continue
            holes.append((float(cx), float(cy)))

        if holes and cfg.verbose:
            ys_sorted = sorted(h[1] for h in holes)
            print(f"[_find_holes] {len(holes)} holes | "
                  f"bbox=({bx},{by},{bw},{bh}) | "
                  f"margin_top={margin_top:.0f} margin_bot={margin_bot:.0f} | "
                  f"y_lo={y_lo:.0f} y_hi={y_hi:.0f} | "
                  f"y_range=[{ys_sorted[0]:.0f}..{ys_sorted[-1]:.0f}]")

        return holes

    # ── Step 3: Grid fitting ──────────────────────────────────────────────────

    def _fit_grid(self, holes: List[Tuple[float, float]],
                   cell_est: float,
                   min_x: float, max_x: float,
                   min_y: float, max_y: float) -> Optional[np.ndarray]:
        """Cluster hole centres into 6 rows × 7 columns."""
        xs = np.array([h[0] for h in holes])
        ys = np.array([h[1] for h in holes])

        verbose = self.config.verbose
        col_means = self._cluster_1d(xs, 7, cell_est, min_x, max_x, drop_last=False, verbose=verbose)
        # For rows, prefer dropping the bottom-most extra cluster (collection tray)
        # rather than merging the two closest (which distorts a real row position).
        row_means = self._cluster_1d(ys, 6, cell_est, min_y, max_y, drop_last=True, verbose=verbose)

        if row_means is not None and self.config.verbose:
            print(f"[_fit_grid] final row_means=[{', '.join(f'{m:.0f}' for m in row_means)}]")

        if col_means is None or row_means is None:
            return None
        if len(col_means) != 7 or len(row_means) != 6:
            return None

        # Regularise columns only: horizontal spacing is consistent (camera
        # is nearly perpendicular to the vertical axis of the board), so a
        # straight-line fit removes noise without moving circles off holes.
        #
        # Do NOT regularise rows: the camera is typically angled slightly up
        # or down, making row spacing non-uniform due to perspective.  Fitting
        # a straight line through uneven row spacings shifts every circle away
        # from the actual holes.  Use the raw cluster means instead.
        col_means = self._regularize_means(col_means)
        # row_means intentionally left as raw cluster positions

        centers = np.zeros((6, 7, 2), dtype=np.int32)
        for r, y in enumerate(row_means):
            for c, x in enumerate(col_means):
                centers[r, c] = [int(round(x)), int(round(y))]
        return centers

    @staticmethod
    def _regularize_means(means: List[float]) -> List[float]:
        """
        Fit a straight line through cluster means (index → position) and
        return the perfectly-evenly-spaced positions on that line.
        Noise in individual cluster means is smoothed out; the overall
        scale and offset come from all detected positions together.
        """
        n   = len(means)
        idx = np.arange(n, dtype=float)
        # np.polyfit: degree-1 → [slope, intercept]
        slope, intercept = np.polyfit(idx, means, 1)
        return [slope * i + intercept for i in range(n)]

    @staticmethod
    def _cluster_1d(values: np.ndarray, n_target: int,
                     cell_size: float,
                     lo: Optional[float] = None,
                     hi: Optional[float] = None,
                     drop_last: bool = False,
                     verbose: bool = False) -> Optional[List[float]]:
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

        if drop_last and n_found != n_target and verbose:
            print(f"[_cluster_1d] rows: {n_found} clusters found, need {n_target} | "
                  f"means=[{', '.join(f'{m:.0f}' for m in means)}]")

        # Reduce to n_target clusters
        while len(means) > n_target:
            if drop_last:
                # For row clustering: extra clusters come from the guide rail
                # (top) or collection tray (bottom).
                #
                # Strategy: compare each end gap to the MEDIAN interior gap.
                # The end that deviates more from the median is the outlier.
                #
                # Key insight: guide rail gaps are SMALLER than normal (rail
                # is close to row 1), while tray gaps may be larger OR
                # smaller.  The old code checked "gap_top > gap_bot" which
                # fails for guide rails — a small top gap looks normal when
                # compared to the bottom gap, but it's actually abnormal
                # relative to the true row spacing.
                if len(means) >= 4:
                    interior_gaps = [means[i + 1] - means[i]
                                     for i in range(1, len(means) - 2)]
                    median_gap = float(np.median(interior_gaps)) if interior_gaps else cell_size
                    gap_top = means[1] - means[0]
                    gap_bot = means[-1] - means[-2]
                    dev_top = abs(gap_top - median_gap)
                    dev_bot = abs(gap_bot - median_gap)
                    if verbose:
                        drop_end = "top" if dev_top > dev_bot else "bottom"
                        print(f"[_cluster_1d] median_gap={median_gap:.0f} "
                              f"gap_top={gap_top:.0f}(dev={dev_top:.0f}) "
                              f"gap_bot={gap_bot:.0f}(dev={dev_bot:.0f}) → drop {drop_end}")
                    if dev_top > dev_bot:
                        means = means[1:]      # top deviates more → outlier
                    else:
                        means = means[:-1]     # bottom deviates more (or tie)
                elif len(means) >= 3:
                    gap_top = means[1] - means[0]
                    gap_bot = means[-1] - means[-2]
                    if abs(gap_top - cell_size) > abs(gap_bot - cell_size):
                        means = means[1:]
                    else:
                        means = means[:-1]
                else:
                    means = means[:-1]
            else:
                # For column clustering (or generic): merge the two groups with
                # the smallest gap between them (noise split).
                diffs = [means[i + 1] - means[i] for i in range(len(means) - 1)]
                idx   = int(np.argmin(diffs))
                means = (means[:idx]
                         + [(means[idx] + means[idx + 1]) / 2]
                         + means[idx + 2:])

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
        # 0.28 × cell_spacing ≈ 74% of the hole radius — sampling the inner
        # core of each hole.  Avoids the hole-edge fringe where lens aberration,
        # blue-frame reflections, and the tray ledge can create spurious colour.
        sample_radius = max(int(cell_spacing * 0.28), 8)

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
                # Skip cells that landed far outside the board bbox (can
                # happen when the perspective warp extrapolates aggressively).
                # A 1-cell margin accommodates slight warp overshoot.
                if hasattr(self, '_board_bbox') and self._board_bbox is not None:
                    bbx, bby, bbw, bbh = self._board_bbox
                    margin = cell_spacing
                    if not (bbx - margin <= cx <= bbx + bbw + margin and
                            bby - margin <= cy <= bby + bbh + margin):
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

                # Debug: log cells with HSV stats for calibration.
                # Log any cell where either colour mask registers > 0.10,
                # or where a piece was detected.
                detected = ("R" if board[r, c] == 1
                            else "Y" if board[r, c] == 2
                            else ".")
                notable = (red_r > 0.10 or yel_r > 0.10 or detected != ".")
                if notable and self.config.verbose:
                    # Include median HSV of the sample area for calibration
                    roi_pixels = hsv[roi_mask > 0]
                    if len(roi_pixels) > 0:
                        h_med = int(np.median(roi_pixels[:, 0]))
                        s_med = int(np.median(roi_pixels[:, 1]))
                        v_med = int(np.median(roi_pixels[:, 2]))
                        print(f"  [cell {r},{c}] red={red_r:.2f} yel={yel_r:.2f}"
                              f"  cx={cx} cy={cy}"
                              f"  HSV=({h_med},{s_med},{v_med}) -> {detected}")

        return board   # raw, un-filtered — caller applies gravity filter

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_gravity_filter(board: np.ndarray) -> np.ndarray:
        """
        Gravity-compress each column: collect all detected pieces regardless
        of their detected row and stack them from the bottom upward.

        This is better than simply removing floating pieces because:
        - A real piece detected 1–2 rows too high (grid slightly misaligned)
          gets repositioned to its correct gravity-valid row instead of deleted.
        - True false-positive pieces (e.g. from background) are still moved
          to valid rows, but the confidence score (computed on the raw board
          BEFORE this step) already penalises them — the quality gate in
          StableStateDetector will reject high-floating-piece frames.

        Call _compute_confidence on the RAW board BEFORE calling this.
        """
        out = np.zeros_like(board)
        for col in range(7):
            # Scan bottom→top so bottom-most detections stay at the bottom
            pieces = [board[row, col]
                      for row in range(5, -1, -1)
                      if board[row, col] != 0]
            # Re-stack from row 5 upward
            for i, color in enumerate(pieces):
                out[5 - i, col] = color
        return out

    def _compute_confidence(self, board: np.ndarray) -> float:
        """
        Score the raw (un-gravity-filtered) board.

        For each column, check that detected pieces form a contiguous stack
        (no gaps between them).  A gap indicates a phantom detection floating
        above a real piece, or a piece detection at the wrong row.  Each gap
        subtracts 0.15.  Piece-count imbalance > 1 subtracts 0.2.

        Why stack-contiguity instead of "must reach row 5":
          The physical board's bottom row of holes may map to camera row 4
          rather than row 5 (row 5 samples the tray/slider area below the
          board).  Real pieces will always appear at row 4 at the lowest, so
          requiring them at row 5 would unfairly penalise every valid state.
          Contiguity (no empty gap between two pieces in the same column) is
          the physically meaningful invariant.
        """
        confidence = 1.0
        for col in range(7):
            piece_rows = [r for r in range(6) if board[r, col] != 0]
            if len(piece_rows) < 2:
                continue
            # Sort top→bottom (ascending row index = higher up in camera frame)
            piece_rows.sort()
            # A gap exists if consecutive piece rows are not adjacent
            for i in range(len(piece_rows) - 1):
                if piece_rows[i + 1] - piece_rows[i] > 1:
                    confidence -= 0.15   # gap in stack = floating piece
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

        # Grid cells — draw at the actual sample radius so the overlay shows
        # exactly what pixels are being classified for each hole.
        cell_spacing  = abs(int(grid_centers[0, 1, 0]) - int(grid_centers[0, 0, 0]))
        sample_radius = max(int(cell_spacing * 0.28), 8)
        for r in range(6):
            for c in range(7):
                cx, cy = int(grid_centers[r, c, 0]), int(grid_centers[r, c, 1])
                cell   = board[r, c]
                if cell == 0:
                    col_fill = (60, 60, 60)
                    col_ring = (160, 160, 160)
                elif cell == 1:
                    col_fill = (0, 0, 200)
                    col_ring = (0, 80, 255)
                else:
                    col_fill = (0, 180, 220)
                    col_ring = (0, 255, 255)
                # Filled inner dot + ring at sample radius
                cv2.circle(dbg, (cx, cy), sample_radius, col_fill, -1)
                cv2.circle(dbg, (cx, cy), sample_radius, col_ring, 1)
                label = "." if cell == 0 else ("R" if cell == 1 else "Y")
                cv2.putText(dbg, label, (cx - 5, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

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
    LOCK_FRAMES     = 4      # Good frames needed to confirm lock
    CANDIDATE_RESET = 20     # Consecutive bad frames to drop candidate entirely
    UNLOCK_FRAMES   = 20     # Consecutive bad locked frames → unlock
    CANDIDATE_GEOM_MAX_SHIFT_FRAC = 0.35  # max median centre shift vs cell spacing
    # Allow perspective-distorted spacing: when the board is at an angle, near
    # rows/columns have larger gaps than far ones.  0.30 tolerates even moderate
    # angles where the near/far gap ratio is up to ~3:1.  Severely extrapolated
    # grids (wrong column count, collapsed clusters) still fail because their
    # gaps deviate much further — outside [0.30, 2.50] of the median.
    GRID_GAP_MIN_FRAC = 0.30
    GRID_GAP_MAX_FRAC = 2.50

    # New piece acceptance: must be seen this many consecutive frames before
    # entering the stable board.  Filters transient YOLO hits from hands or
    # face visible through empty holes (typically 1–2 frames).  A real piece
    # once settled stays visible indefinitely, so 3 frames ≈ 0.2 s at 15 fps
    # adds negligible latency while eliminating false additions.
    ADD_THRESHOLD = 3

    # Piece removal requires this many consecutive absent frames.
    # 15 frames ≈ 1 s at 15 fps — real removals are permanent; brief YOLO
    # misses from shadows or hand occlusion resolve well within 15 frames.
    REMOVE_THRESHOLD = 15

    # Maximum number of gravity-valid new-piece candidates allowed in one frame.
    # A real play adds exactly 1 piece at the bottom of a column (gravity-valid).
    # A face/hand through upper-row holes is NOT gravity-valid and is excluded.
    # Clothing bursts that fill several bottom-row holes still trigger this guard.
    # 1–2 simultaneous detections are tolerated for YOLO jitter; 3+ = contamination.
    MAX_BURST_NEW = 2

    # Post-lock quality verification.  If average confidence over the first
    # VERIFY_WINDOW frames is below VERIFY_MIN_CONF the lock was probably made
    # on poor (e.g. camera still warming up) frames — auto-relock.
    VERIFY_WINDOW   = 20
    # Lowered 0.55→0.35: boards with manually-placed pieces that violate
    # gravity (or a slightly-off grid mapping) can legitimately score 0.45-0.55
    # without being a bad lock.  The old threshold caused infinite relock loops.
    # False-lock prevention now relies on S-min thresholds and piece_threshold.
    VERIFY_MIN_CONF = 0.35

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.detector = BoardDetector(config)
        self._locked_centers: Optional[np.ndarray] = None
        self._locked_contour: Optional[np.ndarray] = None
        self._cand_centers:   Optional[np.ndarray] = None
        self._cand_contour:   Optional[np.ndarray] = None
        self._cand_center_history: List[np.ndarray] = []  # fresh grids during streak
        self._good_count  = 0
        self._bad_streak  = 0
        self._lock_bad    = 0
        self.is_locked    = False
        self._last_board:    Optional[np.ndarray] = None  # for change detection
        self._stable_board:  Optional[np.ndarray] = None  # temporally-smoothed state
        self._absent_count:  np.ndarray = np.zeros((6, 7), dtype=np.int32)
        self._add_count:     np.ndarray = np.zeros((6, 7), dtype=np.int32)
        # True once a cell has been observed as empty since lock.
        # Prevents shirt/background that occupies holes from the very first locked
        # frame from ever being accepted as a piece (it never transitions empty→piece).
        self._cell_was_empty: np.ndarray = np.zeros((6, 7), dtype=bool)
        self._verify_count: int   = 0
        self._verify_sum:   float = 0.0
        self._dump_next: bool = False  # dump all cell HSVs on next locked frame
        # Per-cell empty-board HSV baselines for adaptive mode.
        # Shape (6, 7, 3) — median H, S, V for each cell at lock time.
        self._empty_baselines: Optional[np.ndarray] = None

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
        self._cand_center_history.clear()
        self._good_count  = 0
        self._bad_streak  = 0
        self._lock_bad    = 0
        self.is_locked    = False
        self._last_board      = None
        self._stable_board    = None
        self._absent_count    = np.zeros((6, 7), dtype=np.int32)
        self._add_count       = np.zeros((6, 7), dtype=np.int32)
        self._cell_was_empty  = np.zeros((6, 7), dtype=bool)
        self._verify_count = 0
        self._verify_sum   = 0.0
        self._empty_baselines = None

    def detect(self, image: np.ndarray, debug: bool = False) -> DetectionResult:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if self.is_locked:
            return self._detect_locked(image, hsv, debug)
        return self._detect_unlocked(image, hsv, debug)

    # ── Locked fast path ─────────────────────────────────────────────────────

    def _raw_board(self, image: np.ndarray, hsv: np.ndarray,
                   grid_centers: np.ndarray) -> np.ndarray:
        """Return raw (un-filtered) board classification. Override for YOLO."""
        if self.config.adaptive and self._empty_baselines is not None:
            return self._classify_cells_adaptive(hsv, grid_centers)
        return self.detector._classify_cells(hsv, grid_centers)

    def _detect_locked(self, image: np.ndarray, hsv: np.ndarray,
                        debug: bool) -> DetectionResult:
        board_raw = self._raw_board(image, hsv, self._locked_centers)

        # Track which cells have ever been seen as empty since lock.
        # Used by _temporal_smooth to reject contamination (shirt/face) that
        # occupies holes without ever transitioning through an empty state.
        self._cell_was_empty |= (board_raw == 0)

        confidence = self.detector._compute_confidence(board_raw)

        # ── Full HSV diagnostic dump on first locked frame ───────────────────
        if self._dump_next:
            self._dump_next = False
            if self.detector.config.verbose:
                print("[HSV-DUMP] All 42 cells (first locked frame):")
                cfg = self.detector.config
                red_mask    = (cv2.inRange(hsv, np.array(cfg.red_hsv_low1),  np.array(cfg.red_hsv_high1)) |
                               cv2.inRange(hsv, np.array(cfg.red_hsv_low2),  np.array(cfg.red_hsv_high2)))
                yellow_mask = cv2.inRange(hsv, np.array(cfg.yellow_hsv_low), np.array(cfg.yellow_hsv_high))
                spacing = (self._locked_centers[0, 1, 0] - self._locked_centers[0, 0, 0] if
                           self._locked_centers.shape[1] > 1 else 50)
                radius  = max(4, int(cfg.sample_radius * spacing)) if hasattr(cfg, 'sample_radius') else max(4, int(0.28 * spacing))
                for r in range(6):
                    for c in range(7):
                        cx, cy = int(self._locked_centers[r, c, 0]), int(self._locked_centers[r, c, 1])
                        roi_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                        cv2.circle(roi_mask, (cx, cy), radius, 255, -1)
                        total = cv2.countNonZero(roi_mask)
                        if total == 0:
                            continue
                        roi_pixels = hsv[roi_mask > 0]
                        h_med = int(np.median(roi_pixels[:, 0]))
                        s_med = int(np.median(roi_pixels[:, 1]))
                        v_med = int(np.median(roi_pixels[:, 2]))
                        red_r = cv2.countNonZero(red_mask    & roi_mask) / total
                        yel_r = cv2.countNonZero(yellow_mask & roi_mask) / total
                        det   = ("R" if board_raw[r, c] == 1 else "Y" if board_raw[r, c] == 2 else ".")
                        print(f"  [{r},{c}] HSV=({h_med:3d},{s_med:3d},{v_med:3d})"
                              f"  red={red_r:.2f} yel={yel_r:.2f} -> {det}")
        # Temporal smooth FIRST, then gravity.  If gravity runs on raw
        # board_raw, a brief HSV dropout of a mid-column piece creates a
        # gap → gravity rearranges the whole column → the temporal smoother
        # sees 3+ cell changes and can't recover.  Smoothing first preserves
        # the known piece while it flickers, so gravity never sees the gap.
        board_smooth = self._temporal_smooth(board_raw)
        board        = self.detector._apply_gravity_filter(board_smooth)

        # ── Post-lock quality verification (first VERIFY_WINDOW frames) ─────
        if self._verify_count < self.VERIFY_WINDOW:
            self._verify_count += 1
            self._verify_sum   += confidence
            if self._verify_count == self.VERIFY_WINDOW:
                avg = self._verify_sum / self.VERIFY_WINDOW
                if avg < self.VERIFY_MIN_CONF:
                    print(f"[Board] Poor initial lock (avg conf {avg:.2f}) — auto-relocking")
                    self.unlock()
                    return DetectionResult(
                        board=np.zeros((6, 7), dtype=np.int8),
                        confidence=0.0,
                        errors=["Auto-relocking: poor lock quality"],
                    )

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
            raw_board=board_raw,
        )

    # ── Pre-lock detection ────────────────────────────────────────────────────

    def _detect_unlocked(self, image: np.ndarray, hsv: np.ndarray,
                          debug: bool) -> DetectionResult:
        result = self.detector.detect(image, debug=debug)
        fresh_supports_candidate = False

        # Only update the candidate when the fresh detection is itself verified
        # good — this prevents a slightly-wrong bounding box on the next frame
        # from overwriting a good candidate and tanking confidence.
        # Also require non-fallback (≥8 actual holes detected) so we don't
        # accidentally lock on a random blue kitchen object with no real holes.
        if (result.board_contour is not None
                and result.grid_centers is not None
                and not result.fallback_used):
            fresh_raw  = self.detector._classify_cells(hsv, result.grid_centers)
            fresh_conf = self.detector._compute_confidence(fresh_raw)
            if fresh_conf >= self.LOCK_MIN_CONF and self._grid_spacing_plausible(result.grid_centers):
                if (self._cand_centers is not None and
                        not self._candidate_geometry_matches(result.grid_centers)):
                    print("[Board] Candidate geometry changed too much — restarting lock streak")
                    self._good_count = 0
                    self._bad_streak = 0
                    self._cand_center_history.clear()
                self._cand_centers = result.grid_centers.copy()
                self._cand_contour = result.board_contour.copy()
                # Accumulate fresh grids for median-averaging at lock time.
                # Capped at 20 to bound memory; older frames are less relevant anyway.
                self._cand_center_history.append(result.grid_centers.copy())
                if len(self._cand_center_history) > 20:
                    self._cand_center_history.pop(0)
                fresh_supports_candidate = True
            elif fresh_conf >= self.LOCK_MIN_CONF:
                print("[Board] Rejecting implausible grid spacing before lock")

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

            # Progress toward lock only when the fresh detection agrees with the
            # candidate position. This prevents a stale candidate from reaching
            # LOCK_FRAMES while subsequent fresh fits have gone off the rails.
            is_good = cand_conf >= self.LOCK_MIN_CONF and fresh_supports_candidate

            if is_good:
                self._bad_streak = 0
                self._good_count += 1
                if self._good_count >= self.LOCK_FRAMES:
                    # Lock on the pixel-wise median of all fresh-detection grids
                    # accumulated during this streak.  Averaging multiple frames
                    # smooths out single-frame hole-detection noise and gives a
                    # significantly more accurate grid than any individual frame.
                    if len(self._cand_center_history) >= 2:
                        stacked = np.stack(self._cand_center_history)
                        self._locked_centers = np.round(
                            np.median(stacked, axis=0)
                        ).astype(np.int32)
                    else:
                        self._locked_centers = self._cand_centers.copy()
                    self._cand_center_history.clear()
                    self._locked_contour = self._cand_contour.copy()
                    self.is_locked    = True
                    self._good_count  = 0
                    self._last_board  = cand_board.copy()
                    self._dump_next   = True   # full HSV dump on next frame
                    # Capture per-cell HSV baselines for adaptive mode
                    if self.config.adaptive:
                        self._capture_empty_baselines(hsv)
                    print("[Board] Locked ✓")
                    self._print_board(cand_board)
                else:
                    print(f"[Board] Good frame {self._good_count}/{self.LOCK_FRAMES} "
                          f"— conf {cand_conf:.2f}")
            elif cand_conf < self.LOCK_MIN_CONF:
                # Board is genuinely not visible at the candidate position
                # (confidence dropped, not just a missed blue-frame detection).
                # Only in this case count toward the candidate-drop threshold.
                self._bad_streak += 1
                if self._bad_streak >= self.CANDIDATE_RESET:
                    print(f"[Board] Candidate dropped ({self._bad_streak} bad frames)")
                    self._cand_centers = None
                    self._cand_contour = None
                    self._cand_center_history.clear()
                    self._good_count   = 0
                    self._bad_streak   = 0
            # else: fresh blue detection failed but candidate position still
            # classifies well — board is visible but HSV didn't find the blue
            # frame this frame (glare, lighting shift).  Don't reset the streak
            # or increment bad_streak; just wait for the next frame.

        return result

    def _candidate_geometry_matches(self, new_centers: np.ndarray) -> bool:
        """Reject candidate streaks whose fitted grid jumps between frames."""
        if self._cand_centers is None or new_centers is None:
            return True

        old_pts = self._cand_centers.reshape(-1, 2).astype(np.float32)
        new_pts = new_centers.reshape(-1, 2).astype(np.float32)
        if old_pts.shape != new_pts.shape or old_pts.size == 0:
            return False

        old_spacing = float(abs(self._cand_centers[0, 1, 0] - self._cand_centers[0, 0, 0]))
        new_spacing = float(abs(new_centers[0, 1, 0] - new_centers[0, 0, 0]))
        cell_spacing = max((old_spacing + new_spacing) * 0.5, 1.0)

        dists = np.sqrt(np.sum((old_pts - new_pts) ** 2, axis=1))
        median_shift = float(np.median(dists))
        return median_shift <= cell_spacing * self.CANDIDATE_GEOM_MAX_SHIFT_FRAC

    def _grid_spacing_plausible(self, centers: np.ndarray) -> bool:
        """Reject grids with row/column spacing jumps that indicate bad extrapolation."""
        if centers is None or centers.shape != (6, 7, 2):
            return False

        row_means = centers[:, :, 1].mean(axis=1).astype(np.float32)
        col_means = centers[:, :, 0].mean(axis=0).astype(np.float32)
        row_gaps = np.diff(row_means)
        col_gaps = np.diff(col_means)

        def gaps_ok(gaps: np.ndarray, label: str) -> bool:
            if len(gaps) == 0:
                return False
            med = float(np.median(gaps))
            if med <= 1.0:
                return False
            lo = float(np.min(gaps)) / med
            hi = float(np.max(gaps)) / med
            ok = lo >= self.GRID_GAP_MIN_FRAC and hi <= self.GRID_GAP_MAX_FRAC
            if not ok:
                print(f"[Board] {label} gap ratios: min={lo:.2f} max={hi:.2f} "
                      f"(need [{self.GRID_GAP_MIN_FRAC:.2f}, {self.GRID_GAP_MAX_FRAC:.2f}])")
            return ok

        return gaps_ok(row_gaps, "row") and gaps_ok(col_gaps, "col")

    # ── Helpers ──────────────────────────────────────────────────────────────

    # ── Adaptive colour detection ──────────────────────────────────────────

    def _capture_empty_baselines(self, hsv: np.ndarray):
        """
        Sample median HSV at each grid cell position and store as the
        empty-board baseline.  Called once at lock time.

        Baselines are captured unconditionally for every cell — we do NOT
        skip cells that appear occupied to the fixed-threshold classifier.
        Different cameras (e.g. Orbbec Astra) render empty board holes with
        HSV values that the fixed yellow range flags as pieces.  Capturing
        the true per-cell baseline lets the adaptive ΔS classifier work
        correctly regardless of the camera's color rendering.
        """
        centers = self._locked_centers
        h_img, w_img = hsv.shape[:2]
        baselines = np.full((6, 7, 3), -1.0, dtype=np.float32)
        spacing = abs(int(centers[0, 1, 0]) - int(centers[0, 0, 0]))
        radius  = max(int(spacing * 0.28), 8)

        empty_count = 0
        for r in range(6):
            for c in range(7):
                cx = int(centers[r, c, 0])
                cy = int(centers[r, c, 1])
                if not (0 <= cx < w_img and 0 <= cy < h_img):
                    continue
                roi_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                cv2.circle(roi_mask, (cx, cy), radius, 255, -1)
                roi_pixels = hsv[roi_mask > 0]
                if len(roi_pixels) > 0:
                    baselines[r, c, 0] = np.median(roi_pixels[:, 0])
                    baselines[r, c, 1] = np.median(roi_pixels[:, 1])
                    baselines[r, c, 2] = np.median(roi_pixels[:, 2])
                    empty_count += 1

        self._empty_baselines = baselines
        print(f"[Adaptive] Baselines captured for {empty_count}/42 empty cells")
        if self.config.verbose:
            for r in range(6):
                for c in range(7):
                    h, s, v = baselines[r, c]
                    if h >= 0:
                        print(f"  [{r},{c}] baseline HSV=({h:.0f},{s:.0f},{v:.0f})")
                    else:
                        print(f"  [{r},{c}] (piece at lock — no baseline)")

    def _classify_cells_adaptive(self, hsv: np.ndarray,
                                  grid_centers: np.ndarray) -> np.ndarray:
        """
        Classify cells by comparing current HSV against the empty-board
        baseline.  A cell is considered to have a piece when its saturation
        has increased significantly from the baseline.  Hue distinguishes
        red from yellow.

        Falls back to the standard fixed-threshold method if baselines
        aren't available.
        """
        if self._empty_baselines is None:
            return self.detector._classify_cells(hsv, grid_centers)

        cfg = self.config
        board = np.zeros((6, 7), dtype=np.int8)
        h_img, w_img = hsv.shape[:2]
        spacing = abs(int(grid_centers[0, 1, 0]) - int(grid_centers[0, 0, 0]))
        radius  = max(int(spacing * 0.28), 8)

        min_ds = cfg.adaptive_min_delta_s
        hue_bnd = cfg.adaptive_hue_boundary

        for r in range(6):
            for c in range(7):
                cx = int(grid_centers[r, c, 0])
                cy = int(grid_centers[r, c, 1])
                if not (0 <= cx < w_img and 0 <= cy < h_img):
                    continue
                # Out-of-bounds guard (same as fixed-threshold path)
                if hasattr(self.detector, '_board_bbox') and self.detector._board_bbox is not None:
                    bbx, bby, bbw, bbh = self.detector._board_bbox
                    margin = spacing
                    if not (bbx - margin <= cx <= bbx + bbw + margin and
                            bby - margin <= cy <= bby + bbh + margin):
                        continue

                roi_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                cv2.circle(roi_mask, (cx, cy), radius, 255, -1)
                roi_pixels = hsv[roi_mask > 0]
                if len(roi_pixels) == 0:
                    continue

                h_med = float(np.median(roi_pixels[:, 0]))
                s_med = float(np.median(roi_pixels[:, 1]))
                v_med = float(np.median(roi_pixels[:, 2]))

                base_h = self._empty_baselines[r, c, 0]
                base_s = self._empty_baselines[r, c, 1]
                base_v = self._empty_baselines[r, c, 2]

                # Sentinel baseline (-1) → no empty reference for this cell.
                # Fall back to fixed-threshold mask classification.
                if base_s < 0:
                    cfg_f = self.config
                    roi_mask2 = np.zeros(hsv.shape[:2], dtype=np.uint8)
                    cv2.circle(roi_mask2, (cx, cy), radius, 255, -1)
                    total = cv2.countNonZero(roi_mask2)
                    if total > 0:
                        red_mask = (
                            cv2.inRange(hsv, np.array(cfg_f.red_hsv_low1), np.array(cfg_f.red_hsv_high1)) |
                            cv2.inRange(hsv, np.array(cfg_f.red_hsv_low2), np.array(cfg_f.red_hsv_high2))
                        )
                        yel_mask = cv2.inRange(hsv, np.array(cfg_f.yellow_hsv_low), np.array(cfg_f.yellow_hsv_high))
                        red_r = cv2.countNonZero(red_mask & roi_mask2) / total
                        yel_r = cv2.countNonZero(yel_mask & roi_mask2) / total
                        if red_r > cfg_f.piece_threshold and red_r > yel_r:
                            board[r, c] = 1
                        elif yel_r > cfg_f.piece_threshold and yel_r > red_r:
                            board[r, c] = 2
                    continue

                delta_s = s_med - base_s
                delta_v = v_med - base_v

                # A piece produces a substantial saturation increase from
                # the (typically low-saturation) background seen through the
                # empty hole.  The value channel also shifts but is less
                # reliable.
                #
                # Additionally check that current S is at least somewhat
                # saturated (>60) to avoid classifying desaturated noise.
                is_piece = delta_s >= min_ds and s_med > 60

                if is_piece:
                    # Distinguish red vs yellow by hue.
                    # Red:    H < boundary OR H > (180 - boundary)  (wraps at 0/180)
                    # Yellow: boundary <= H <= yellow_hue_max  (real yellows ~H=20-50)
                    # Anything else (H=60-140: green/cyan/blue board colour) → empty.
                    yel_max = cfg.adaptive_yellow_hue_max
                    if h_med < hue_bnd or h_med > (180 - hue_bnd):
                        if s_med >= cfg.adaptive_red_min_s and v_med >= cfg.adaptive_red_min_v:
                            board[r, c] = 1   # red
                    elif h_med <= yel_max:
                        if s_med >= cfg.adaptive_yellow_min_s and v_med >= cfg.adaptive_yellow_min_v:
                            board[r, c] = 2   # yellow
                    # else: hue is board-blue or other non-piece colour → leave as empty

                if cfg.verbose:
                    detected = "R" if board[r, c] == 1 else "Y" if board[r, c] == 2 else "."
                    notable = is_piece or delta_s > 10
                    if notable:
                        print(f"  [cell {r},{c}] H={h_med:.0f} S={s_med:.0f} V={v_med:.0f}"
                              f"  ΔS={delta_s:+.0f} ΔV={delta_v:+.0f}"
                              f"  base=({base_h:.0f},{base_s:.0f},{base_v:.0f})"
                              f" -> {detected}")

        return board

    # ── Temporal smoothing ───────────────────────────────────────────────────

    def _temporal_smooth(self, board: np.ndarray) -> np.ndarray:
        """
        Debounce both piece additions and removals.

        Addition: a new piece must appear in ADD_THRESHOLD consecutive frames
        before entering the stable board.  This blocks 1–2 frame YOLO hits
        from hands or skin visible through empty holes.

        Removal: a piece is only removed after REMOVE_THRESHOLD consecutive
        absent frames, filtering shadows and brief YOLO misses.
        """
        if self._stable_board is None:
            # Start from zeros — not board.copy() — so ADD_THRESHOLD applies
            # to every piece from the very first frame, including any shirt or
            # background that is already filling holes at lock time.
            self._stable_board = np.zeros((6, 7), dtype=np.int8)
            self._absent_count = np.zeros((6, 7), dtype=np.int32)
            self._add_count    = np.zeros((6, 7), dtype=np.int32)
            return self._stable_board.copy()

        # Burst detection — only count gravity-valid new candidates.
        # "Gravity-valid" = lowest empty slot in its column (all rows below
        # are already occupied).  A real piece always satisfies this; a face
        # or hand visible through upper-row holes does NOT, so it is excluded
        # from the count and cannot trigger a false burst.  A shirt filling
        # several bottom-row holes on an empty board IS gravity-valid and will
        # still fire the guard.  When burst fires, skip all additions this frame
        # without touching _add_count so contamination doesn't accumulate.
        def _gravity_valid(r, c):
            for row_below in range(r + 1, 6):
                if self._stable_board[row_below, c] == 0:
                    return False
            return True

        new_candidates = [
            (r, c)
            for r in range(6) for c in range(7)
            if self._stable_board[r, c] == 0
            and board[r, c] != 0
            and self._cell_was_empty[r, c]
            and _gravity_valid(r, c)
        ]
        burst = len(new_candidates) > self.MAX_BURST_NEW

        for r in range(6):
            for c in range(7):
                detected = board[r, c]
                stable   = self._stable_board[r, c]

                if stable == 0:
                    # Cell is currently empty in stable state
                    if detected != 0:
                        if not self._cell_was_empty[r, c]:
                            # Never seen as empty since lock — persistent background
                            # contamination (shirt at lock time). Ignore entirely.
                            pass
                        elif burst:
                            # Too many simultaneous new detections — contamination
                            # burst. Don't increment; wait for a clean frame.
                            pass
                        else:
                            # Seen as empty before, single new detection: real candidate.
                            # Gravity constraint: only accept if all cells below in this
                            # column are already occupied in the stable board.  This
                            # prevents skin/hand contamination from cascading up a column
                            # (each newly-committed false piece would otherwise make the
                            # row above gravity-valid, chaining false positives upward).
                            gravity_ok = all(
                                self._stable_board[row_below, c] != 0
                                for row_below in range(r + 1, 6)
                            )
                            if gravity_ok:
                                self._add_count[r, c] += 1
                                if self._add_count[r, c] >= self.ADD_THRESHOLD:
                                    self._stable_board[r, c] = detected
                                    self._add_count[r, c] = 0
                                    self._absent_count[r, c] = 0
                            else:
                                # Not gravity-valid — reset counter so transient hits
                                # in upper rows don't accumulate across frames.
                                self._add_count[r, c] = 0
                    else:
                        # Still empty — reset add counter
                        self._add_count[r, c] = 0
                else:
                    # Cell has a piece in stable state
                    if detected != 0:
                        # Piece still visible (possibly different colour — accept update)
                        self._stable_board[r, c] = detected
                        self._absent_count[r, c] = 0
                        self._add_count[r, c] = 0
                    else:
                        # Piece not seen this frame
                        self._absent_count[r, c] += 1
                        if self._absent_count[r, c] >= self.REMOVE_THRESHOLD:
                            self._stable_board[r, c] = 0
                            self._absent_count[r, c] = 0

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


def board_confidence(board: np.ndarray) -> float:
    """Compute detection confidence from a board array (gravity-filtered or merged).

    Same formula as BoardDetector._compute_confidence: penalises column gaps and
    piece-count imbalance.  Exposed as a module-level function so callers can
    re-score the merged/gravity-filtered board rather than the raw detection.
    """
    confidence = 1.0
    for col in range(7):
        piece_rows = [r for r in range(6) if board[r, col] != 0]
        if len(piece_rows) < 2:
            continue
        piece_rows.sort()
        for i in range(len(piece_rows) - 1):
            if piece_rows[i + 1] - piece_rows[i] > 1:
                confidence -= 0.15
    red    = int(np.sum(board == 1))
    yellow = int(np.sum(board == 2))
    if abs(red - yellow) > 1:
        confidence -= 0.2
    return max(0.0, min(1.0, confidence))


# ─────────────────────────────────────────────────────────────────────────────
# YOLO-enhanced detector
# ─────────────────────────────────────────────────────────────────────────────

class YOLOEnhancedBoardDetector(LockedBoardDetector):
    """
    Hybrid board detector: OpenCV/HSV for board location + grid fitting,
    YOLO for piece classification.

    Board-frame detection, perspective warp, hole detection, and grid
    fitting all use the proven OpenCV/HSV pipeline from LockedBoardDetector.
    Once the grid is locked, YOLO classifies each cell as red / yellow / empty
    instead of HSV thresholding — making piece detection robust to skin tones,
    clothing colours, and lighting changes that defeat fixed HSV ranges.

    Fusion strategy (per cell, each locked frame):
      1. YOLO confidence ≥ min_piece_conf   → trust YOLO (primary)
      2. YOLO saw nothing + HSV adaptive fires → trust HSV (fallback for
         partially-occluded or low-contrast pieces YOLO might miss)
      3. Both silent                          → empty

    Drop-in replacement for LockedBoardDetector.  Defaults to loading
    ``yolo-detect/detect/train/weights/best.pt`` relative to this file.
    """

    # Class indices in the fine-tuned 2-class model (best_v2.pt)
    # names: ['red piece', 'yellow piece']  (from data.yaml)
    YOLO_CLASS_RED    = 0
    YOLO_CLASS_YELLOW = 1

    # Default model: prefer fine-tuned v2, fall back to original
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_MODEL_PATH = os.path.join(
        _BASE_DIR, "yolo-detect", "detect", "train", "weights", "best_v2.pt",
    )
    _FALLBACK_MODEL_PATH = os.path.join(
        _BASE_DIR, "yolo-detect", "detect", "train", "weights", "best.pt",
    )

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[DetectionConfig] = None,
        min_piece_conf: float = 0.45,
    ):
        super().__init__(config)
        self._yolo_model = None
        self._min_piece_conf = min_piece_conf
        if model_path is not None:
            path = model_path
        elif os.path.exists(self.DEFAULT_MODEL_PATH):
            path = self.DEFAULT_MODEL_PATH
        else:
            path = self._FALLBACK_MODEL_PATH
        if os.path.exists(path):
            self._load_yolo(path)
        else:
            print(f"[YOLOEnhanced] Model not found at {path} — HSV-only mode")

    def _load_yolo(self, path: str) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
            self._yolo_model = YOLO(path)
            print(f"[YOLOEnhanced] Loaded YOLO model: {path}")
        except ImportError:
            print("[YOLOEnhanced] ultralytics not installed — HSV-only mode")
        except Exception as exc:
            print(f"[YOLOEnhanced] Model load error ({exc}) — HSV-only mode")

    # ── Override: classification step ────────────────────────────────────────

    def _raw_board(self, image: np.ndarray, hsv: np.ndarray,
                   grid_centers: np.ndarray) -> np.ndarray:
        """
        YOLO-only classification: fine-tuned model is the sole ground truth.

        YOLO conf ≥ min_piece_conf → red or yellow piece.
        YOLO silent or below threshold → empty.

        Temporal smoothing (REMOVE_THRESHOLD) in the parent class absorbs any
        single-frame misses without destabilising the stable board.
        """
        if self._yolo_model is None:
            return super()._raw_board(image, hsv, grid_centers)

        yolo_board, yolo_confs = self._classify_with_yolo(image, grid_centers)

        board = np.zeros((6, 7), dtype=np.int8)
        for r in range(6):
            for c in range(7):
                yc = yolo_confs[r, c]
                yp = yolo_board[r, c]
                if yp != 0 and yc >= self._min_piece_conf:
                    board[r, c] = yp

        if self.config.verbose:
            for r in range(6):
                for c in range(7):
                    if board[r, c] != 0 or yolo_board[r, c] != 0:
                        yolo_sym = "." if yolo_board[r, c] == 0 else ("R" if yolo_board[r, c] == self.YOLO_CLASS_RED else "Y")
                        out_sym  = "." if board[r, c]       == 0 else ("R" if board[r, c]       == 1 else "Y")
                        print(f"  [YOLOEnh {r},{c}] yolo={yolo_sym}({yolo_confs[r,c]:.2f}) -> {out_sym}")

        return board

    def _classify_with_yolo(
        self, image: np.ndarray, grid_centers: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run YOLO inference and snap each detection to the nearest grid cell.

        Returns:
            board  — (6, 7) int8   0=empty 1=red 2=yellow (highest-conf hit)
            confs  — (6, 7) float32  confidence of the winning detection per cell
        """
        board = np.zeros((6, 7), dtype=np.int8)
        confs = np.zeros((6, 7), dtype=np.float32)

        results = self._yolo_model.predict(image, imgsz=640, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return board, confs

        # Snap radius: detections whose centre is more than half a cell away
        # from every grid centre are treated as background noise.
        cell_spacing = float(abs(
            int(grid_centers[0, 1, 0]) - int(grid_centers[0, 0, 0])
        ))
        max_snap = cell_spacing * 0.55
        # Size guard: a real game piece fits inside roughly one cell.
        # Anything much larger (face, hand, torso) is rejected before snapping.
        max_piece_diameter = cell_spacing * 1.5

        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if cls not in (self.YOLO_CLASS_RED, self.YOLO_CLASS_YELLOW):
                continue  # ignore empty-class detections and other classes
            if conf < self._min_piece_conf * 0.5:
                continue  # cheap pre-filter before distance calculation

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bw = x2 - x1
            bh = y2 - y1
            if bw > max_piece_diameter or bh > max_piece_diameter:
                continue  # bounding box too large to be a game piece (face/hand/body)
            bx = (x1 + x2) * 0.5
            by = (y1 + y2) * 0.5

            best_r, best_c, best_d = -1, -1, float("inf")
            for r in range(6):
                for c in range(7):
                    gx = float(grid_centers[r, c, 0])
                    gy = float(grid_centers[r, c, 1])
                    d  = ((bx - gx) ** 2 + (by - gy) ** 2) ** 0.5
                    if d < best_d:
                        best_d = d
                        best_r, best_c = r, c

            if best_r < 0 or best_d > max_snap:
                continue  # too far from any cell

            piece = 1 if cls == self.YOLO_CLASS_RED else 2
            if conf > confs[best_r, best_c]:
                board[best_r, best_c] = piece
                confs[best_r, best_c] = conf

        return board, confs
