// =============================================================
//  connect_four_piece_tray.scad
//  Parametric piece-supply tray for Connect Four robot arm
//
//  QUICK START
//  -----------
//  1. Open in OpenSCAD. Press F5 (preview) — check the echoed
//     footprint in the console confirms it fits your print bed.
//  2. Press F6 (render) then File → Export → Export as STL.
//  3. Per color: print 2× this tray (red filament / yellow filament).
//     Each tray holds 12 pockets; 21 of the 24 total are used.
//     The 3 unused pockets (top-right corner of Tray B) can be left
//     empty or marked with an engraved X if you add one in the slicer.
//
//  FLASHFORGE ADVENTURER 3 (150 × 150 × 150 mm bed)
//  --------------------------------------------------
//  At 32 mm pieces, one tray = 111.5 × 147.5 × 7 mm — single-print,
//  no snap-together required. Fits with room to spare.
//
//  LAYOUT ON THE BENCH (per color)
//  --------------------------------
//  Place Tray A and Tray B end-to-end (in Y) within the SO-101's
//  reach envelope (≤ 429 mm horizontal from arm base).
//  Bolt or clamp both trays to the bench — consistent XY position
//  is critical for the arm's pre-recorded pick positions.
//
//  POCKET COUNT
//  ------------
//  3 cols × 4 rows = 12 pockets per tray
//  2 trays per color = 24 pockets total
//  21 used  (one full game's worth)
//  3 spare  (leave empty; define only 21 pickup poses in software)
//
//  ROBOT PICKUP NOTES
//  ------------------
//  - Pocket depth 4 mm → disc protrudes ~3 mm above rim for gripper.
//  - Chamfer at pocket opening self-centers the disc on return/reload.
//  - The arm should always approach from directly above (−Z) and grip
//    the disc rim with the parallel jaw fingers on opposite sides.
//  - Pickup XY for each pocket is fixed; record poses once per color.
//
//  PIECE STACKING NOTE
//  -------------------
//  Pieces have interlocking teeth and can stack vertically, but that
//  design is deferred to v2. Vertical stacking requires either (a) Z
//  tracking as pieces are consumed (tricky with loose tooth nesting)
//  or (b) a spring-loaded magazine with a fixed pickup height. Flat
//  pockets with a known Z per pocket are simpler and more reliable
//  for v1 teleoperated demo recording.
// =============================================================

/* ── USER PARAMETERS ──────────────────────────────────────── */

// Measured disc diameter (calipers). Update if your set differs.
piece_diameter   = 32.0;   // mm  ← measured 2026-05-01

// Pocket geometry
pocket_clearance = 1.5;    // total diametric clearance (0.75 mm/side)
pocket_depth     = 4.0;    // pocket depth: disc protrudes ~3 mm for gripper
chamfer_h        = 2.0;    // height of self-centering flare at pocket lip
chamfer_angle    = 20;     // degrees from vertical (wider = more forgiving)

// Grid layout: 3 cols × 4 rows = 12 pockets per tray
// With 32 mm pieces, 4 rows fits within the 150 mm bed (body_l ≈ 147.5 mm).
cols     = 3;
rows     = 4;
wall_gap = 4.0;            // gap between adjacent pocket edges

// Shell
outer_wall = 3.0;          // outer perimeter wall thickness
base_t     = 3.0;          // floor thickness below pockets

/* ── DERIVED — do not edit below ─────────────────────────────*/
$fn = 64;

pocket_d = piece_diameter + pocket_clearance;   // pocket internal diameter
pitch    = piece_diameter + wall_gap;            // pocket center-to-center
tray_h   = base_t + pocket_depth;               // total tray height

body_w = 2 * outer_wall + pocket_d + (cols - 1) * pitch;
body_l = 2 * outer_wall + pocket_d + (rows - 1) * pitch;

echo(str("═══ Tray footprint: ", body_w, " × ", body_l, " × ", tray_h,
         " mm  (bed limit: 150 × 150 mm) ═══"));
echo(str("    Pocket diameter: ", pocket_d, " mm   pitch: ", pitch, " mm"));
echo(str("    Pockets per tray: ", cols * rows,
         "   Total (2 trays): ", cols * rows * 2,
         "   Used: 21   Spare: ", cols * rows * 2 - 21));

/* ── MODULES ──────────────────────────────────────────────── */

// Single pocket well with chamfer (self-centering flare) at opening.
module pocket() {
    // Cylindrical well
    cylinder(h = pocket_depth + 0.01, d = pocket_d);
    // Chamfer cone: widens from pocket_d at depth chamfer_h to a wider
    // opening at the tray surface — guides gripper fingers onto disc rim.
    translate([0, 0, pocket_depth - chamfer_h])
        cylinder(
            h  = chamfer_h + 0.01,
            d1 = pocket_d,
            d2 = pocket_d + 2 * chamfer_h * tan(chamfer_angle)
        );
}

// Full pocket grid, origin at tray body corner.
module pocket_grid() {
    first_cx = outer_wall + pocket_d / 2;
    first_cy = outer_wall + pocket_d / 2;
    for (r = [0 : rows - 1], c = [0 : cols - 1]) {
        translate([
            first_cx + c * pitch,
            first_cy + r * pitch,
            base_t
        ]) pocket();
    }
}

/* ── TRAY ─────────────────────────────────────────────────── */
module tray() {
    difference() {
        cube([body_w, body_l, tray_h]);
        pocket_grid();
    }
}

/* ── RENDER ───────────────────────────────────────────────── */
tray();
