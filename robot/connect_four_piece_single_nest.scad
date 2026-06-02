// =============================================================
//  connect_four_piece_single_nest.scad
//  Single-pocket disc nest for Connect Four robot arm
//
//  HOW IT WORKS
//  ------------
//  One disc sits flat in the pocket, rim exposed ~3 mm above the
//  surface. The robot arm descends from above (-Z), gripper fingers
//  straddle the rim, close on opposite edges, then lift and rotate
//  the disc vertical for a clean drop into the board column.
//
//  No bolting needed. Place the nest at a marked spot on the bench
//  (tape cross is enough). Rubber bumper feet on the bottom keep it
//  from sliding during the pick. ACT learns the pickup from the
//  consistent-enough position.
//
//  LOADING
//  -------
//  Drop a disc from above — the chamfer self-centers it in the pocket.
//
//  PRINT NOTES
//  -----------
//  Print flat (pocket facing up). No supports needed.
//  PLA is fine. Print one per color or one shared.
//  Footprint: ~50 mm diameter, 8 mm tall — trivial on any bed.
//
//  RUBBER FEET
//  -----------
//  Press standard 8–10 mm self-adhesive bumper feet into the four
//  recesses on the bottom. These prevent sliding during pickup.
//
//  DISC DIMENSIONS: measured 2026-05-01
// =============================================================

/* ── PARAMETERS ───────────────────────────────────────────── */

piece_d          = 32.0;  // disc diameter (mm)
pocket_clearance =  1.5;  // total diametric clearance (0.75 mm/side)
pocket_depth     =  4.0;  // disc sits 4 mm down → ~3 mm rim exposed
chamfer_h        =  2.0;  // self-centering flare height at pocket lip
chamfer_angle    = 20;    // flare angle from vertical (degrees)

outer_wall       =  9.0;  // wall thickness around pocket (no bolts → wider)
base_t           =  4.0;  // floor thickness below pocket

foot_d           =  9.0;  // rubber bumper foot recess diameter (mm)
foot_depth       =  1.2;  // recess depth — just enough to locate the foot
foot_inset       = 12.0;  // foot center distance from body center

/* ── DERIVED ──────────────────────────────────────────────── */

$fn = 96;

pocket_id  = piece_d + pocket_clearance;
body_r     = pocket_id / 2 + outer_wall;
total_h    = base_t + pocket_depth;

protrusion = piece_d - pocket_depth;  // how much disc sticks up above rim

echo(str("═══ Single nest: OD=", body_r*2, " mm  H=", total_h, " mm ═══"));
echo(str("    Pocket ID: ", pocket_id, " mm   depth: ", pocket_depth, " mm"));
echo(str("    Disc protrusion above rim: ", protrusion, " mm"));

/* ── MODULES ──────────────────────────────────────────────── */

module pocket() {
    // Cylindrical well
    cylinder(h = pocket_depth + 0.01, d = pocket_id);
    // Self-centering chamfer flare at the lip
    translate([0, 0, pocket_depth - chamfer_h])
        cylinder(
            h  = chamfer_h + 0.01,
            d1 = pocket_id,
            d2 = pocket_id + 2 * chamfer_h * tan(chamfer_angle)
        );
}

module foot_recesses() {
    for (a = [45, 135, 225, 315])
        rotate([0, 0, a])
            translate([foot_inset, 0, -0.01])
                cylinder(h = foot_depth + 0.01, d = foot_d);
}

/* ── NEST ─────────────────────────────────────────────────── */

difference() {
    // Body: simple cylinder — prints clean, no corners to warp
    cylinder(h = total_h, r = body_r);

    // Pocket (from top)
    translate([0, 0, base_t])
        pocket();

    // Rubber foot recesses (from bottom)
    foot_recesses();
}

// Optional: preview disc in pocket (uncomment for F5 eyeball check)
// %translate([0, 0, base_t - 0.5])
//     cylinder(h = piece_d, d = piece_d, $fn = 96);
