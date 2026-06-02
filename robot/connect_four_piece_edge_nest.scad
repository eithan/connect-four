// =============================================================
//  connect_four_piece_edge_nest.scad
//  Stage-1 "edge nest" piece holder for the Connect Four robot arm
//
//  WHY THIS EXISTS (vs. the other fixtures in this folder)
//  -------------------------------------------------------
//  tray / chute / magazine all hold the disc FLAT and grip its RIM,
//  which leaves the disc horizontal after pickup -> a 90 deg wrist
//  rotation is then needed to drop it into a vertical board slot
//  (the "wrist rotation direction" question parked in column_mover.py).
//
//  This nest holds ONE disc STANDING ON EDGE. The gripper descends
//  from straight above and clamps the two flat faces across the
//  8.5 mm thickness (the "narrow" grip). The disc comes up already
//  vertical, hanging edge-down, and drops straight into the column.
//  No reorientation.
//
//        gripper jaws close along X (across the 8.5 mm thickness)
//                 │            │
//                 ▼            ▼
//              ┌────┐  ●●●  ┌────┐    ● = exposed upper faces (grip here)
//              │    │ ●███● │    │
//   side walls │    │ █████ │    │    cradle only the LOWER ~12 mm so the
//   (cut low)  └────┘ █████ └────┘    jaws clear the walls
//   ───────────────────────────────  floor (disc rests on its edge here)
//
//  STAGE 1 = THIS FILE: a single nest, hand-reloaded between episodes.
//  Fixed pickup pose -> ideal for recording the first ACT dataset.
//  STAGE 2 (later): bolt an inclined feed channel onto the back so the
//  next disc auto-feeds. Parameters here are named to make that easy.
//
//  HOW TO USE
//  ----------
//  1. Open in OpenSCAD. F5 (preview) — read the echoed footprint,
//     grip-exposure, and pickup-pose numbers in the console.
//  2. F6 (render) -> File > Export > Export as STL.
//  3. Print in PLA. Print "as drawn" (open side up) — no supports
//     needed; the chamfers are self-supporting at 45 deg.
//  4. Bolt to the bench through the 4 mounting holes. A fixed XY/Z
//     pose is critical: you teach the pickup pose ONCE, then every
//     recorded episode reuses it.
//
//  ORIENTATION ON THE BENCH
//  ------------------------
//  The gripper closes along X. Set the nest so X is PERPENDICULAR to
//  the Connect Four board face — then the lifted disc's plane is
//  already parallel to the board slot and no wrist roll is required
//  between pick and place.
//
//  DISC DIMENSIONS: measured 2026-05-01 (see connect_four_piece_tray.scad)
// =============================================================

/* ── USER PARAMETERS ──────────────────────────────────────── */

// Measured disc (calipers). Update if your set differs.
piece_d        = 32.0;   // disc diameter  (mm)
piece_t        = 8.5;    // disc thickness (mm)  <- this is the gripped dimension

// Slot fit
slot_clearance = 1.0;    // TOTAL extra across thickness (0.5 mm/side).
                         // Tight on purpose: disc must stand vertical and
                         // land in a repeatable pose. Loosen to ~1.5 if
                         // your printer runs tight and the disc binds.
diam_clearance = 1.5;    // TOTAL extra along the diameter (front/back),
                         // so the disc drops in without jamming.

// Cradle / walls
cradle_h       = 12.0;   // how high the walls hug the disc (mm).
                         // MUST stay well below piece_d so the upper faces
                         // are exposed for the jaws. See grip-exposure echo.
wall_t         = 3.0;    // wall thickness (mm)
floor_t        = 3.0;    // floor thickness below the disc (mm)

// Self-centering chamfer on the top inner edges (funnels the disc on
// reload and funnels the jaws on approach). Kept <= wall_t so a solid
// lip remains at the top outer edge.
chamfer        = 2.0;    // 45 deg chamfer leg (mm)

// Mounting (bolt-down)
mount_hole_d   = 3.4;    // clearance for M3 (mm)
mount_inset    = 6.0;    // hole center inset from the base edge (mm)
base_margin    = 9.0;    // base plate overhang around the cradle (mm)
                         // -> gives room for the mounting holes

// Gripper reality-check (for the echo only; not geometry).
// SO-101 stock parallel jaw: set your measured numbers to get a warning
// if the design asks for more than the gripper can do.
jaw_max_open   = 40.0;   // max jaw opening (mm)  <- MEASURE yours
finger_len     = 20.0;   // finger pad length (mm) <- MEASURE yours

/* ── DERIVED — do not edit below ─────────────────────────────*/
$fn = 96;

slot_w   = piece_t + slot_clearance;          // inner channel width (X)
inner_l  = piece_d + diam_clearance;          // inner length front/back (Y)

cradle_outer_w = slot_w + 2 * wall_t;         // X outer of the cradle box
cradle_outer_l = inner_l + 2 * wall_t;        // Y outer of the cradle box
wall_top_z     = floor_t + cradle_h;          // top of the cradle walls

base_w = cradle_outer_w + 2 * base_margin;    // X of base plate
base_l = cradle_outer_l + 2 * base_margin;    // Y of base plate

disc_top_z   = floor_t + piece_d;             // top of the disc above table
exposed_h    = disc_top_z - wall_top_z;       // exposed face height for jaws
grip_center_z = (wall_top_z + disc_top_z) / 2; // suggested jaw-center height

/* ── SANITY ECHOES ───────────────────────────────────────────*/
echo(str("═══ Edge-nest footprint: ", base_w, " × ", base_l,
         " × ", wall_top_z, " mm ═══"));
echo(str("    Slot (X) inner width: ", slot_w,
         " mm   Cradle length (Y): ", inner_l, " mm"));
echo(str("    Disc top: ", disc_top_z, " mm   Wall top: ", wall_top_z,
         " mm   -> exposed grip height: ", exposed_h, " mm"));
echo(str("    Suggested jaw-center Z (above table): ", grip_center_z, " mm"));
echo(str("    Grip span needed: ", piece_t, " mm   (jaw max open ",
         jaw_max_open, " mm -> ",
         (piece_t < jaw_max_open) ? "OK" : "TOO WIDE", ")"));
echo(str("    Exposed grip height ", exposed_h, " mm vs finger length ",
         finger_len, " mm -> ",
         (exposed_h >= 8) ? "OK (>=8 mm of face to grip)"
                          : "LOW — lower cradle_h for more grip"));

/* ── MODULES ──────────────────────────────────────────────── */

// Self-centering funnel: flares the slot opening outward by `chamfer`
// on all four sides over the top `chamfer` mm of wall. Built as the hull
// of a narrow rectangle (slot cross-section, lower) and a wider rectangle
// (slot + 2*chamfer, at the lip) -> a clean 45 deg funnel on every edge.
module slot_funnel() {
    hull() {
        translate([0, 0, wall_top_z - chamfer])
            linear_extrude(0.01)
                square([slot_w, inner_l], center = true);
        translate([0, 0, wall_top_z + 0.01])
            linear_extrude(0.01)
                square([slot_w + 2*chamfer, inner_l + 2*chamfer], center = true);
    }
}

module mount_holes() {
    for (sx = [-1, 1], sy = [-1, 1])
        translate([sx * (base_w/2 - mount_inset),
                   sy * (base_l/2 - mount_inset),
                   -0.01])
            cylinder(h = floor_t + 0.02, d = mount_hole_d);
}

/* ── NEST ─────────────────────────────────────────────────── */
module edge_nest() {
    difference() {
        union() {
            // Base plate
            translate([-base_w/2, -base_l/2, 0])
                cube([base_w, base_l, floor_t]);
            // Cradle box (walls), centered on origin in XY
            translate([-cradle_outer_w/2, -cradle_outer_l/2, 0])
                cube([cradle_outer_w, cradle_outer_l, wall_top_z]);
        }

        // Hollow the standing-disc slot (open top), from just above the
        // floor up through the top so the disc rests on the floor.
        translate([-slot_w/2, -inner_l/2, floor_t])
            cube([slot_w, inner_l, wall_top_z]);  // through the top

        // Self-centering funnel on the four top inner edges
        slot_funnel();

        // Mounting holes
        mount_holes();
    }
}

/* ── RENDER ───────────────────────────────────────────────── */
edge_nest();

// Optional: drop a translucent disc in place to eyeball the grip.
// Uncomment to preview (F5). The disc stands on edge in the slot.
//
// %translate([0, 0, floor_t + piece_d/2])
//     rotate([0, 90, 0])
//         cylinder(h = piece_t, d = piece_d, center = true, $fn = 96);
