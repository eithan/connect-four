// =============================================================
//  connect_four_piece_chute.scad
//  Vertical coin-tube piece dispenser for Connect Four robot arm
//
//  HOW IT WORKS
//  ------------
//  1. Pieces are dropped in from the open top and stack flat.
//  2. The bottom-front face of the tube is cut away at floor level,
//     exposing the bottom piece on its left and right edges.
//  3. The robot arm approaches horizontally from the front, slides
//     its gripper fingers in from both sides, pinches the bottom
//     piece's rim (like picking a coin off a table edge), then pulls
//     the piece straight forward and out of the tube.
//  4. Gravity drops the remaining stack down one piece-height.
//     Repeat until empty, then reload from the top.
//
//  PICKUP GEOMETRY (looking from the front)
//  -----------------------------------------
//
//       |         |
//       |    ●    |   ← piece 2 (drops down after piece 1 is taken)
//       |         |
//  ─────┤         ├─  ← top of front opening
//       |   [●]   |   ← piece 1 (bottom), fully exposed at sides
//  ─────┴─────────┴─  ← floor
//
//  → arm comes from the front
//  → fingers enter from ±X (left and right)
//  → gripper closes, pinches piece rim at 9 o'clock and 3 o'clock
//  → arm pulls forward (+Y) → piece slides out
//  → piece 2 drops to floor, ready for next pick
//
//  CAPACITY vs PRINT BED (Flashforge Adventurer 3: 150×150×150 mm)
//  ----------------------------------------------------------------
//  At piece_t = 8.5 mm, max 16 pieces fit in the 150 mm Z height.
//  Print 2 chutes per color. Load up to 16 pieces in each.
//  Total capacity (2×16 = 32) comfortably covers a full game (21).
//
//  PRINT PLAN  —  per color, 4 prints total for both colors
//  ---------------------------------------------------------
//  RENDER = "chute"  → chute.stl   print 2×
//  RENDER = "base"   → base.stl    print 2×
//  Use red filament for red chutes, yellow for yellow.
//  Tube press-fits into base socket; CA glue for permanence.
//
//  BENCH ORIENTATION
//  -----------------
//  The front opening (pickup face) must face the arm. The base has
//  a front notch to mark which side is the pickup face so you always
//  replace the chute in the same orientation after reloading.
// =============================================================

/* ── SELECT WHAT TO RENDER ────────────────────────────────── */

RENDER = "chute";   // "chute" | "base"

/* ── DISC DIMENSIONS (measured) ─────────────────────────────*/

piece_d = 32.0;   // disc diameter (mm)
piece_t = 8.5;    // disc thickness (mm)

/* ── CHUTE PARAMETERS ────────────────────────────────────────*/

n_pieces  = 16;    // pieces per chute (max 16 to fit 150 mm Z bed).
                   // Print 2× per color for full 21-piece game coverage.

wall_t    = 3.0;   // tube wall thickness (mm)
floor_t   = 3.0;   // floor below disc stack (mm)
id_clr    = 1.0;   // radial clearance per side — disc slides freely (mm)
top_clr   = 10.0;  // headroom above top disc for easy loading (mm)

// Front opening (pickup face)
// The entire front half of the tube wall is removed at floor level,
// from floor_t up to floor_t + opening_h. This exposes the bottom
// disc fully on its sides so the gripper can reach in and grip it.
opening_extra = 2.0;   // opening extends this much above the disc top (mm)
                       // gives the gripper fingers a little vertical clearance

// Loading chamfer at top — flared inner rim makes dropping pieces in easy
chamfer_h = 6.0;   // height of loading chamfer (mm)
chamfer_r = 3.0;   // extra inner radius at the very top (mm)

/* ── BASE PARAMETERS ─────────────────────────────────────────*/

base_d      = 84.0;  // base outer diameter (mm) — wide for stability
base_h      = 12.0;  // base total height (mm)
socket_clr  = 0.3;   // radial clearance of socket over tube OD (mm/side)
socket_depth = 8.0;  // how deep tube sits in base socket (mm)
notch_w     = 8.0;   // width of front-face orientation notch on base (mm)
notch_d     = 3.0;   // depth of orientation notch (mm)

/* ── DERIVED — do not edit ───────────────────────────────────*/

$fn = 128;

inner_r  = piece_d / 2 + id_clr;   // tube inner radius
outer_r  = inner_r + wall_t;       // tube outer radius
inner_d  = inner_r * 2;
outer_d  = outer_r * 2;

tube_h   = floor_t + n_pieces * piece_t + top_clr;
opening_h = piece_t + opening_extra;    // front opening height

max_n = floor((150 - floor_t - top_clr) / piece_t);

echo(str("═══════════════════════════════════════════"));
echo(str("  Tube:     OD=", outer_d, "  ID=", inner_d, "  H=", tube_h, " mm"));
echo(str("  Base:     D=",  base_d,  "  H=", base_h, " mm"));
echo(str("  Opening:  W=", inner_d, " (full bore width)  H=", opening_h, " mm"));
echo(str("  Bed fit:  ", (tube_h <= 150)
    ? str("OK  (", 150 - tube_h, " mm to spare)")
    : str("OVER by ", tube_h - 150, " mm — set n_pieces ≤ ", max_n)));
echo(str("═══════════════════════════════════════════"));

/* ── CHUTE ────────────────────────────────────────────────── */

module chute() {
    difference() {
        union() {
            // Main tube cylinder
            cylinder(h = tube_h, r = outer_r);

            // Loading chamfer: flared inner rim at the top so pieces
            // dropped from above don't catch on the edge.
            translate([0, 0, tube_h])
                cylinder(h = chamfer_h, r1 = outer_r, r2 = outer_r + chamfer_r);
        }

        // ── Inner bore (above floor) ───────────────────────────
        translate([0, 0, floor_t])
            cylinder(h = tube_h - floor_t + 0.01, r = inner_r);

        // ── Chamfer inner bore (flares with outer chamfer) ─────
        translate([0, 0, tube_h - 0.01])
            cylinder(h = chamfer_h + 0.02, r1 = inner_r, r2 = inner_r + chamfer_r);

        // ── Front pickup opening ───────────────────────────────
        // Removes the entire front half of the tube (y > 0 side)
        // at disc height. The bottom piece is now fully accessible
        // from the left and right sides for the gripper.
        //
        // Cut spans:
        //   X: full tube outer diameter (-outer_r to +outer_r)
        //   Y: tube center to outer face (0 to outer_r + margin)
        //   Z: top of floor to top of opening (floor_t to floor_t + opening_h)
        translate([-outer_r - 0.01, -0.01, floor_t])
            cube([outer_d + 0.02, outer_r + 0.02, opening_h]);
    }
}

/* ── BASE ─────────────────────────────────────────────────── */

module base() {
    difference() {
        // Wide flat disc — weighted for stability
        cylinder(h = base_h, r = base_d / 2);

        // Socket: tube press-fits here.
        // Oriented so the tube's front opening (+Y) faces outward.
        translate([0, 0, base_h - socket_depth])
            cylinder(h = socket_depth + 0.01, r = outer_r + socket_clr);

        // Orientation notch on the front face (+Y side).
        // Tells you which way the pickup opening faces when
        // replacing the chute on the bench after reloading.
        translate([-notch_w / 2, base_d / 2 - notch_d, -0.01])
            cube([notch_w, notch_d + 0.01, base_h + 0.02]);
    }
}

/* ── RENDER ───────────────────────────────────────────────── */

if (RENDER == "chute") chute();
if (RENDER == "base")  base();
