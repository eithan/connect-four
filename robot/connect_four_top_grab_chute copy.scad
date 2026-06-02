// =============================================================
//  connect_four_top_grab_chute.scad  (v3 — cylinder with open sides)
//  Vertical piece dispenser — top-down gripper access
//
//  HOW IT WORKS
//  ------------
//  1. Drop pieces in from the open top — they stack flat (horizontal).
//  2. The arm descends from above (-Z) with fingers open (spread wide
//     in X, outside the tube diameter). No walls at ±X, so fingers
//     have unobstructed vertical travel.
//  3. At disc level, gripper closes: fingers move inward in X and
//     contact the TOP disc's rim at the 9-o'clock and 3-o'clock
//     positions.
//  4. Arm lifts the disc straight up, clears the tube.
//  5. Wrist rotates 90° — disc is now vertical (edge-down).
//  6. Arm moves over the target column, opens gripper → disc falls in.
//  7. Gravity drops the remaining stack down one disc-height. Repeat.
//
//  WHY A CYLINDER WITH WIDE OPENINGS
//  -----------------------------------
//  The ±X openings remove the cylinder wall in the finger-approach
//  zone, giving the gripper unobstructed vertical travel. The
//  remaining arc walls at ±Y self-center the round disc (curved wall
//  matches disc perimeter), which is better than flat walls.
//  The disc cannot fall out: the ±Y arcs constrain it to < 1.5 mm
//  lateral play in X.
//
//  TOP VIEW (cross-section)
//  ------------------------
//
//     finger              finger
//       │                    │
//       ↓                    ↓
//   (open +X side)     (open -X side)
//
//       ╔══════════════════╗
//       ║  arc wall (+Y)   ║
//       ╠══╡           ╞══╣   ← opening_w wide — fingers pass here
//       ║       ●          ║   ← disc, constrained by ±Y arcs
//       ╠══╡           ╞══╣
//       ║  arc wall (-Y)   ║
//       ╚══════════════════╝
//
//  CAPACITY vs PRINT BED (Flashforge Adventurer 3: 150×150×150 mm)
//  ----------------------------------------------------------------
//  At piece_t = 8.5 mm, max 16 pieces fit within 150 mm Z height.
//  Print 2 chutes per color. 2×16 = 32 total, covers a full game.
//
//  PRINT PLAN — per color, 4 prints total
//  ----------------------------------------
//  RENDER = "chute"  → chute.stl   print 2×
//  RENDER = "base"   → base.stl    print 2×
//  Tube press-fits into base socket; CA glue for permanence.
//
//  BENCH ORIENTATION
//  -----------------
//  Orient so the ±X openings align with the gripper's open/close axis.
//  The orientation notch on the base marks the +Y (solid arc) face.
// =============================================================

/* ── SELECT WHAT TO RENDER ────────────────────────────────── */

RENDER = "chute";   // "chute" | "base"

/* ── DISC DIMENSIONS (measured) ─────────────────────────────*/

piece_d = 32.0;   // disc diameter (mm)
piece_t =  8.5;   // disc thickness (mm)

/* ── CHUTE PARAMETERS ────────────────────────────────────────*/

n_pieces   = 2;    // pieces per chute — set to 2 for preview print, 16 for production

wall_t     = 3.0;   // tube wall thickness (mm)
floor_t    = 3.0;   // floor below disc stack (mm)
id_clr     = 1.0;   // radial clearance per side — disc slides freely (mm)
top_clr    = 10.0;  // headroom above top disc for easy loading (mm)

// ±X side opening — cut through the cylinder wall at ±X in this Y band.
// opening_w must be:
//   > finger pad width (~10 mm for SO-101)  so fingers close without hitting arc
//   < disc diameter (32 mm)                 so disc cannot fall out sideways
opening_w  = 18.0;  // opening width in Y (mm). Default 18 mm.

// Loading chamfer at top
chamfer_h  = 6.0;   // height of inner flare at top (mm)
chamfer_r  = 3.0;   // extra inner radius at very top (mm)

/* ── BASE PARAMETERS ─────────────────────────────────────────*/

base_d       = 84.0;
base_h       = 12.0;
socket_clr   =  0.3;
socket_depth =  8.0;
notch_w      =  8.0;
notch_d      =  3.0;

/* ── DERIVED — do not edit ───────────────────────────────────*/

$fn = 128;

inner_r  = piece_d / 2 + id_clr;
outer_r  = inner_r + wall_t;
inner_d  = inner_r * 2;
outer_d  = outer_r * 2;

tube_h   = floor_t + n_pieces * piece_t + top_clr;
max_n    = floor((150 - floor_t - top_clr) / piece_t);

// Disc lateral play in X: how far disc can slide before arc wall stops it.
// At Y = opening_w/2, arc wall inner face is at X = sqrt(inner_r²-(opening_w/2)²)
// and disc edge at Y = opening_w/2 is at X = sqrt((piece_d/2)²-(opening_w/2)²).
disc_play_x = sqrt(inner_r*inner_r - (opening_w/2)*(opening_w/2))
            - sqrt((piece_d/2)*(piece_d/2) - (opening_w/2)*(opening_w/2));

echo(str("═══════════════════════════════════════════"));
echo(str("  Tube:     OD=", outer_d, "  ID=", inner_d, "  H=", tube_h, " mm"));
echo(str("  Opening:  W=", opening_w, " mm in Y  (disc=", piece_d,
         " mm — retained: ", (opening_w < piece_d) ? "YES" : "NO", ")"));
echo(str("  Disc lateral play: ", disc_play_x, " mm  (should be < 2 mm)"));
echo(str("  Bed fit:  ", (tube_h <= 150)
    ? str("OK  (", 150 - tube_h, " mm to spare)")
    : str("OVER by ", tube_h - 150, " mm — set n_pieces ≤ ", max_n)));
echo(str("═══════════════════════════════════════════"));

/* ── CHUTE ────────────────────────────────────────────────── */

module chute() {
    difference() {
        union() {
            // Main cylinder
            cylinder(h = tube_h, r = outer_r);

            // Loading chamfer: flared inner rim at top for easy loading
            translate([0, 0, tube_h])
                cylinder(h = chamfer_h, r1 = outer_r, r2 = outer_r + chamfer_r);
        }

        // ── Inner bore ─────────────────────────────────────────
        translate([0, 0, floor_t])
            cylinder(h = tube_h - floor_t + 0.01, r = inner_r);

        // ── Inner bore chamfer ─────────────────────────────────
        translate([0, 0, tube_h - 0.01])
            cylinder(h = chamfer_h + 0.02, r1 = inner_r, r2 = inner_r + chamfer_r);

        // ── +X side opening ────────────────────────────────────
        // Removes the cylinder wall in the +X half within the Y band.
        // Finger descends at X > outer_r, closes inward to grip disc rim.
        // X extent reaches outer_r + chamfer_r so the chamfer flare is
        // also cleared — without this a sliver of chamfer is left at ±X.
        translate([0, -opening_w / 2, floor_t])
            cube([outer_r + chamfer_r + 1, opening_w, tube_h - floor_t + chamfer_h + 0.02]);

        // ── -X side opening (mirror) ───────────────────────────
        translate([-(outer_r + chamfer_r + 1), -opening_w / 2, floor_t])
            cube([outer_r + chamfer_r + 1, opening_w, tube_h - floor_t + chamfer_h + 0.02]);
    }
}

/* ── BASE ─────────────────────────────────────────────────── */

module base() {
    difference() {
        cylinder(h = base_h, r = base_d / 2);

        // Circular socket — tube press-fits here.
        // Lower floor_t of tube is a full cylinder; socket grips it.
        translate([0, 0, base_h - socket_depth])
            cylinder(h = socket_depth + 0.01, r = outer_r + socket_clr);

        // Orientation notch on +Y face (the solid arc side)
        translate([-notch_w / 2, base_d / 2 - notch_d, -0.01])
            cube([notch_w, notch_d + 0.01, base_h + 0.02]);
    }
}

/* ── RENDER ───────────────────────────────────────────────── */

if (RENDER == "chute") chute();
if (RENDER == "base")  base();
