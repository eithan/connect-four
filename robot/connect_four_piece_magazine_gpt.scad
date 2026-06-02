// ==============================
// Cylindrical Dispenser WITH Base
// ==============================

// ---- Parameters ----
disc_diameter = 32;
disc_thickness = 8.5;

clearance = 0.4;
wall = 3;

stack_count = 3;

// Base
base_diameter = 90;
base_thickness = 6;

// Retention
lip_height = 3;
lip_gap_angle = 80;

// ==============================
// Derived
// ==============================

inner_d = disc_diameter + 2*clearance;
outer_d = inner_d + 2*wall;

tube_height = stack_count * (disc_thickness + 0.3) + 20;

// ==============================
// Base
// ==============================

module base() {
    difference() {
        cylinder(d=base_diameter, h=base_thickness, $fn=100);

        // Optional mounting holes
        for (a = [0:90:270]) {
            rotate([0,0,a])
                translate([base_diameter/2 - 10, 0, 0])
                    cylinder(d=4, h=base_thickness + 1, $fn=30);
        }
    }
}

// ==============================
// Tube
// ==============================

module tube() {
    difference() {

        // Outer
        cylinder(d=outer_d, h=tube_height, $fn=100);

        // Inner hollow
        translate([0,0,wall])
            cylinder(d=inner_d, h=tube_height, $fn=100);

        // Front pickup opening
        translate([-outer_d/2 -1, -outer_d/2, 0])
            cube([outer_d+2, outer_d, disc_thickness + 2]);
    }
}

// ==============================
// Retention Ring
// ==============================

module retention_ring() {
    difference() {
        cylinder(d=inner_d, h=lip_height, $fn=100);

        translate([0,0,-1])
            cylinder(d=inner_d - 4, h=lip_height + 2, $fn=100);

        rotate_extrude(angle=lip_gap_angle)
            translate([inner_d/2,0,0])
                square([2, lip_height + 1]);
    }
}

// ==============================
// Assembly
// ==============================

union() {

    // Base
    base();

    // Tube on top of base
    translate([0,0,base_thickness])
        tube();

    // Retention ring
    translate([0,0,base_thickness + disc_thickness])
        retention_ring();

    // Backstop
    translate([-1, inner_d/2 - 2, base_thickness])
        cube([2, 4, disc_thickness + 2]);
}