#!/usr/bin/env python3
import math
from pathlib import Path

BOND = 1.44              # Angstrom
NX = 100                 # zigzag direction
NY = 60                  # armchair direction
LZ = 20.0                # box length in the non-periodic z direction, Angstrom
OUT = Path("model.xyz")

ax = math.sqrt(3.0) * BOND
ay = 3.0 * BOND
lx = NX * ax
ly = NY * ay
num_atoms = 4 * NX * NY

basis = (
    (0.0, 0.0, 0.0),
    (0.0, BOND, 0.0),
    (0.5 * ax, 1.5 * BOND, 0.0),
    (0.5 * ax, 2.5 * BOND, 0.0),
)

with OUT.open("w", encoding="utf-8") as f:
    f.write(f"{num_atoms}\n")
    f.write(
        f'pbc="T T F" '
        f'Lattice="{lx:.12f} 0 0 0 {ly:.12f} 0 0 0 {LZ:.12f}" '
        'Properties=species:S:1:pos:R:3\n'
    )
    for iy in range(NY):
        y0 = iy * ay
        for ix in range(NX):
            x0 = ix * ax
            for bx, by, bz in basis:
                f.write(f"C {x0 + bx:.10f} {y0 + by:.10f} {bz:.10f}\n")

print(f"Wrote {OUT}")
print(f"Atoms      : {num_atoms}")
print(f"Box x      : {lx:.6f} A (zigzag)")
print(f"Box y      : {ly:.6f} A (armchair)")
print(f"Aspect Ly/Lx: {ly/lx:.6f}")
