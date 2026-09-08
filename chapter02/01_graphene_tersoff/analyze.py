#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

N_ATOMS = 24000
DT_FS = 1.0
THERMO_INTERVAL = 1000
THERMO_FILE = Path("thermo.out")
FINAL_FILE = Path("final.xyz")


def read_thermo(path: Path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 3:
        raise RuntimeError(f"Expected at least 3 columns in {path}, got {data.shape[1]}")
    time_ps = (np.arange(1, len(data) + 1) * THERMO_INTERVAL * DT_FS) / 1000.0
    temperature = data[:, 0]
    kinetic = data[:, 1]
    potential = data[:, 2]
    total_per_atom = (kinetic + potential) / N_ATOMS
    potential_per_atom = potential / N_ATOMS
    return time_ps, temperature, total_per_atom, potential_per_atom


def analyze_thermo():
    time_ps, temperature, total_e, potential_e = read_thermo(THERMO_FILE)
    start = len(temperature) // 2

    print("Thermodynamic summary (second half of the run)")
    print(f"  Mean T          = {temperature[start:].mean():.3f} K")
    print(f"  Std(T)          = {temperature[start:].std(ddof=1):.3f} K")
    print(f"  Mean E/N        = {total_e[start:].mean():.8f} eV/atom")
    print(f"  Mean U/N        = {potential_e[start:].mean():.8f} eV/atom")

    fig, ax = plt.subplots()
    ax.plot(time_ps, temperature)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Temperature (K)")
    fig.tight_layout()
    fig.savefig("temperature.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(time_ps, potential_e, label="Potential energy")
    ax.plot(time_ps, total_e, label="Total energy")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Energy (eV/atom)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("energy.png", dpi=200)
    plt.close(fig)


def read_final_positions(path: Path):
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
        if not first:
            raise RuntimeError(f"Empty file: {path}")
        n = int(first)
    if n != N_ATOMS:
        raise RuntimeError(f"Expected {N_ATOMS} atoms, found {n}")
    xyz = np.loadtxt(path, skiprows=2, usecols=(1, 2, 3))
    if xyz.shape != (n, 3):
        raise RuntimeError(f"Expected {(n, 3)} coordinate array, got {xyz.shape}")
    return xyz


def analyze_ripple():
    xyz = read_final_positions(FINAL_FILE)
    x, y, z = xyz.T
    h = z - z.mean()
    rms = np.sqrt(np.mean(h * h))
    p2p = h.max() - h.min()
    p05, p95 = np.percentile(h, [5, 95])

    print("Ripple summary from final.xyz")
    print(f"  Mean z          = {z.mean():.6f} A")
    print(f"  RMS height      = {rms:.6f} A")
    print(f"  Peak-to-peak    = {p2p:.6f} A")
    print(f"  5-95% range     = {p95-p05:.6f} A")

    bins = (250, 250)
    weighted, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=h)
    counts, _, _ = np.histogram2d(x, y, bins=(xedges, yedges))
    height = np.divide(weighted, counts, out=np.full_like(weighted, np.nan), where=counts > 0)

    fig, ax = plt.subplots()
    image = ax.imshow(
        height.T,
        origin="lower",
        extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
        aspect="equal",
    )
    ax.set_xlabel("x (A), zigzag")
    ax.set_ylabel("y (A), armchair")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean height relative to average plane (A)")
    fig.tight_layout()
    fig.savefig("ripple_height_map.png", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    if not THERMO_FILE.exists():
        raise SystemExit("thermo.out not found. Run GPUMD first.")
    analyze_thermo()
    if FINAL_FILE.exists():
        analyze_ripple()
    else:
        print("final.xyz not found; skipping ripple analysis.")
    print("Wrote temperature.png, energy.png, and (if final.xyz exists) ripple_height_map.png")
