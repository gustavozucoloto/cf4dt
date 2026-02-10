#!/usr/bin/env python
"""Plot Ulamec (2007) k, cp, and rho vs temperature."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cf4dt.forward import k_ice_ulamec, cp_ice_ulamec, rho_ice_ulamec


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="outputs/ulamec_properties.svg")
    p.add_argument("--tmin", type=float, default=80.0)
    p.add_argument("--tmax", type=float, default=273.15)
    p.add_argument("--n", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()

    T = np.linspace(args.tmin, args.tmax, args.n)
    k = k_ice_ulamec(T)
    cp = cp_ice_ulamec(T)
    rho = rho_ice_ulamec(T)
    alpha = k / (rho * cp)

    fig, ax = plt.subplots(4, 1, figsize=(4, 5), sharex=True)

    ax[0].plot(T, alpha, color="C0")
    ax[0].set_ylabel(r"$\alpha$ (m$^2$/s)", fontsize=12)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(T, k, color="C0")
    ax[1].set_ylabel(r"$k$ (W/m/K)", fontsize=12)
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(T, cp, color="C0")
    ax[2].set_ylabel(r"$c_p$ (J/kg/K)", fontsize=12)
    ax[2].grid(True, alpha=0.3)

    ax[3].plot(T, rho, color="C0")
    ax[3].set_ylabel(r"$\rho$ (kg/m$^3$)", fontsize=12)
    ax[3].set_xlabel(r"$T$ (K)", fontsize=12)
    ax[3].grid(True, alpha=0.3)

    for axis in ax:
        axis.tick_params(axis="both", labelsize=11)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
