#!/usr/bin/env python
"""Plot Ulamec (2007) ice properties rho, k, cp, and alpha on one figure.

Uses the implementations in cf4dt.forward and saves an SVG
under outputs/.
"""

import sys
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cf4dt.forward import rho_ice_ulamec, k_ice_ulamec, cp_ice_ulamec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Ulamec (2007) ice properties rho, k, cp, and alpha "
            "over a configurable temperature range."
        )
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=10.0,
        help="Minimum temperature in K (default: 10.0)",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=273.0,
        help="Maximum temperature in K (default: 273.0)",
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=264,
        help="Number of temperature samples (default: 264)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.tmax <= args.tmin:
        raise ValueError("tmax must be greater than tmin")
    if args.npoints < 2:
        raise ValueError("npoints must be at least 2")

    # Use a configurable temperature range (defaults stay within validity)
    T = np.linspace(args.tmin, args.tmax, args.npoints)

    rho = rho_ice_ulamec(T)
    k = k_ice_ulamec(T)
    cp = cp_ice_ulamec(T)
    alpha = k / (rho * cp)

    fig, axes = plt.subplots(4, 1, figsize=(7, 11), sharex=True)
    # Transparent background for embedding in other figures/documents
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.set_facecolor("none")

    ax = axes[0]
    ax.plot(T, rho, linewidth=2)
    ax.set_ylabel("rho(T) [kg/m³]")
    ax.set_title("Ulamec (2007) ice properties")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(T, k, linewidth=2)
    ax.set_ylabel("k(T) [W/(m·K)]")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(T, cp, linewidth=2)
    ax.set_xlabel("T (K)")
    ax.set_ylabel("cp(T) [J/(kg·K)]")
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(T, alpha, linewidth=2)
    ax.set_xlabel("T (K)")
    ax.set_ylabel("alpha(T) [m²/s]")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ice_properties_ulamec.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved Ulamec ice properties plot to: {out_path}")


if __name__ == "__main__":
    main()
