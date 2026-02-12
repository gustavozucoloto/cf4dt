#!/usr/bin/env python
"""Plot Ulamec (2007) ice density rho_ice_ulamec(T).

This uses the implementation in cf4dt.forward and saves a PNG
under outputs/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cf4dt.forward import rho_ice_ulamec  # noqa: E402


def main() -> None:
    # Temperature range ~0–273 K (validity range of correlation)
    T = np.linspace(0.0, 273.0, 274)
    rho = rho_ice_ulamec(T)

    plt.figure(figsize=(7, 5))
    plt.plot(T, rho, linewidth=2)
    plt.xlabel("T (K)")
    plt.ylabel("rho_ice_ulamec(T) [kg/m³]")
    plt.title("Ulamec (2007) ice density rho_ice_ulamec(T)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rho_ice_ulamec.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved rho_ice_ulamec(T) plot to: {out_path}")


if __name__ == "__main__":
    main()
