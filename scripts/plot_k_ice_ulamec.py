#!/usr/bin/env python
"""Plot Ulamec (2007) ice thermal conductivity k_ice_ulamec(T).

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

from cf4dt.forward import k_ice_ulamec  # noqa: E402


def main() -> None:
    # Temperature range ~10–273 K (validity range of correlation)
    T = np.linspace(10.0, 273.0, 264)
    k = k_ice_ulamec(T)

    plt.figure(figsize=(7, 5))
    plt.plot(T, k, linewidth=2)
    plt.xlabel("T (K)")
    plt.ylabel("k_ice_ulamec(T) [W/(m·K)]")
    plt.title("Ulamec (2007) ice thermal conductivity k_ice_ulamec(T)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "k_ice_ulamec.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved k_ice_ulamec(T) plot to: {out_path}")


if __name__ == "__main__":
    main()
