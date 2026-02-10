#!/usr/bin/env python
"""Plot artificial calibration data (Qlc vs W) grouped by Ts."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/artificial_Qlc_data.csv")
    p.add_argument("--out", default="outputs/artificial_data_plot.svg")
    p.add_argument("--use-obs", action="store_true", help="Plot y_obs_kW instead of Qlc_true_kW")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.data)
    y_col = "y_obs_kW" if args.use_obs else "Qlc_true_kW"

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    for Ts in np.sort(df["Ts_K"].unique()):
        sub = df[df["Ts_K"] == Ts]
        sub = sub.sort_values("W_mph")
        ax.plot(sub["W_mph"], sub[y_col], marker="o", label=f"Ts={Ts:.0f} K")

    ax.set_xlabel(r"$W$ (m/hour)", fontsize=16)
    ax.set_ylabel(r"$Q_{lc}$ (kW)", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=12, loc="center right", bbox_to_anchor=(-0.25, 0.5), frameon=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
