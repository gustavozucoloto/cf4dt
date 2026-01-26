#!/usr/bin/env python
"""CLI to generate synthetic Q_lc dataset using full FEniCSx model."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cf4dt.data_generation import generate_artificial_data


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="artificial_Qlc_data.csv", help="Output CSV path")
    p.add_argument("--n-lhs", type=int, default=120, help="LHS points before edges")
    p.add_argument("--seed-design", type=int, default=1)
    p.add_argument("--seed-noise", type=int, default=2)
    p.add_argument("--sigma-kw", type=float, default=0.1, help="Sensor noise std (kW)")
    p.add_argument("--Ts-min", type=float, default=80.0)
    p.add_argument("--Ts-max", type=float, default=260.0)
    p.add_argument("--W-min", type=float, default=0.05, help="m/hour")
    p.add_argument("--W-max", type=float, default=5.0, help="m/hour")
    p.add_argument("--Ro", type=float, default=0.1, help="Inner radius (m)")
    p.add_argument("--L", type=float, default=3.7, help="Length (m)")
    p.add_argument("--Tm", type=float, default=273.15, help="Melt temperature (K)")
    p.add_argument("--Rinf-offset", type=float, default=5.0, help="Rinf = Ro + offset (m)")
    p.add_argument("--num-cells", type=int, default=1000)
    p.add_argument("--p-grade", type=float, default=3.0)
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--dt-ratio", type=float, default=1.03)
    return p.parse_args()


def main():
    args = parse_args()
    generate_artificial_data(
        out_csv=args.out,
        n_lhs=args.n_lhs,
        seed_design=args.seed_design,
        seed_noise=args.seed_noise,
        sigma_kW=args.sigma_kw,
        Ts_min=args.Ts_min,
        Ts_max=args.Ts_max,
        W_mph_min=args.W_min,
        W_mph_max=args.W_max,
        Ro=args.Ro,
        L=args.L,
        Tm=args.Tm,
        Rinf_offset=args.Rinf_offset,
        num_cells=args.num_cells,
        p_grade=args.p_grade,
        num_steps=args.num_steps,
        dt_ratio=args.dt_ratio,
    )


if __name__ == "__main__":
    main()
