#!/usr/bin/env python
"""CLI to train GP emulator for a toy forward model."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cf4dt.emulator import train_and_save_emulator


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["powerlaw", "exponential"], required=True)
    p.add_argument("--data", default="artificial_Qlc_data.csv", help="Input CSV with W/Ts/y")
    p.add_argument("--out", default="gp_powerlaw.joblib", help="Output GP bundle path")
    p.add_argument("--n-theta", type=int, default=40, help="Number of theta samples")
    p.add_argument("--subset", type=int, default=64, help="Number of design points to subsample")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-jobs", type=int, default=1, help="Number of parallel processes (1=serial)")
    return p.parse_args()


def main():
    args = parse_args()
    train_and_save_emulator(
        df_path=args.data,
        model_name=args.model,
        out_path=args.out,
        n_theta=args.n_theta,
        use_subset_points=args.subset,
        seed=args.seed,
        solver_kwargs={},
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
