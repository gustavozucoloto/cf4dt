#!/usr/bin/env python
"""CLI to run Bayesian calibration (emcee) using a GP emulator."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cf4dt.calibration import calibrate_and_save


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["powerlaw", "exponential"], required=True)
    p.add_argument("--data", default="artificial_Qlc_data.csv", help="CSV with observations")
    p.add_argument("--gp", default="gp_powerlaw.joblib", help="GP bundle path")
    p.add_argument("--out", default="posterior_powerlaw.npy", help="Output posterior .npy")
    p.add_argument("--nwalkers", type=int, default=32)
    p.add_argument("--nsteps", type=int, default=6000)
    p.add_argument("--burn", type=int, default=1500)
    p.add_argument("--thin", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    calibrate_and_save(
        model_name=args.model,
        data_csv=args.data,
        gp_path=args.gp,
        out_path=args.out,
        nwalkers=args.nwalkers,
        nsteps=args.nsteps,
        burn=args.burn,
        thin=args.thin,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
