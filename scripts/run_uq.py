#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cf4dt.uq import uq_maps


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["powerlaw", "exponential"], required=True)
    p.add_argument("--gp", default="gp_powerlaw.joblib", help="GP bundle path")
    p.add_argument("--posterior", default="posterior_powerlaw.npy", help="Posterior samples .npy")
    p.add_argument("--out-prefix", default="outputs/uq", help="Prefix for output figures")
    p.add_argument("--n-post", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    uq_maps(
        model_name=args.model,
        gp_path=args.gp,
        posterior_path=args.posterior,
        out_prefix=args.out_prefix,
        n_post=args.n_post,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
