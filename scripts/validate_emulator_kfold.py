#!/usr/bin/env python

"""Validate GP emulator with K-fold cross-validation."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cf4dt.emulator import validate_emulator_kfold


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["powerlaw", "exponential"], required=True)
    p.add_argument("--data", default="artificial_Qlc_data.csv", help="Input CSV with W/Ts/y")
    p.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    p.add_argument("--n-theta", type=int, default=40, help="Number of theta samples")
    p.add_argument("--subset", type=int, default=64, help="Number of design points to subsample")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-jobs", type=int, default=1, help="Number of parallel processes (1=serial)")
    p.add_argument("--log", default=None, help="Optional log file path")
    p.add_argument("--save-dir", default=None, help="Directory to save per-fold model bundles")
    return p.parse_args()


def main():
    args = parse_args()
    results = validate_emulator_kfold(
        df_path=args.data,
        model_name=args.model,
        n_splits=args.n_splits,
        n_theta=args.n_theta,
        use_subset_points=args.subset,
        seed=args.seed,
        solver_kwargs={},
        n_jobs=args.n_jobs,
        save_dir=args.save_dir,
    )

    summary = results["summary"]
    print("K-fold validation summary")
    print(
        f"  RMSE: {summary['rmse_mean']:.4f} +/- {summary['rmse_std']:.4f} (kW)"
    )
    print(f"  MAE:  {summary['mae_mean']:.4f} +/- {summary['mae_std']:.4f} (kW)")
    print(f"  R2:   {summary['r2_mean']:.4f} +/- {summary['r2_std']:.4f}")
    print(f"  Samples: {summary['n_samples']} | Folds: {summary['n_splits']}")

    print("\nPer-fold metrics")
    for m in results["folds"]:
        print(
            f"  Fold {m['fold']:>2}: RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} R2={m['r2']:.4f} n={m['n']}"
        )

    if args.log:
        lines = []
        lines.append("K-fold validation summary")
        lines.append(
            f"  RMSE: {summary['rmse_mean']:.4f} +/- {summary['rmse_std']:.4f} (kW)"
        )
        lines.append(
            f"  MAE:  {summary['mae_mean']:.4f} +/- {summary['mae_std']:.4f} (kW)"
        )
        lines.append(f"  R2:   {summary['r2_mean']:.4f} +/- {summary['r2_std']:.4f}")
        lines.append(
            f"  Samples: {summary['n_samples']} | Folds: {summary['n_splits']}"
        )
        lines.append("")
        lines.append("Per-fold metrics")
        for m in results["folds"]:
            lines.append(
                f"  Fold {m['fold']:>2}: RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} R2={m['r2']:.4f} n={m['n']}"
            )

        with open(args.log, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        print(f"\nSaved log: {args.log}")


if __name__ == "__main__":
    main()
