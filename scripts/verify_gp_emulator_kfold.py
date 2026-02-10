#!/usr/bin/env python
"""
Lightweight k-fold cross-validation for GP emulators.

Builds a training set from the forward model once, then performs
k-fold CV by re-fitting the GP on each training split.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cf4dt.emulator import build_training_set, fit_gp
from cf4dt.gp_utils import gp_predict


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        default="powerlaw,exponential,logarithmic",
        help="Comma-separated model list (powerlaw,exponential,logarithmic)",
    )
    p.add_argument("--data", default="data/artificial_Qlc_data.csv")
    p.add_argument("--n-theta", type=int, default=10)
    p.add_argument("--subset", type=int, default=20)
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--out", default="outputs/04_gp_kfold_report.txt")
    return p.parse_args()


def compute_metrics(y_true, mu, std):
    rmse = float(np.sqrt(np.mean((y_true - mu) ** 2)))
    mae = float(np.mean(np.abs(y_true - mu)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum((y_true - mu) ** 2) / denom) if denom > 0 else float("nan")
    coverage = float(np.mean((y_true >= mu - 2 * std) & (y_true <= mu + 2 * std)))
    return rmse, mae, r2, coverage


def main():
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    df = pd.read_csv(args.data)

    lines = []
    lines.append("=" * 70)
    lines.append("GP EMULATOR K-FOLD REPORT")
    lines.append(f"kfold={args.kfold}, n_theta={args.n_theta}, subset={args.subset}, seed={args.seed}")
    lines.append("=" * 70)

    for model_name in models:
        lines.append("")
        lines.append(f"MODEL: {model_name}")
        X, y = build_training_set(
            df,
            model_name=model_name,
            n_theta=args.n_theta,
            seed=args.seed,
            use_subset_points=args.subset,
            n_jobs=args.n_jobs,
        )

        n_samples = X.shape[0]
        kfold = min(args.kfold, n_samples)
        if kfold < 2:
            raise ValueError("kfold must be >= 2 and <= number of samples")

        kf = KFold(n_splits=kfold, shuffle=True, random_state=args.seed)
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            gp, xscaler, yscaler = fit_gp(X_train, y_train)
            bundle = dict(gp=gp, xscaler=xscaler, yscaler=yscaler)
            mu, std = gp_predict(bundle, X_test)

            rmse, mae, r2, coverage = compute_metrics(y_test, mu, std)
            fold_metrics.append([rmse, mae, r2, coverage])

            lines.append(
                f"  Fold {fold_idx:02d}: RMSE={rmse:.4f} kW, MAE={mae:.4f} kW, "
                f"R2={r2:.6f}, 95%Cov={coverage:.1%}"
            )

        metrics = np.array(fold_metrics)
        mean = metrics.mean(axis=0)
        std = metrics.std(axis=0)

        lines.append("  Summary (mean +/- std):")
        lines.append(
            f"    RMSE={mean[0]:.4f} +/- {std[0]:.4f} kW, "
            f"MAE={mean[1]:.4f} +/- {std[1]:.4f} kW, "
            f"R2={mean[2]:.6f} +/- {std[2]:.6f}, "
            f"95%Cov={mean[3]:.1%} +/- {std[3]:.1%}"
        )

    lines.append("")
    lines.append("DONE")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
