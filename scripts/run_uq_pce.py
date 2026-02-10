#!/usr/bin/env python
"""PCE-based UQ using polynomial regression on posterior samples."""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cf4dt.gp_utils import gp_predict


def total_degree_exponents(dim, degree):
    exps = []

    def rec(curr, left, idx):
        if idx == dim - 1:
            exps.append(curr + [left])
            return
        for k in range(left + 1):
            rec(curr + [k], left - k, idx + 1)

    for d in range(degree + 1):
        rec([], d, 0)
    return exps


def design_matrix(X, exps):
    Phi = np.ones((X.shape[0], len(exps)), dtype=float)
    for j, exp in enumerate(exps):
        for i, p in enumerate(exp):
            if p:
                Phi[:, j] *= X[:, i] ** p
    return Phi


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["powerlaw", "exponential", "logarithmic"], required=True)
    p.add_argument("--gp", default="data/gp_powerlaw.joblib", help="GP bundle path")
    p.add_argument("--posterior", default="data/posterior_powerlaw.npy", help="Posterior samples .npy")
    p.add_argument("--data", default="data/artificial_Qlc_data.csv", help="CSV with W_mph/Ts_K")
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--n-train", type=int, default=30, help="Posterior samples to fit PCE")
    p.add_argument("--n-eval", type=int, default=1000, help="Posterior samples to evaluate PCE")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-prefix", default="outputs/uq_pce", help="Prefix for output figures")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.data)
    W_grid = np.unique(df["W_mph"].to_numpy())
    Ts_grid = np.unique(df["Ts_K"].to_numpy())

    bundle = joblib.load(args.gp)
    samples = np.load(args.posterior)

    n_train = min(args.n_train, len(samples))
    n_eval = min(args.n_eval, len(samples))

    theta_mean = samples.mean(axis=0)
    theta_std = samples.std(axis=0)
    theta_std = np.where(theta_std == 0.0, 1.0, theta_std)

    exps = total_degree_exponents(2, args.degree)

    Q_med = np.zeros((len(Ts_grid), len(W_grid)))
    Q_lo = np.zeros_like(Q_med)
    Q_hi = np.zeros_like(Q_med)

    for iT, Ts in enumerate(Ts_grid):
        for iW, W_mph in enumerate(W_grid):
            train_idx = rng.choice(len(samples), size=n_train, replace=False)
            eval_idx = rng.choice(len(samples), size=n_eval, replace=False)

            theta_train = samples[train_idx]
            theta_eval = samples[eval_idx]

            Z_train = (theta_train - theta_mean) / theta_std
            Z_eval = (theta_eval - theta_mean) / theta_std

            X_train = np.column_stack([
                np.full(n_train, W_mph),
                np.full(n_train, Ts),
                theta_train[:, 0],
                theta_train[:, 1],
            ])
            mu_train, _ = gp_predict(bundle, X_train)

            Phi_train = design_matrix(Z_train, exps)
            coeffs, _, _, _ = np.linalg.lstsq(Phi_train, mu_train, rcond=None)

            Phi_eval = design_matrix(Z_eval, exps)
            mu_eval = Phi_eval @ coeffs

            Q_med[iT, iW] = np.median(mu_eval)
            Q_lo[iT, iW] = np.percentile(mu_eval, 2.5)
            Q_hi[iT, iW] = np.percentile(mu_eval, 97.5)

    # Plot slices
    plt.figure()
    for Ts_pick in [80, 120, 160, 200]:
        iT = np.argmin(np.abs(Ts_grid - Ts_pick))
        plt.fill_between(W_grid, Q_lo[iT], Q_hi[iT], alpha=0.2)
        plt.plot(W_grid, Q_med[iT], label=f"Ts={Ts_grid[iT]:.0f} K")
    plt.xlabel("W (m/hour)")
    plt.ylabel("Q_lc (kW)")
    plt.title(f"PCE UQ: Q_lc bands ({args.model})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_Qlc_slices_{args.model}.png", dpi=200)

    # Save arrays
    np.savez(
        f"{args.out_prefix}_{args.model}.npz",
        W_mph=W_grid,
        Ts_K=Ts_grid,
        Q_med=Q_med,
        Q_lo=Q_lo,
        Q_hi=Q_hi,
        degree=args.degree,
        n_train=n_train,
        n_eval=n_eval,
    )

    print(f"Saved: {args.out_prefix}_Qlc_slices_{args.model}.png")
    print(f"Saved: {args.out_prefix}_{args.model}.npz")


if __name__ == "__main__":
    main()
