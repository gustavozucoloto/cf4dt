#!/usr/bin/env python
"""Compute prior bounds for toy models based on the Ulamec alpha(T)."""

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cf4dt.forward import k_ice_ulamec, cp_ice_ulamec, rho_ice_ulamec, alpha_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tmin", type=float, default=80.0)
    p.add_argument("--tmax", type=float, default=260.0)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--t0", type=float, default=200.0)
    p.add_argument("--rel-pad", type=float, default=0.10, help="Relative padding for bounds")
    p.add_argument("--plot", action="store_true", help="Save a plot of fits and bounds")
    p.add_argument("--out-plot", default="outputs/prior_bounds_fit.svg")
    return p.parse_args()


def alpha_ulamec(T):
    k = k_ice_ulamec(T)
    rho = rho_ice_ulamec(T)
    cp = cp_ice_ulamec(T)
    return k / (rho * cp)


def fit_powerlaw(T, alpha, T0):
    x = np.log(T / T0)
    y = np.log(alpha)
    A = np.column_stack([np.ones_like(x), x])
    beta0, beta1 = np.linalg.lstsq(A, y, rcond=None)[0]
    return beta0, beta1


def fit_exponential(T, alpha, T0):
    x = T - T0
    y = np.log(alpha)
    A = np.column_stack([np.ones_like(x), x])
    beta0, beta1 = np.linalg.lstsq(A, y, rcond=None)[0]
    return beta0, beta1


def fit_logarithmic(T, alpha, T0):
    x = np.log(T / T0)
    A = np.column_stack([np.ones_like(x), x])
    a, b = np.linalg.lstsq(A, alpha, rcond=None)[0]
    if a <= 0:
        a = np.finfo(float).eps
    beta0 = np.log(a)
    beta1 = b / a
    return beta0, beta1


def padded_bounds(value, rel_pad):
    lo = value * (1.0 - rel_pad)
    hi = value * (1.0 + rel_pad)
    return (min(lo, hi), max(lo, hi))


def alpha_bounds(T, model, beta0_bounds, beta1_bounds, T0):
    b0_lo, b0_hi = beta0_bounds
    b1_lo, b1_hi = beta1_bounds
    corners = [
        (b0_lo, b1_lo),
        (b0_lo, b1_hi),
        (b0_hi, b1_lo),
        (b0_hi, b1_hi),
    ]
    values = np.vstack([alpha_model(T, theta, model=model, T0=T0) for theta in corners])
    return values.min(axis=0), values.max(axis=0)


def main():
    args = parse_args()

    T = np.linspace(args.tmin, args.tmax, args.n)
    alpha = alpha_ulamec(T)

    fits = {
        "powerlaw": fit_powerlaw(T, alpha, args.t0),
        "exponential": fit_exponential(T, alpha, args.t0),
        "logarithmic": fit_logarithmic(T, alpha, args.t0),
    }

    print("Prior bounds based on Ulamec alpha(T)")
    print(f"T range: {args.tmin:.2f} to {args.tmax:.2f} K, n={args.n}, T0={args.t0:.2f}")
    print(f"Relative padding: {args.rel_pad:.2%}\n")

    bounds_by_model = {}
    for model, (beta0, beta1) in fits.items():
        b0_bounds = padded_bounds(beta0, args.rel_pad)
        b1_bounds = padded_bounds(beta1, args.rel_pad)
        bounds_by_model[model] = (b0_bounds, b1_bounds)
        print(f"MODEL: {model}")
        print(f"  fit beta0={beta0:.6g}, beta1={beta1:.6g}")
        print(
            "  bounds: beta0=(%.6g, %.6g), beta1=(%.6g, %.6g)"
            % (b0_bounds[0], b0_bounds[1], b1_bounds[0], b1_bounds[1])
        )
        print("")

    if args.plot:
        fig, axes = plt.subplots(3, 1, figsize=(5.0, 7.0), sharex=True)
        models = ["powerlaw", "exponential", "logarithmic"]

        for ax, model in zip(axes, models):
            beta0, beta1 = fits[model]
            b0_bounds, b1_bounds = bounds_by_model[model]

            alpha_fit = alpha_model(T, (beta0, beta1), model=model, T0=args.t0)
            alpha_lo, alpha_hi = alpha_bounds(T, model, b0_bounds, b1_bounds, args.t0)

            ax.plot(T, alpha, color="black", lw=1.5, label="Ulamec alpha")
            ax.plot(T, alpha_fit, color="C0", lw=1.5, label="Fit")
            ax.fill_between(T, alpha_lo, alpha_hi, color="C0", alpha=0.2, label="Bounds")
            ax.set_ylabel(r"$\alpha$ (m$^2$/s)")
            ax.set_title(model)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("T (K)")
        axes[0].legend(loc="best", fontsize=9)

        out_path = Path(args.out_plot)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
