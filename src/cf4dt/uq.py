"""Uncertainty quantification utilities using GP emulator and posterior samples."""

import numpy as np
import joblib
import matplotlib.pyplot as plt

from .gp_utils import gp_predict
from .forward import alpha_model


def uq_maps(model_name, gp_path, posterior_path, out_prefix="uq", n_post=400, seed=0):
    bundle = joblib.load(gp_path)
    samples = np.load(posterior_path)

    rng = np.random.default_rng(seed)
    if len(samples) > n_post:
        idx = rng.choice(len(samples), size=n_post, replace=False)
        samples = samples[idx]

    W_mph_grid = np.logspace(np.log10(0.05), np.log10(5.0), 30)
    W_grid = W_mph_grid / 3600.0
    Ts_grid = np.linspace(80, 260, 30)

    Q_med = np.zeros((len(Ts_grid), len(W_grid)))
    Q_lo = np.zeros_like(Q_med)
    Q_hi = np.zeros_like(Q_med)

    for iT, Ts in enumerate(Ts_grid):
        for iW, W in enumerate(W_grid):
            X = np.column_stack(
                [
                    np.full(len(samples), W),
                    np.full(len(samples), Ts),
                    samples[:, 0],
                    samples[:, 1],
                ]
            )
            mu, _ = gp_predict(bundle, X)
            # Parametric uncertainty only (no GP predictive noise)
            Q_med[iT, iW] = np.median(mu)
            Q_lo[iT, iW] = np.percentile(mu, 2.5)
            Q_hi[iT, iW] = np.percentile(mu, 97.5)

    plt.figure()
    for Ts_pick in [100, 160, 220, 260]:
        iT = np.argmin(np.abs(Ts_grid - Ts_pick))
        plt.fill_between(W_mph_grid, Q_lo[iT], Q_hi[iT], alpha=0.2)
        plt.plot(W_mph_grid, Q_med[iT], label=f"Ts={Ts_grid[iT]:.0f} K")
    plt.xscale("log")
    plt.xlabel("W (m/hour)")
    plt.ylabel("Q_lc (kW)")
    plt.title(f"Posterior predictive Q_lc bands ({model_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_Qlc_slices_{model_name}.png", dpi=200)

    T = np.linspace(80, 273.15, 200)
    alpha_draws = np.array([alpha_model(T, th, model=model_name) for th in samples])
    a_med = np.median(alpha_draws, axis=0)
    a_lo = np.percentile(alpha_draws, 2.5, axis=0)
    a_hi = np.percentile(alpha_draws, 97.5, axis=0)

    plt.figure()
    plt.fill_between(T, a_lo, a_hi, alpha=0.3)
    plt.plot(T, a_med)
    plt.yscale("log")
    plt.xlabel("T (K)")
    plt.ylabel("alpha(T)")
    plt.title(f"Posterior alpha(T) band ({model_name})")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_alpha_band_{model_name}.png", dpi=200)
