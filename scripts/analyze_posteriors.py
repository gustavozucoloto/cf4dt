#!/usr/bin/env python3
"""
Plots from posterior .npy files:

A) 2x2 contraction plot:
   Row 1: beta0 full prior range | beta0 zoom on posterior
   Row 2: beta1 full prior range | beta1 zoom on posterior

   - Prior: Gaussian N(mu, sigma^2) shown as line
   - Posterior: histogram density
   - Median line + 95% credible interval lines
   - Informative figure title with posterior median and std

B) Correlation scatter:
   - Scatter beta0 vs beta1
   - 95% confidence ellipse (Gaussian approximation from covariance)
   - Pearson r annotation
   - Informative title with posterior median and std
"""

import argparse
import glob
import os
import math
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})  # Increase global font size

# === Priors (Gaussian) ===
DEFAULT_BOUNDS = {
    "powerlaw": {"mu": (-13.0461, -1.90311), "sigma": (0.6523, 0.0952)},
    "exponential": {"mu": (-12.9974, -0.011974), "sigma": (0.6499, 0.000599)},
    "logarithmic": {"mu": (-12.9822, -3.40888), "sigma": (0.6491, 0.1704)},
}

MODELS = ["powerlaw", "exponential", "logarithmic"]


def gaussian_pdf(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * math.sqrt(2.0 * math.pi))


def infer_model_from_filename(fname: str) -> str:
    low = fname.lower()
    for m in MODELS:
        if m in low:
            return m
    return "unknown"


def validate_samples(samples: np.ndarray, path: str):
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError(f"{path}: expected shape (N,2), got {samples.shape}")
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"{path}: contains NaN/inf values")


def posterior_stats(samples: np.ndarray):
    """
    Returns:
      med: (2,)
      std: (2,)
      q025: (2,)
      q975: (2,)
      corr: float
    """
    med = np.median(samples, axis=0)
    std = samples.std(axis=0, ddof=0)
    q025 = np.quantile(samples, 0.025, axis=0)
    q975 = np.quantile(samples, 0.975, axis=0)
    corr = float(np.corrcoef(samples[:, 0], samples[:, 1])[0, 1])
    return med, std, q025, q975, corr


def add_median_and_ci_lines(ax, median, qlo, qhi):
    # Median (solid)
    ax.axvline(median, linewidth=2.0, linestyle="-", alpha=0.9, label="median" if "median" not in ax.get_legend_handles_labels()[1] else None)
    # 95% CI (dashed)
    ax.axvline(qlo, linewidth=1.5, linestyle="--", alpha=0.9, label="95% CI" if "95% CI" not in ax.get_legend_handles_labels()[1] else None)
    ax.axvline(qhi, linewidth=1.5, linestyle="--", alpha=0.9)


def make_2x2_contraction_plot(model: str, samples: np.ndarray, outpath: str):
    if model not in DEFAULT_BOUNDS:
        raise ValueError(f"Unknown model '{model}'. Add it to DEFAULT_BOUNDS.")

    mu0, mu1 = DEFAULT_BOUNDS[model]["mu"]
    s0, s1 = DEFAULT_BOUNDS[model]["sigma"]
    mus = [float(mu0), float(mu1)]
    sigs = [float(s0), float(s1)]

    med, std, q025, q975, _corr = posterior_stats(samples)

    # title = (
    #     rf"$\beta_0$: median={med[0]:.6g}, std={std[0]:.3g}   "
    #     rf"$\beta_1$: median={med[1]:.6g}, std={std[1]:.3g}"
    # )

    param_labels = [r"$\beta_0$", r"$\beta_1$"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)  # Square figure

    for row in range(2):
        mu = mus[row]
        sig = sigs[row]
        x = samples[:, row].astype(float)

        # ----- Left: full prior range (mu ± 4 sigma) -----
        ax = axes[row, 0]
        xmin = mu - 4.0 * sig
        xmax = mu + 4.0 * sig
        xs = np.linspace(xmin, xmax, 800)

        ax.hist(x, bins=60, range=(xmin, xmax), density=True, alpha=0.55, label="posterior")
        ax.plot(xs, gaussian_pdf(xs, mu, sig), linewidth=2.0, label="prior")

        # Median + CI
        add_median_and_ci_lines(ax, med[row], q025[row], q975[row])

        # Show prior clearly (posterior spike can clip here)
        prior_peak = 1.0 / (sig * math.sqrt(2.0 * math.pi))
        ax.set_ylim(0.0, prior_peak * 1.25)

        ax.set_xlim(xmin, xmax)
        ax.grid(True, alpha=0.25)

        # ----- Right: zoom-in around posterior mass -----
        axz = axes[row, 1]
        # Use 0.5%..99.5% for robust zoom
        qlo, qhi = np.quantile(x, [0.005, 0.995])
        pad = 0.05 * (qhi - qlo) if qhi > qlo else 1e-6
        zxmin = float(qlo - pad)
        zxmax = float(qhi + pad)
        zxs = np.linspace(zxmin, zxmax, 600)

        axz.hist(x, bins=60, range=(zxmin, zxmax), density=True, alpha=0.55, label="posterior")
        axz.plot(zxs, gaussian_pdf(zxs, mu, sig), linewidth=2.0, label="prior")

        # Median + CI on zoom as well
        add_median_and_ci_lines(axz, med[row], q025[row], q975[row])

        axz.set_xlim(zxmin, zxmax)
        axz.grid(True, alpha=0.25)

        # Labels (LaTeX)
        ax.set_xlabel(param_labels[row])
        axz.set_xlabel(param_labels[row])
        ax.set_ylabel("density")
        axz.set_ylabel("density")

        # Avoid scientific offset notation
        ax.ticklabel_format(style="plain", useOffset=False, axis="x")
        axz.ticklabel_format(style="plain", useOffset=False, axis="x")

    # Shared legend (top center)
    # Grab unique labels only once
    handles, labels = [], []
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll and ll not in labels:
                labels.append(ll)
                handles.append(hh)

    # Put legend in the upper-left of the first subplot (axes[0,0])
    leg = axes[0, 0].legend(
        handles, labels,
        loc="upper left",
        frameon=True,
        borderpad=0.4,
        handlelength=2.0,
        fontsize=12
    )
    # Make legend readable but not too intrusive
    leg.get_frame().set_alpha(0.85)
    leg.get_frame().set_facecolor("white")

    # fig.suptitle(title, y=1.02)

    plt.savefig(outpath, format='svg', transparent=True)  # Save as SVG for slides


def add_confidence_ellipse(ax, samples: np.ndarray, conf_scale: float = 2.4477):
    """
    Adds a 95% ellipse based on sample covariance.
    conf_scale = sqrt(chi2.ppf(0.95, 2)) ≈ 2.4477
    """
    x = samples[:, 0]
    y = samples[:, 1]
    mean = np.array([x.mean(), y.mean()])
    cov = np.cov(np.vstack([x, y]), ddof=0)

    # Eigen-decomposition
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    t = np.linspace(0, 2 * np.pi, 400)
    circle = np.vstack((np.cos(t), np.sin(t)))
    ell = (vecs @ (np.diag(np.sqrt(vals)) @ circle)) * conf_scale
    ell[0, :] += mean[0]
    ell[1, :] += mean[1]

    ax.plot(ell[0, :], ell[1, :], linewidth=2.0, label="95% ellipse")


def make_corr_scatter(samples: np.ndarray, outpath: str):
    med, std, _q025, _q975, corr = posterior_stats(samples)

    # title = (
    #     rf"$\beta_0$: median={med[0]:.6g}, std={std[0]:.3g}   "
    #     rf"$\beta_1$: median={med[1]:.6g}, std={std[1]:.3g}"
    # )

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.25, label="samples")

    add_confidence_ellipse(ax, samples)

    ax.set_xlabel(r"$\beta_0$")
    ax.set_ylabel(r"$\beta_1$")
    ax.grid(True, alpha=0.25)

    ax.text(
        0.02, 0.98, rf"Pearson $r$ = {corr:+.3f}",
        transform=ax.transAxes, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
    )

    # Avoid scientific offset notation
    ax.ticklabel_format(style="plain", useOffset=False, axis="x")
    ax.ticklabel_format(style="plain", useOffset=False, axis="y")

    ax.legend(loc="lower right", frameon=True)
    # fig.suptitle(title, y=1.02)

    fig.savefig(outpath, format='svg', bbox_inches="tight", transparent=True)  # Save as SVG for slides
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="data", help="Folder containing posterior_*.npy files")
    ap.add_argument("--outdir", default="figures", help="Folder to write PNG figures")
    ap.add_argument("--pattern", default="posterior_*.npy", help="Glob pattern inside indir")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.indir, args.pattern)))
    if not paths:
        raise SystemExit(f"No files found: {os.path.join(args.indir, args.pattern)}")

    for p in paths:
        base = os.path.basename(p)
        model = infer_model_from_filename(base)
        if model == "unknown":
            print(f"[SKIP] Could not infer model from filename: {base}")
            continue

        samples = np.load(p)
        validate_samples(samples, p)

        stem = os.path.splitext(base)[0]
        out_contr = os.path.join(args.outdir, stem + "__2x2_contraction.svg")
        out_corr = os.path.join(args.outdir, stem + "__corr_scatter.svg")

        make_2x2_contraction_plot(model, samples, out_contr)
        make_corr_scatter(samples, out_corr)

        print(f"[OK] {base}")
        print(f"     -> {out_contr}")
        print(f"     -> {out_corr}")

    print("Done.")


if __name__ == "__main__":
    main()
