#!/usr/bin/env python
"""Plot posterior distributions of calibrated parameters from saved files.

This script reads the posterior samples stored in data/posterior_powerlaw.npy
and data/posterior_exponential.npy (produced by the Bayesian calibration
workflow) and creates histograms of the calibrated parameters.

It does *not* rerun the calibration; it only uses the already-saved data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    powerlaw_path = data_dir / "posterior_powerlaw.npy"
    exponential_path = data_dir / "posterior_exponential.npy"
    logarithmic_path = data_dir / "posterior_logarithmic.npy"

    if not (powerlaw_path.exists() and exponential_path.exists() and logarithmic_path.exists()):
        raise FileNotFoundError(
            "Expected posterior files at data/posterior_powerlaw.npy, "
            "data/posterior_exponential.npy, and data/posterior_logarithmic.npy. "
            "Run the calibration first to generate these files."
        )

    posterior_powerlaw = np.load(powerlaw_path)
    posterior_exponential = np.load(exponential_path)
    posterior_logarithmic = np.load(logarithmic_path)

    if posterior_powerlaw.ndim != 2 or posterior_powerlaw.shape[1] != 2:
        raise ValueError(
            f"Expected powerlaw posterior of shape (nsamples, 2), got {posterior_powerlaw.shape}"
        )
    if posterior_exponential.ndim != 2 or posterior_exponential.shape[1] != 2:
        raise ValueError(
            f"Expected exponential posterior of shape (nsamples, 2), got {posterior_exponential.shape}"
        )
    if posterior_logarithmic.ndim != 2 or posterior_logarithmic.shape[1] != 2:
        raise ValueError(
            f"Expected logarithmic posterior of shape (nsamples, 2), got {posterior_logarithmic.shape}"
        )

    # Use plain-text labels to avoid mathtext parsing issues
    param_labels = ["beta_0", "beta_1"]

    # Create one separate figure per model, each with the beta_0/beta_1 pair
    model_names = ["powerlaw", "exponential", "logarithmic"]
    posteriors = [posterior_powerlaw, posterior_exponential, posterior_logarithmic]
    colors = ["tab:red", "tab:blue", "tab:green"]

    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model, samples, color in zip(model_names, posteriors, colors):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for j_param, (ax, param_label) in enumerate(zip(axes, param_labels)):
            ax.hist(
                samples[:, j_param],
                bins=50,
                density=True,
                alpha=0.7,
                color=color,
            )
            ax.set_xlabel(param_label)
            ax.set_ylabel("Density")
            ax.set_title(f"{model.capitalize()} - {param_label}")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Posterior distributions for {model} model")
        fig.tight_layout(rect=(0, 0, 1, 0.9))

        out_path = out_dir / f"posterior_parameters_{model}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved posterior distribution plot for {model} to: {out_path}")


if __name__ == "__main__":
    main()
