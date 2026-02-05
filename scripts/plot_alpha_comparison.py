#!/usr/bin/env python
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cf4dt.forward import alpha_model, k_ice_ulamec, rho_ice_ulamec, cp_ice_ulamec


def load_samples(path):
    samples = np.load(path)
    return samples


def summarize_alpha(samples, model_name, T):
    alpha_draws = np.array([alpha_model(T, th, model=model_name) for th in samples])
    a_med = np.median(alpha_draws, axis=0)
    a_lo = np.percentile(alpha_draws, 2.5, axis=0)
    a_hi = np.percentile(alpha_draws, 97.5, axis=0)
    return a_lo, a_med, a_hi


def main():
    powerlaw_path = Path("data/posterior_powerlaw.npy")
    exp_path = Path("data/posterior_exponential.npy")

    if not powerlaw_path.exists() or not exp_path.exists():
        raise FileNotFoundError(
            "Expected posterior files at data/posterior_powerlaw.npy and data/posterior_exponential.npy"
        )

    samples_pl = load_samples(powerlaw_path)
    samples_exp = load_samples(exp_path)

    T = np.linspace(80, 273.15, 200)

    a_lo_pl, a_med_pl, a_hi_pl = summarize_alpha(samples_pl, "powerlaw", T)
    a_lo_exp, a_med_exp, a_hi_exp = summarize_alpha(samples_exp, "exponential", T)

    alpha_ulamec = k_ice_ulamec(T) / (rho_ice_ulamec(T) * cp_ice_ulamec(T))

    plt.figure(figsize=(7, 5))
    plt.fill_between(T, a_lo_pl, a_hi_pl, alpha=0.25, label="Powerlaw 95% band")
    plt.plot(T, a_med_pl, label="Powerlaw median")

    plt.fill_between(T, a_lo_exp, a_hi_exp, alpha=0.25, label="Exponential 95% band")
    plt.plot(T, a_med_exp, label="Exponential median")

    plt.plot(T, alpha_ulamec, "k--", linewidth=2.0, label="Ulamec (2007)")

    plt.yscale("log")
    plt.xlabel("T (K)")
    plt.ylabel("alpha(T) [m²/s]")
    plt.title("Posterior alpha(T) comparison")
    plt.legend()
    plt.tight_layout()

    out_path = Path("outputs/alpha_models_posterior_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
