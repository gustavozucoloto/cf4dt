#!/usr/bin/env python
"""Quick plot to compare calibrated models against Ulamec reference."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cf4dt.forward import alpha_model, k_ice_ulamec, rho_ice_ulamec, cp_ice_ulamec

# Load posterior samples
posterior_powerlaw = np.load(ROOT / "data/posterior_powerlaw_clean.npy")
posterior_exponential = np.load(ROOT / "data/posterior_exponential.npy")
posterior_logarithmic = np.load(ROOT / "data/posterior_logarithmic.npy")

# Get mean calibrated parameters
theta_powerlaw_mean = posterior_powerlaw.mean(axis=0)
theta_exponential_mean = posterior_exponential.mean(axis=0)
theta_logarithmic_mean = posterior_logarithmic.mean(axis=0)

print(f"Powerlaw calibrated: β₀={theta_powerlaw_mean[0]:.3f}, β₁={theta_powerlaw_mean[1]:.3f}")
print(f"Exponential calibrated: β₀={theta_exponential_mean[0]:.3f}, β₁={theta_exponential_mean[1]:.3f}")
print(f"Logarithmic calibrated: β₀={theta_logarithmic_mean[0]:.3f}, β₁={theta_logarithmic_mean[1]:.3f}")

# Temperature range
T = np.linspace(80, 273, 200)

# Compute alpha for each model
alpha_ulamec = k_ice_ulamec(T) / (rho_ice_ulamec(T) * cp_ice_ulamec(T))
alpha_powerlaw_calib = alpha_model(T, theta=theta_powerlaw_mean, model="powerlaw")
alpha_exponential_calib = alpha_model(T, theta=theta_exponential_mean, model="exponential")
alpha_logarithmic_calib = alpha_model(T, theta=theta_logarithmic_mean, model="logarithmic")

# Sample from posterior for uncertainty bands
n_samples = 10000
idx_powerlaw = np.random.choice(len(posterior_powerlaw), n_samples, replace=False)
idx_exponential = np.random.choice(len(posterior_exponential), n_samples, replace=False)
idx_logarithmic = np.random.choice(len(posterior_logarithmic), n_samples, replace=False)

alpha_powerlaw_samples = np.array([
    alpha_model(T, theta=posterior_powerlaw[i], model="powerlaw")
    for i in idx_powerlaw
])
alpha_exponential_samples = np.array([
    alpha_model(T, theta=posterior_exponential[i], model="exponential")
    for i in idx_exponential
])
alpha_logarithmic_samples = np.array([
    alpha_model(T, theta=posterior_logarithmic[i], model="logarithmic")
    for i in idx_logarithmic
])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

model_specs = [
    ("Powerlaw", theta_powerlaw_mean, alpha_powerlaw_calib, alpha_powerlaw_samples, "crimson"),
    ("Exponential", theta_exponential_mean, alpha_exponential_calib, alpha_exponential_samples, "teal"),
    ("Logarithmic", theta_logarithmic_mean, alpha_logarithmic_calib, alpha_logarithmic_samples, "slateblue"),
]

for ax, (label, theta_mean, alpha_calib, alpha_samples, color) in zip(axes, model_specs):
    ax.plot(T, alpha_ulamec * 1e6, 'k-', linewidth=2, label='Ulamec (2007) - Truth', zorder=10)
    ax.plot(T, alpha_calib * 1e6, color=color, linewidth=2, label=f"{label} (calibrated mean)", zorder=5)
    ax.fill_between(
        T,
        np.percentile(alpha_samples, 2.5, axis=0) * 1e6,
        np.percentile(alpha_samples, 97.5, axis=0) * 1e6,
        color=color,
        alpha=0.2,
        label='95% Credible Interval',
    )
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_title(
        f"{label} Model\nβ₀={theta_mean[0]:.3f}, β₁={theta_mean[1]:.3f}",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel('Thermal Diffusivity α (10⁻⁶ m²/s)', fontsize=12)
axes[-1].legend(fontsize=9, loc="upper left")

plt.tight_layout()
output_path = ROOT / "outputs/calibrated_vs_ulamec.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")
plt.close()

# Compute RMSE
rmse_powerlaw = np.sqrt(np.mean((alpha_powerlaw_calib - alpha_ulamec)**2))
rmse_exponential = np.sqrt(np.mean((alpha_exponential_calib - alpha_ulamec)**2))
rmse_logarithmic = np.sqrt(np.mean((alpha_logarithmic_calib - alpha_ulamec)**2))

print(f"\nRMSE vs Ulamec:")
print(f"  Powerlaw: {rmse_powerlaw:.2e} m²/s")
print(f"  Exponential: {rmse_exponential:.2e} m²/s")
print(f"  Logarithmic: {rmse_logarithmic:.2e} m²/s")
