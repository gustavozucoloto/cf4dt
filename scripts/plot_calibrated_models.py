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

# Load posterior samples (powerlaw only)
posterior_powerlaw = np.load(ROOT / "data/posterior_powerlaw.npy")

# Get mean calibrated parameters
theta_powerlaw_mean = posterior_powerlaw.mean(axis=0)

print(f"Powerlaw calibrated: β₀={theta_powerlaw_mean[0]:.3f}, β₁={theta_powerlaw_mean[1]:.3f}")

# Temperature range
T = np.linspace(80, 273, 200)

# Compute alpha for each model
alpha_ulamec = k_ice_ulamec(T) / (rho_ice_ulamec(T) * cp_ice_ulamec(T))
alpha_powerlaw_calib = alpha_model(T, theta=theta_powerlaw_mean, model="powerlaw")

# Sample from posterior for uncertainty bands
n_samples = 100
idx = np.random.choice(len(posterior_powerlaw), n_samples, replace=False)

alpha_powerlaw_samples = np.array([
    alpha_model(T, theta=posterior_powerlaw[i], model="powerlaw") 
    for i in idx
])
fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))

# Powerlaw model
ax1.plot(T, alpha_ulamec * 1e6, 'k-', linewidth=2, label='Ulamec (2007) - Truth', zorder=10)
ax1.plot(T, alpha_powerlaw_calib * 1e6, 'r-', linewidth=2, label='Powerlaw (calibrated mean)', zorder=5)
ax1.fill_between(T, 
                  np.percentile(alpha_powerlaw_samples, 2.5, axis=0) * 1e6,
                  np.percentile(alpha_powerlaw_samples, 97.5, axis=0) * 1e6,
                  color='red', alpha=0.2, label='95% Credible Interval')
ax1.set_xlabel('Temperature (K)', fontsize=12)
ax1.set_ylabel('Thermal Diffusivity α (10⁻⁶ m²/s)', fontsize=12)
ax1.set_title(f'Powerlaw Model\nβ₀={theta_powerlaw_mean[0]:.3f}, β₁={theta_powerlaw_mean[1]:.3f}', fontsize=13)
ax1.set_ylim(-1, 5)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
output_path = ROOT / "outputs/calibrated_vs_ulamec.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")
plt.close()

# Compute RMSE
rmse_powerlaw = np.sqrt(np.mean((alpha_powerlaw_calib - alpha_ulamec)**2))

print(f"\nRMSE vs Ulamec:")
print(f"  Powerlaw: {rmse_powerlaw:.2e} m²/s")
