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
posterior_powerlaw = np.load(ROOT / "data/posterior_powerlaw.npy")
posterior_exponential = np.load(ROOT / "data/posterior_exponential.npy")

# Get mean calibrated parameters
theta_powerlaw_mean = posterior_powerlaw.mean(axis=0)
theta_exponential_mean = posterior_exponential.mean(axis=0)

print(f"Powerlaw calibrated: β₀={theta_powerlaw_mean[0]:.3f}, β₁={theta_powerlaw_mean[1]:.3f}")
print(f"Exponential calibrated: β₀={theta_exponential_mean[0]:.3f}, β₁={theta_exponential_mean[1]:.4f}")

# Temperature range
T = np.linspace(80, 273, 200)

# Compute alpha for each model
alpha_ulamec = k_ice_ulamec(T) / (rho_ice_ulamec(T) * cp_ice_ulamec(T))
alpha_powerlaw_calib = alpha_model(T, theta=theta_powerlaw_mean, model="powerlaw")
alpha_exponential_calib = alpha_model(T, theta=theta_exponential_mean, model="exponential")

# Sample from posterior for uncertainty bands
n_samples = 100
idx = np.random.choice(len(posterior_powerlaw), n_samples, replace=False)

alpha_powerlaw_samples = np.array([
    alpha_model(T, theta=posterior_powerlaw[i], model="powerlaw") 
    for i in idx
])
alpha_exponential_samples = np.array([
    alpha_model(T, theta=posterior_exponential[i], model="exponential") 
    for i in idx
])

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Powerlaw model
ax1.plot(T, alpha_ulamec * 1e6, 'k-', linewidth=2, label='Ulamec (2007) - Truth', zorder=10)
ax1.plot(T, alpha_powerlaw_calib * 1e6, 'r-', linewidth=2, label='Powerlaw (calibrated mean)', zorder=5)
for i in range(n_samples):
    ax1.plot(T, alpha_powerlaw_samples[i] * 1e6, 'r-', alpha=0.05, linewidth=0.5)
ax1.fill_between(T, 
                  np.percentile(alpha_powerlaw_samples, 2.5, axis=0) * 1e6,
                  np.percentile(alpha_powerlaw_samples, 97.5, axis=0) * 1e6,
                  color='red', alpha=0.2, label='95% Credible Interval')
ax1.set_xlabel('Temperature (K)', fontsize=12)
ax1.set_ylabel('Thermal Diffusivity α (10⁻⁶ m²/s)', fontsize=12)
ax1.set_title(f'Powerlaw Model\nβ₀={theta_powerlaw_mean[0]:.3f}, β₁={theta_powerlaw_mean[1]:.3f}', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Exponential model
ax2.plot(T, alpha_ulamec * 1e6, 'k-', linewidth=2, label='Ulamec (2007) - Truth', zorder=10)
ax2.plot(T, alpha_exponential_calib * 1e6, 'b-', linewidth=2, label='Exponential (calibrated mean)', zorder=5)
for i in range(n_samples):
    ax2.plot(T, alpha_exponential_samples[i] * 1e6, 'b-', alpha=0.05, linewidth=0.5)
ax2.fill_between(T, 
                  np.percentile(alpha_exponential_samples, 2.5, axis=0) * 1e6,
                  np.percentile(alpha_exponential_samples, 97.5, axis=0) * 1e6,
                  color='blue', alpha=0.2, label='95% Credible Interval')
ax2.set_xlabel('Temperature (K)', fontsize=12)
ax2.set_ylabel('Thermal Diffusivity α (10⁻⁶ m²/s)', fontsize=12)
ax2.set_title(f'Exponential Model\nβ₀={theta_exponential_mean[0]:.3f}, β₁={theta_exponential_mean[1]:.5f}', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_path = ROOT / "outputs/calibrated_vs_ulamec.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")
plt.close()

# Compute RMSE
rmse_powerlaw = np.sqrt(np.mean((alpha_powerlaw_calib - alpha_ulamec)**2))
rmse_exponential = np.sqrt(np.mean((alpha_exponential_calib - alpha_ulamec)**2))

print(f"\nRMSE vs Ulamec:")
print(f"  Powerlaw: {rmse_powerlaw:.2e} m²/s")
print(f"  Exponential: {rmse_exponential:.2e} m²/s")
