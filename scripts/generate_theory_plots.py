#!/usr/bin/env python
"""Generate alpha model comparison plot for THEORY.md"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path
ROOT = Path.cwd()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cf4dt.forward import (
    alpha_model, 
    k_ice_ulamec, 
    rho_ice_ulamec, 
    cp_ice_ulamec
)

# Temperature range (K)
T = np.linspace(80, 273.15, 200)

# Compute Ulamec alpha
k = k_ice_ulamec(T)
rho = rho_ice_ulamec(T)
cp = cp_ice_ulamec(T)
alpha_ulamec = k / (rho * cp)

# Sample powerlaw and exponential from prior ranges
np.random.seed(42)
theta_powerlaw_samples = np.column_stack([
    np.random.uniform(-12, -6, 50),
    np.random.uniform(-4, 4, 50)
])
theta_exp_samples = np.column_stack([
    np.random.uniform(-12, -6, 50),
    np.random.uniform(-0.03, 0.03, 50)
])

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Powerlaw
ax = axes[0]
ax.loglog(T, alpha_ulamec, 'k-', linewidth=3, label='Ulamec (2007)', zorder=10)
for theta in theta_powerlaw_samples:
    alpha_pw = alpha_model(T, theta, model='powerlaw', T0=200.0)
    ax.loglog(T, alpha_pw, 'b-', alpha=0.1, linewidth=0.5)
ax.set_xlabel('Temperature (K)', fontsize=12)
ax.set_ylabel('α(T) [m²/s]', fontsize=12)
ax.set_title('Powerlaw Model: $α(T; θ) = \\exp(β_0)(T/200)^{β_1}$', fontsize=12)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)

# Exponential
ax = axes[1]
ax.loglog(T, alpha_ulamec, 'k-', linewidth=3, label='Ulamec (2007)', zorder=10)
for theta in theta_exp_samples:
    alpha_exp = alpha_model(T, theta, model='exponential', T0=200.0)
    ax.loglog(T, alpha_exp, 'r-', alpha=0.1, linewidth=0.5)
ax.set_xlabel('Temperature (K)', fontsize=12)
ax.set_ylabel('α(T) [m²/s]', fontsize=12)
ax.set_title('Exponential Model: $α(T; θ) = \\exp(β_0 + β_1(T-200))$', fontsize=12)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/alpha_models_prior_samples.png', dpi=200, bbox_inches='tight')
print("Saved: outputs/alpha_models_prior_samples.png")
plt.close()

print("\nAlpha models comparison plot generated successfully!")
