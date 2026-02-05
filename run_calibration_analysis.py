#!/usr/bin/env python
"""
Calibration Quality Evaluation Script
Runs all analysis and saves results to files
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import joblib

# Add src to path
ROOT = Path.cwd()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cf4dt.gp_utils import gp_predict
from cf4dt.forward import alpha_model, k_ice_ulamec, rho_ice_ulamec, cp_ice_ulamec
from cf4dt.emulator import build_training_set

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-jobs", type=int, default=1, help="Number of parallel processes (1=serial)")
    return p.parse_args()

args = parse_args()

print("="*70)
print("CALIBRATION QUALITY EVALUATION")
print(f"Using {args.n_jobs} parallel processes")
print("="*70)

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n[1/8] Loading data...")
df_truth = pd.read_csv("data/artificial_Qlc_data.csv")
posterior_pl = np.load("data/posterior_powerlaw.npy")
posterior_exp = np.load("data/posterior_exponential.npy")

print(f"Artificial data shape: {df_truth.shape}")
print(f"Posterior powerlaw shape: {posterior_pl.shape}")
print(f"Posterior exponential shape: {posterior_exp.shape}")

# ============================================================================
# 2. Load GP Models
# ============================================================================
print("\n[2/8] Loading GP emulators...")
bundle_pl = joblib.load("data/gp_powerlaw.joblib")
bundle_exp = joblib.load("data/gp_exponential.joblib")

# Regenerate training data from the artificial data
X_train_pl, y_train = build_training_set(
    df_truth,
    model_name="powerlaw",
    n_theta=40,
    seed=1,
    use_subset_points=64,
    n_jobs=args.n_jobs,
)
X_train_exp, _ = build_training_set(
    df_truth,
    model_name="exponential",
    n_theta=40,
    seed=1,
    use_subset_points=64,
    n_jobs=args.n_jobs,
)

mu_pl, std_pl = gp_predict(bundle_pl, X_train_pl)
mu_exp, std_exp = gp_predict(bundle_exp, X_train_exp)

print(f"Powerlaw GP: {X_train_pl.shape[0]} training points")
print(f"Exponential GP: {X_train_exp.shape[0]} training points")

# ============================================================================
# 3. Plot GP vs Data
# ============================================================================
print("\n[3/8] Plotting GP vs. artificial data...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Powerlaw
ax = axes[0]
sort_idx_pl = np.argsort(mu_pl)
ax.scatter(np.arange(len(y_train)), y_train[sort_idx_pl], alpha=0.6, s=30, 
           label="Artificial data", color="C1")
ax.plot(np.arange(len(y_train)), mu_pl[sort_idx_pl], 'C0-', lw=2, label="GP mean")
ax.fill_between(np.arange(len(y_train)), 
                 (mu_pl - 2*std_pl)[sort_idx_pl], 
                 (mu_pl + 2*std_pl)[sort_idx_pl], 
                 alpha=0.3, label="95% PI")
ax.set_xlabel("Sample index (sorted)")
ax.set_ylabel("$Q_{lc}$ (kW)")
ax.set_title("Powerlaw: GP vs. Data")
ax.legend()
ax.grid(True, alpha=0.3)

# Exponential
ax = axes[1]
sort_idx_exp = np.argsort(mu_exp)
ax.scatter(np.arange(len(y_train)), y_train[sort_idx_exp], alpha=0.6, s=30, 
           label="Artificial data", color="C1")
ax.plot(np.arange(len(y_train)), mu_exp[sort_idx_exp], 'C2-', lw=2, label="GP mean")
ax.fill_between(np.arange(len(y_train)), 
                 (mu_exp - 2*std_exp)[sort_idx_exp], 
                 (mu_exp + 2*std_exp)[sort_idx_exp], 
                 alpha=0.3, color='C2', label="95% PI")
ax.set_xlabel("Sample index (sorted)")
ax.set_ylabel("$Q_{lc}$ (kW)")
ax.set_title("Exponential: GP vs. Data")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/01_gp_vs_data.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved: outputs/01_gp_vs_data.png")

# ============================================================================
# 4. Calculate GP Metrics
# ============================================================================
print("\n[4/8] Computing emulator metrics...")
rmse_pl = np.sqrt(np.mean((y_train - mu_pl)**2))
mae_pl = np.mean(np.abs(y_train - mu_pl))
r2_pl = 1 - np.sum((y_train - mu_pl)**2) / np.sum((y_train - y_train.mean())**2)

rmse_exp = np.sqrt(np.mean((y_train - mu_exp)**2))
mae_exp = np.mean(np.abs(y_train - mu_exp))
r2_exp = 1 - np.sum((y_train - mu_exp)**2) / np.sum((y_train - y_train.mean())**2)

coverage_pl = np.mean((y_train >= mu_pl - 2*std_pl) & (y_train <= mu_pl + 2*std_pl))
coverage_exp = np.mean((y_train >= mu_exp - 2*std_exp) & (y_train <= mu_exp + 2*std_exp))

print(f"\nPowerlaw:")
print(f"    RMSE: {rmse_pl:.4f} kW")
print(f"    R²:   {r2_pl:.6f}")
print(f"    Coverage 95%: {coverage_pl:.1%}")

print(f"\nExponential:")
print(f"    RMSE: {rmse_exp:.4f} kW")
print(f"    R²:   {r2_exp:.6f}")
print(f"    Coverage 95%: {coverage_exp:.1%}")

# ============================================================================
# 5. Compare α(T) Models
# ============================================================================
print("\n[5/8] Comparing α(T) models...")
T_range = np.linspace(80, 273.15, 200)
theta_median_pl = np.median(posterior_pl, axis=0)
theta_median_exp = np.median(posterior_exp, axis=0)

def alpha_ulamec(T):
    """Alpha effective from Ulamec model: alpha = k/(rho*cp)"""
    k = k_ice_ulamec(T)
    rho = rho_ice_ulamec(T)
    cp = cp_ice_ulamec(T)
    return k / (rho * cp)

alpha_ulamec_vals = np.array([alpha_ulamec(t) for t in T_range])
alpha_pl_vals = alpha_model(T_range, theta_median_pl, model='powerlaw')
alpha_exp_vals = alpha_model(T_range, theta_median_exp, model='exponential')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(T_range, alpha_ulamec_vals, 'k-', lw=2.5, label="Ulamec (truth)")
ax.plot(T_range, alpha_pl_vals, 'C0--', lw=2, label="Powerlaw (calibrated)")
ax.plot(T_range, alpha_exp_vals, 'C2-.', lw=2, label="Exponential (calibrated)")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("α(T) (m²/s)")
ax.set_title("Effective Thermal Conductivity: Model Comparison")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1]
err_pl_rel = np.abs(alpha_pl_vals - alpha_ulamec_vals) / (alpha_ulamec_vals + 1e-10) * 100
err_exp_rel = np.abs(alpha_exp_vals - alpha_ulamec_vals) / (alpha_ulamec_vals + 1e-10) * 100
ax.semilogy(T_range, err_pl_rel, 'C0--', lw=2, label="Powerlaw")
ax.semilogy(T_range, err_exp_rel, 'C2-.', lw=2, label="Exponential")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Relative Error (%)")
ax.set_title("Relative Error vs. Ulamec")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig("outputs/02_alpha_model_comparison.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved: outputs/02_alpha_model_comparison.png")

# ============================================================================
# 6. UQ Analysis
# ============================================================================
print("\n[6/8] Generating UQ predictions...")
n_samples_uq = min(200, len(posterior_pl))
n_scenarios = 20

W_range = np.logspace(np.log10(0.05), np.log10(5.0), n_scenarios) / 3600.0
Ts_range = np.linspace(100, 260, n_scenarios)

np.random.seed(42)
idx_pl = np.random.choice(len(posterior_pl), size=min(n_samples_uq, len(posterior_pl)), replace=False)
idx_exp = np.random.choice(len(posterior_exp), size=min(n_samples_uq, len(posterior_exp)), replace=False)

samples_pl = posterior_pl[idx_pl]
samples_exp = posterior_exp[idx_exp]

Q_med_pl = np.zeros((len(Ts_range), len(W_range)))
Q_lower_pl = np.zeros_like(Q_med_pl)
Q_upper_pl = np.zeros_like(Q_med_pl)

Q_med_exp = np.zeros((len(Ts_range), len(W_range)))
Q_lower_exp = np.zeros_like(Q_med_exp)
Q_upper_exp = np.zeros_like(Q_med_exp)

for i_ts, Ts in enumerate(Ts_range):
    for i_w, W in enumerate(W_range):
        # Powerlaw
        X_pl = np.column_stack([
            np.full(len(samples_pl), W),
            np.full(len(samples_pl), Ts),
            samples_pl[:, 0],
            samples_pl[:, 1]
        ])
        mu_pl_uq, std_pl_uq = gp_predict(bundle_pl, X_pl)
        Q_med_pl[i_ts, i_w] = np.median(mu_pl_uq)
        Q_lower_pl[i_ts, i_w] = np.percentile(mu_pl_uq, 2.5)
        Q_upper_pl[i_ts, i_w] = np.percentile(mu_pl_uq, 97.5)
        
        # Exponential
        X_exp = np.column_stack([
            np.full(len(samples_exp), W),
            np.full(len(samples_exp), Ts),
            samples_exp[:, 0],
            samples_exp[:, 1]
        ])
        mu_exp_uq, std_exp_uq = gp_predict(bundle_exp, X_exp)
        Q_med_exp[i_ts, i_w] = np.median(mu_exp_uq)
        Q_lower_exp[i_ts, i_w] = np.percentile(mu_exp_uq, 2.5)
        Q_upper_exp[i_ts, i_w] = np.percentile(mu_exp_uq, 97.5)

print(f"Generated predictions for {len(Ts_range)}×{len(W_range)} scenarios")

# ============================================================================
# 7. UQ Diagnostics
# ============================================================================
print("\n[7/8] Plotting UQ diagnostics...")
ci_width_pl = Q_upper_pl - Q_lower_pl
ci_width_exp = Q_upper_exp - Q_lower_exp

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Slices at different Ts
W_mph = W_range * 3600
Ts_select = [120, 160, 220, 260]

# Powerlaw
ax = axes[0, 0]
for Ts_pick in Ts_select:
    i_ts = np.argmin(np.abs(Ts_range - Ts_pick))
    ax.fill_between(W_mph, Q_lower_pl[i_ts], Q_upper_pl[i_ts], alpha=0.2)
    ax.plot(W_mph, Q_med_pl[i_ts], marker='o', label=f"Ts={Ts_range[i_ts]:.0f} K", markersize=4)
ax.set_xscale('log')
ax.set_xlabel("W (m/h)")
ax.set_ylabel("$Q_{lc}$ (kW)")
ax.set_title("Powerlaw: Posterior Predictive Q_lc Slices")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Exponential
ax = axes[0, 1]
for Ts_pick in Ts_select:
    i_ts = np.argmin(np.abs(Ts_range - Ts_pick))
    ax.fill_between(W_mph, Q_lower_exp[i_ts], Q_upper_exp[i_ts], alpha=0.2)
    ax.plot(W_mph, Q_med_exp[i_ts], marker='o', label=f"Ts={Ts_range[i_ts]:.0f} K", markersize=4)
ax.set_xscale('log')
ax.set_xlabel("W (m/h)")
ax.set_ylabel("$Q_{lc}$ (kW)")
ax.set_title("Exponential: Posterior Predictive Q_lc Slices")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# CI width distribution
ax = axes[1, 0]
ax.hist(ci_width_pl.flatten(), bins=30, alpha=0.6, label="Powerlaw", density=True)
ax.hist(ci_width_exp.flatten(), bins=30, alpha=0.6, label="Exponential", density=True)
ax.set_xlabel("95% CI Width (kW)")
ax.set_ylabel("Density")
ax.set_title("Uncertainty Distribution")
ax.legend()
ax.grid(True, alpha=0.3)

# Relative uncertainty
ax = axes[1, 1]
ratio_pl = ci_width_pl / (2 * (np.abs(Q_med_pl) + 1e-10))
ratio_exp = ci_width_exp / (2 * (np.abs(Q_med_exp) + 1e-10))
ax.scatter(Q_med_pl.flatten(), ratio_pl.flatten(), alpha=0.5, s=20, label="Powerlaw")
ax.scatter(Q_med_exp.flatten(), ratio_exp.flatten(), alpha=0.5, s=20, label="Exponential")
ax.set_xlabel("Median $Q_{lc}$ (kW)")
ax.set_ylabel("CI Width / |Median|")
ax.set_title("Relative Uncertainty vs. Magnitude")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/03_uq_diagnostics.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved: outputs/03_uq_diagnostics.png")

# ============================================================================
# 8. Summary Report
# ============================================================================
print("\n[8/8] Generating summary report...")

cv_pl = np.median(ci_width_pl / (2 * Q_med_pl + 1e-10))
cv_exp = np.median(ci_width_exp / (2 * Q_med_exp + 1e-10))

summary = f"""
{'='*70}
CALIBRATION QUALITY EVALUATION - FINAL REPORT
{'='*70}

EMULATOR QUALITY:

  Powerlaw:
    • R²:                {r2_pl:.6f}
    • RMSE:              {rmse_pl:.4f} kW
    • MAE:               {mae_pl:.4f} kW
    • 95% Coverage:      {coverage_pl:.1%}

  Exponential:
    • R²:                {r2_exp:.6f}
    • RMSE:              {rmse_exp:.4f} kW
    • MAE:               {mae_exp:.4f} kW
    • 95% Coverage:      {coverage_exp:.1%}

    ALPHA MODEL COMPARISON:

  Mean Absolute Error vs. Ulamec:
    • Powerlaw:         {np.mean(np.abs(alpha_pl_vals - alpha_ulamec_vals)):.6f}
    • Exponential:      {np.mean(np.abs(alpha_exp_vals - alpha_ulamec_vals)):.6f}

UNCERTAINTY QUANTIFICATION:

  95% Credible Interval Width (median):
    • Powerlaw:         {ci_width_pl.mean():.4f} kW
    • Exponential:      {ci_width_exp.mean():.4f} kW

  Relative Uncertainty (Coefficient of Variation):
    • Powerlaw:         {cv_pl*100:.2f}%
    • Exponential:      {cv_exp*100:.2f}%

{'='*70}
"""

print(summary)

# Save report to file
with open("outputs/CALIBRATION_REPORT.txt", "w") as f:
    f.write(summary)

print("Saved: outputs/CALIBRATION_REPORT.txt")

print("\nANALYSIS COMPLETE!")