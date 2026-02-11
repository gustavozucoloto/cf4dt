"""
Uncertainty quantification utilities using GP emulator and posterior samples.

Produces TWO kinds of uncertainty bands for Q_lc:
1) Parameter-only (credible band for the mean): varies ONLY with posterior parameters
2) Full posterior predictive: parameters + GP predictive uncertainty + optional observation noise

alpha(T) band remains parameter uncertainty only (posterior draws through alpha_model).
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt

from .gp_utils import gp_predict
from .forward import alpha_model, k_ice_ulamec, rho_ice_ulamec, cp_ice_ulamec


def uq_maps(
    model_name,
    gp_path,
    posterior_path,
    out_prefix="uq",
    n_post=400,
    seed=0,
    sigma_obs=0.1,     # set to measurement noise std (kW) 
    n_rep=10,          # predictive draws per posterior sample (smooth quantiles)
    gp_second="var",   # "var" if gp_predict returns variance; "std" if it returns std
    data_csv=None,
    use_data_ts=True,
    band_alpha=0.45,
):
    def finalize_plot(ax):
        handles, labels = ax.get_legend_handles_labels()
        if "Artificial data" in labels:
            idx = labels.index("Artificial data")
            handles = handles[:idx] + handles[idx + 1 :] + [handles[idx]]
            labels = labels[:idx] + labels[idx + 1 :] + [labels[idx]]
        ax.legend(handles, labels, loc="best")
    bundle = joblib.load(gp_path)
    samples_all = np.load(posterior_path)

    rng = np.random.default_rng(seed)
    if len(samples_all) > n_post:
        idx = rng.choice(len(samples_all), size=n_post, replace=False)
        samples = samples_all[idx]
    else:
        samples = samples_all

    W_mph_grid = np.linspace(0.05, 5.0, 50)
    Ts_grid = np.linspace(80, 260, 30)

    # Parameter-only (credible for mean)
    Q_mean_med = np.zeros((len(Ts_grid), len(W_mph_grid)))
    Q_mean_lo  = np.zeros_like(Q_mean_med)
    Q_mean_hi  = np.zeros_like(Q_mean_med)

    # Full posterior predictive (params + GP + obs noise)
    Q_pred_med = np.zeros_like(Q_mean_med)
    Q_pred_lo  = np.zeros_like(Q_mean_med)
    Q_pred_hi  = np.zeros_like(Q_mean_med)

    for iT, Ts in enumerate(Ts_grid):
        for iW, W_mph in enumerate(W_mph_grid):
            X = np.column_stack(
                [
                    np.full(len(samples), W_mph),
                    np.full(len(samples), Ts),
                    samples[:, 0],
                    samples[:, 1],
                ]
            )

            mu, s2_or_s = gp_predict(bundle, X)
            mu = np.asarray(mu).reshape(-1)
            s2_or_s = np.asarray(s2_or_s).reshape(-1)

            # Interpret GP second output
            if gp_second.lower() == "std":
                var_gp = np.maximum(s2_or_s, 0.0) ** 2
            else:  # "var"
                var_gp = np.maximum(s2_or_s, 0.0)

            # --- 1) Parameter-only band (credible for mean response) ---
            Q_mean_med[iT, iW] = np.median(mu)
            Q_mean_lo[iT, iW]  = np.percentile(mu, 2.5)
            Q_mean_hi[iT, iW]  = np.percentile(mu, 97.5)

            # --- 2) Full posterior predictive ---
            std_total = np.sqrt(var_gp + float(sigma_obs) ** 2)

            if n_rep <= 1:
                q_draw = rng.normal(mu, std_total)
            else:
                q_draw = rng.normal(mu[:, None], std_total[:, None], size=(len(mu), n_rep)).ravel()

            Q_pred_med[iT, iW] = np.median(q_draw)
            Q_pred_lo[iT, iW]  = np.percentile(q_draw, 2.5)
            Q_pred_hi[iT, iW]  = np.percentile(q_draw, 97.5)

    # ----------------------------
    # Q_lc plots (three versions)
    # ----------------------------
    data = None
    Ts_slices = [100, 160, 220, 260]
    if data_csv is not None:
        data = np.genfromtxt(data_csv, delimiter=",", names=True)
        if use_data_ts:
            Ts_slices = np.unique(data["Ts_K"]).tolist()

    # A) Parameter-only plot
    plt.figure()
    data_plotted = False
    for Ts_pick in Ts_slices:
        iT = int(np.argmin(np.abs(Ts_grid - Ts_pick)))
        plt.fill_between(W_mph_grid, Q_mean_lo[iT], Q_mean_hi[iT], alpha=band_alpha)
        plt.plot(W_mph_grid, Q_mean_med[iT], label=f"Ts={Ts_pick:.0f} K")
        if data is not None:
            mask = np.isclose(data["Ts_K"], Ts_pick)
            if np.any(mask):
                plt.scatter(
                    data["W_mph"][mask],
                    data["y_obs_kW"][mask],
                    s=18,
                    color="black",
                    alpha=0.7,
                    label="Artificial data" if not data_plotted else None,
                )
                data_plotted = True
    plt.xlabel(r"$W$ (m/hour)")
    plt.ylabel(r"$Q_{lc}$ (kW)")
    # plt.title(f"Q_lc credible bands for mean (parameter-only) ({model_name})")
    ax = plt.gca()
    finalize_plot(ax)
    plt.tight_layout()
    plt.savefig(
        f"{out_prefix}_Qlc_slices_{model_name}_param_only.svg",
        transparent=True,
    )

    # B) Full posterior predictive plot
    plt.figure()
    data_plotted = False
    for Ts_pick in Ts_slices:
        iT = int(np.argmin(np.abs(Ts_grid - Ts_pick)))
        plt.fill_between(W_mph_grid, Q_pred_lo[iT], Q_pred_hi[iT], alpha=band_alpha)
        plt.plot(W_mph_grid, Q_pred_med[iT], label=f"Ts={Ts_pick:.0f} K")
        if data is not None:
            mask = np.isclose(data["Ts_K"], Ts_pick)
            if np.any(mask):
                plt.scatter(
                    data["W_mph"][mask],
                    data["y_obs_kW"][mask],
                    s=18,
                    color="black",
                    alpha=0.7,
                    label="Artificial data" if not data_plotted else None,
                )
                data_plotted = True
    plt.xlabel(r"$W$ (m/hour)")
    plt.ylabel(r"$Q_{lc}$ (kW)")
    ax = plt.gca()
    finalize_plot(ax)
    plt.tight_layout()
    plt.savefig(
        f"{out_prefix}_Qlc_slices_{model_name}_full_predictive.svg",
        transparent=True,
    )

    # C) Combined plot (dark = param-only, light = full predictive)
    plt.figure()
    data_plotted = False
    for Ts_pick in Ts_slices:
        iT = int(np.argmin(np.abs(Ts_grid - Ts_pick)))

        # Full predictive (light)
        plt.fill_between(W_mph_grid, Q_pred_lo[iT], Q_pred_hi[iT], alpha=band_alpha * 0.6)
        # Param-only (darker)
        plt.fill_between(W_mph_grid, Q_mean_lo[iT], Q_mean_hi[iT], alpha=band_alpha)

        # Median of mean response (clean reference line)
        plt.plot(W_mph_grid, Q_mean_med[iT], label=f"Ts={Ts_pick:.0f} K")
        if data is not None:
            mask = np.isclose(data["Ts_K"], Ts_pick)
            if np.any(mask):
                plt.scatter(
                    data["W_mph"][mask],
                    data["y_obs_kW"][mask],
                    s=18,
                    color="black",
                    alpha=0.7,
                    label="Artificial data" if not data_plotted else None,
                )
                data_plotted = True

    plt.xlabel(r"$W$ (m/hour)")
    plt.ylabel(r"$Q_{lc}$ (kW)")
    # plt.title(
    #     f"Q_lc uncertainty bands ({model_name})\n"
    #     f"Dark: param-only mean | Light: full predictive (sigma_obs={sigma_obs})"
    # )
    ax = plt.gca()
    finalize_plot(ax)
    plt.tight_layout()
    plt.savefig(
        f"{out_prefix}_Qlc_slices_{model_name}_two_bands.svg",
        transparent=True,
    )

    # ----------------------------
    # alpha(T) band (parameter-only)
    # ----------------------------
    T = np.linspace(80, 273.15, 200)
    alpha_draws = np.array([alpha_model(T, th, model=model_name) for th in samples])

    a_med = np.median(alpha_draws, axis=0)
    a_lo  = np.percentile(alpha_draws, 2.5, axis=0)
    a_hi  = np.percentile(alpha_draws, 97.5, axis=0)

    alpha_ulamec = k_ice_ulamec(T) / (rho_ice_ulamec(T) * cp_ice_ulamec(T))

    plt.figure()
    plt.fill_between(T, a_lo, a_hi, alpha=0.30, label="95% Credible Interval (parameter-only)")
    plt.plot(T, a_med, label=f"{model_name.capitalize()} (median)")
    plt.plot(T, alpha_ulamec, "k--", linewidth=2, label="Ulamec (2007) - Truth")
    plt.yscale("log")
    plt.xlabel(r"$T$ (K)")
    plt.ylabel(r"$\alpha(T)$ (m$^2$/s)")
    # plt.title(f"Posterior alpha(T) band ({model_name})")
    ax = plt.gca()
    finalize_plot(ax)
    plt.tight_layout()
    plt.savefig(
        f"{out_prefix}_alpha_band_{model_name}.svg",
        transparent=True,
    )

    return {
        "W_mph_grid": W_mph_grid,
        "Ts_grid": Ts_grid,
        "Q_mean_med": Q_mean_med,
        "Q_mean_lo": Q_mean_lo,
        "Q_mean_hi": Q_mean_hi,
        "Q_pred_med": Q_pred_med,
        "Q_pred_lo": Q_pred_lo,
        "Q_pred_hi": Q_pred_hi,
        "T_alpha": T,
        "alpha_med": a_med,
        "alpha_lo": a_lo,
        "alpha_hi": a_hi,
        "alpha_ulamec": alpha_ulamec,
    }
