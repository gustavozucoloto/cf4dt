"""Synthetic dataset generation for Q_lc using the full FEniCSx model."""

import numpy as np
import pandas as pd

from .design import sample_design
from .noise import add_noise_kW
from .forward import compute_Qlc


def generate_artificial_data(
    out_csv="artificial_Qlc_data.csv",
    n_lhs=120,
    seed_design=1,
    seed_noise=2,
    sigma_kW=0.1,
    Ts_min=80.0,
    Ts_max=260.0,
    W_mph_min=0.05,
    W_mph_max=10.0,
    Ro=0.1,
    L=3.7,
    Tm=273.15,
    Rinf_offset=1.0,
    num_cells=400,
    p_grade=3.0,
    num_steps=1000,
    dt_ratio=1.03
):
    """
    Generate synthetic Q_lc observations; runs in serial.
    """
    W_all, Ts_all = sample_design(
        n_lhs=n_lhs,
        seed=seed_design,
        Ts_min=Ts_min,
        Ts_max=Ts_max,
        W_mph_min=W_mph_min,
        W_mph_max=W_mph_max,
    )
    n_pts = len(W_all)

    Rinf = Ro + float(Rinf_offset)

    Q_true_W = np.full(n_pts, np.nan, dtype=float)

    for i in range(n_pts):
        W = float(W_all[i])
        Ts = float(Ts_all[i])

        Qlc_W = compute_Qlc(
            W=W,
            Ts=Ts,
            material_model="ulamec",
            Ro=Ro,
            L=L,
            Tm=Tm,
            Rinf=Rinf,
            num_cells=num_cells,
            p_grade=p_grade,
        )

        Q_true_W[i] = Qlc_W

        if (i + 1) % 10 == 0:
            print(f"Computed {i + 1} / {n_pts} points")

    if np.any(np.isnan(Q_true_W)):
        missing = np.where(np.isnan(Q_true_W))[0]
        raise RuntimeError(f"Missing Q_true at indices: {missing.tolist()}")

    y_obs_kW = add_noise_kW(Q_true_W, sigma_kW=sigma_kW, seed=seed_noise)

    df = pd.DataFrame(
        {
            "W_mps": W_all,
            "W_mph": W_all * 3600.0,
            "Ts_K": Ts_all,
            "Ro_m": Ro,
            "Rinf_m": Rinf,
            "L_m": L,
            "Tm_K": Tm,
            "Qlc_true_W": Q_true_W,
            "Qlc_true_kW": Q_true_W / 1000.0,
            "y_obs_kW": y_obs_kW,
            "sigma_kW": sigma_kW,
            "num_cells": num_cells,
            "p_grade": p_grade,
            "seed_design": seed_design,
            "seed_noise": seed_noise,
        }
    )

    df.to_csv(out_csv, index=False)
    print(f"\nSaved dataset with {len(df)} points to: {out_csv}")
    print(
        "Qlc_true_kW: min/median/max =",
        float(df["Qlc_true_kW"].min()),
        float(df["Qlc_true_kW"].median()),
        float(df["Qlc_true_kW"].max()),
    )
