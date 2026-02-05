"""Synthetic dataset generation for Q_lc using the full FEniCSx model."""

import numpy as np
import pandas as pd
from multiprocessing import Pool

from .design import sample_design
from .noise import add_noise_kW
from .forward import compute_Qlc


def _compute_single_point(args):
    """Worker function for parallel data generation."""
    from mpi4py import MPI  # Import inside worker
    
    i, W, Ts, Ro, L, Tm, Rinf, num_cells, p_grade, num_steps, dt_ratio, n_pts = args
    
    Q = compute_Qlc(
        W=float(W),
        Ts=float(Ts),
        material_model="ulamec",
        Ro=Ro,
        L=L,
        Tm=Tm,
        Rinf=Rinf,
        num_cells=num_cells,
        p_grade=p_grade,
        num_steps=num_steps,
        dt_ratio=dt_ratio,
        comm=MPI.COMM_SELF,  # Each worker uses its own communicator
    )
    
    if (i + 1) % 10 == 0:
        print(f"  Progress: {i + 1}/{n_pts}", flush=True)
    
    return i, Q


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
    dt_ratio=1.03,
    n_jobs=1,  # Number of parallel processes (1=serial)
):
    """
    Generate artificial dataset with optional parallel execution.
    
    Parameters
    ----------
    n_jobs : int
        Number of parallel processes. Set to 1 for serial execution.
        Recommended: number of CPU cores for large datasets.
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

    # Prepare arguments for all evaluations
    args_list = [
        (i, W_all[i], Ts_all[i], Ro, L, Tm, Rinf, num_cells, p_grade,
         num_steps, dt_ratio, n_pts)
        for i in range(n_pts)
    ]

    # Parallel or serial execution
    if n_jobs > 1:
        print(f"Running {n_pts} simulations with {n_jobs} parallel processes...")
        with Pool(processes=n_jobs) as pool:
            results = pool.map(_compute_single_point, args_list)
    else:
        print(f"Running {n_pts} simulations serially...")
        results = [_compute_single_point(args) for args in args_list]
    
    # Extract results in correct order
    Q_true_W = np.zeros(n_pts)
    for idx, Q in results:
        Q_true_W[idx] = Q

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
