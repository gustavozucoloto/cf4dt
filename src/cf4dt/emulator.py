"""GP emulator training utilities for toy forward model."""

import os
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from multiprocessing import Pool
import joblib

from .forward import compute_Qlc


def _compute_single_training_point(args):
    """Worker function for parallel emulator training."""
    from mpi4py import MPI
    
    W_mph, W_mps, Ts, th, model_name, solver_kwargs, counter, total = args
    
    Q = compute_Qlc(
        W=float(W_mps),
        Ts=float(Ts),
        material_model=model_name,
        theta=(float(th[0]), float(th[1])),
        comm=MPI.COMM_SELF,
        **solver_kwargs,
    )
    
    Q_kW = Q / 1000.0  # Always convert W to kW
    
    if counter % 50 == 0:
        print(f"  [{model_name}] Progress: {counter}/{total}", flush=True)
    
    return [W_mph, Ts, th[0], th[1]], Q_kW


def lhs(n, d, seed=0):
    rng = np.random.default_rng(seed)
    u = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        u[:, j] = (perm + rng.random(n)) / n
    return u


def sample_theta(n_theta, model_name, seed=0):
    u = lhs(n_theta, 2, seed=seed)
    if model_name == "powerlaw":
        # β₀ ∈ [-14.5, -13.5]: tight range around Ulamec fit (-14.0)
        # β₁ ∈ [0.5, 1.7]: tight range around Ulamec fit (1.1)
        beta0 = -14.5 + 1.0 * u[:, 0]  # β₀ ∈ [-14.5, -13.5]
        beta1 = 0.5 + 1.2 * u[:, 1]     # β₁ ∈ [0.5, 1.7]
    elif model_name == "exponential":
        # β₀ ∈ [-14.5, -13.5]: tight range around Ulamec fit (-14.0)
        # β₁ ∈ [0.002, 0.010]: tight range around Ulamec fit (0.006)
        beta0 = -14.5 + 1.0 * u[:, 0]  # β₀ ∈ [-14.5, -13.5]
        beta1 = 0.002 + 0.008 * u[:, 1]   # β₁ ∈ [0.002, 0.010]
    else:
        raise ValueError(model_name)
    return np.column_stack([beta0, beta1])


def build_training_set(
    df_obs,
    model_name,
    n_theta=40,
    seed=1,
    use_subset_points=64,
    solver_kwargs=None,
    n_jobs=1,  # Number of parallel processes (1=serial)
    idx=None,
):
    """
    Build GP training set with optional parallel execution.
    
    Parameters
    ----------
    n_jobs : int
        Number of parallel processes. Set to 1 for serial execution.
    """
    if solver_kwargs is None:
        solver_kwargs = {}

    if idx is None:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(df_obs))
        if use_subset_points is not None and use_subset_points < len(df_obs):
            idx = rng.choice(idx, size=use_subset_points, replace=False)

    W_pts = df_obs.loc[idx, "W_mph"].to_numpy()
    Ts_pts = df_obs.loc[idx, "Ts_K"].to_numpy()

    thetas = sample_theta(n_theta, model_name, seed=seed + 100)

    # Prepare all arguments
    args_list = []
    counter = 0
    total = n_theta * len(idx)

    for th in thetas:
        for W_mph, Ts in zip(W_pts, Ts_pts):
            counter += 1
            W_mps = W_mph / 3600.0
            args_list.append((W_mph, W_mps, Ts, th, model_name, solver_kwargs, counter, total))

    # Parallel or serial execution
    if n_jobs > 1:
        print(f"[{model_name}] Pre-compiling FEniCS forms to avoid race conditions...")
        # Run one computation in main process to populate JIT cache
        _compute_single_training_point(args_list[0])
        print(f"[{model_name}] Building training set with {n_jobs} parallel processes...")
        with Pool(processes=n_jobs) as pool:
            results = pool.map(_compute_single_training_point, args_list)
    else:
        print(f"[{model_name}] Building training set serially...")
        results = [_compute_single_training_point(args) for args in args_list]

    # Extract X and y
    X_list = [r[0] for r in results]
    y_list = [r[1] for r in results]

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    return X, y


def fit_gp(X, y):
    xscaler = StandardScaler()
    yscaler = StandardScaler()

    Xs = xscaler.fit_transform(X)
    ys = yscaler.fit_transform(y.reshape(-1, 1)).ravel()

    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(
        length_scale=np.ones(X.shape[1]),
        length_scale_bounds=(1e-2, 1e2),
    ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,
        n_restarts_optimizer=3,
        random_state=0,
    )
    gp.fit(Xs, ys)
    return gp, xscaler, yscaler


def train_and_save_emulator(
    df_path,
    model_name,
    out_path,
    n_theta=40,
    seed=1,
    use_subset_points=64,
    solver_kwargs=None,
    n_jobs=1,  # Number of parallel processes (1=serial)
):
    """
    Train GP emulator with optional parallel execution.
    
    Parameters
    ----------
    n_jobs : int
        Number of parallel processes. Set to 1 for serial execution.
    """
    df = pd.read_csv(df_path)

    if solver_kwargs is None:
        solver_kwargs = {}

    X, y = build_training_set(
        df,
        model_name,
        n_theta=n_theta,
        use_subset_points=use_subset_points,
        seed=seed,
        solver_kwargs=solver_kwargs,
        n_jobs=n_jobs,
    )

    gp, xscaler, yscaler = fit_gp(X, y)

    bundle = dict(gp=gp, xscaler=xscaler, yscaler=yscaler, model=model_name)
    joblib.dump(bundle, out_path)
    print(f"Saved GP emulator: {out_path}")
    print("Kernel:", gp.kernel_)


def validate_emulator_kfold(
    df_path,
    model_name,
    n_splits=5,
    n_theta=40,
    seed=1,
    use_subset_points=64,
    solver_kwargs=None,
    n_jobs=1,
    save_dir=None,
):
    """
    Validate GP emulator with K-fold cross-validation on the training set.

    Returns a dict with per-fold metrics and aggregate statistics.
    """
    df = pd.read_csv(df_path)

    if solver_kwargs is None:
        solver_kwargs = {}

    X, y = build_training_set(
        df,
        model_name,
        n_theta=n_theta,
        seed=seed,
        use_subset_points=use_subset_points,
        solver_kwargs=solver_kwargs,
        n_jobs=n_jobs,
    )

    base_idx = np.arange(len(X))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics = []

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for fold_idx, (train_sel, test_sel) in enumerate(kf.split(X), start=1):
        train_idx = base_idx[train_sel]
        test_idx = base_idx[test_sel]

        X_train, y_train = X[train_sel], y[train_sel]
        X_test, y_test = X[test_sel], y[test_sel]

        gp, xscaler, yscaler = fit_gp(X_train, y_train)
        Xs_test = xscaler.transform(X_test)  
        y_pred_s = gp.predict(Xs_test)
        y_pred = yscaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

        rmse = float(mean_squared_error(y_test, y_pred, squared=False))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        fold_metrics.append(
            dict(fold=fold_idx, rmse=rmse, mae=mae, r2=r2, n=len(test_idx))
        )

        if save_dir:
            bundle = dict(
                gp=gp,
                xscaler=xscaler,
                yscaler=yscaler,
                model=model_name,
                fold=fold_idx,
                base_idx=base_idx,
                train_sel=train_sel,
                test_sel=test_sel,
                train_idx=train_idx,
                test_idx=test_idx,
                X_test=X_test,
                y_test=y_test,
            )
            out_path = os.path.join(save_dir, f"gp_{model_name}_fold{fold_idx:02d}.joblib")
            joblib.dump(bundle, out_path)

    rmses = np.array([m["rmse"] for m in fold_metrics])
    maes = np.array([m["mae"] for m in fold_metrics])
    r2s = np.array([m["r2"] for m in fold_metrics])

    summary = dict(
        rmse_mean=float(rmses.mean()),
        rmse_std=float(rmses.std(ddof=1)) if len(rmses) > 1 else 0.0,
        mae_mean=float(maes.mean()),
        mae_std=float(maes.std(ddof=1)) if len(maes) > 1 else 0.0,
        r2_mean=float(r2s.mean()),
        r2_std=float(r2s.std(ddof=1)) if len(r2s) > 1 else 0.0,
        n_splits=n_splits,
        n_samples=len(X),
    )

    return dict(folds=fold_metrics, summary=summary)


