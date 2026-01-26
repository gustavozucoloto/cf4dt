"""GP emulator training utilities for toy forward model."""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
import joblib

from .forward import compute_Qlc


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
        beta0 = -12 + 6 * u[:, 0]
        beta1 = -4 + 8 * u[:, 1]
    elif model_name == "exponential":
        beta0 = -12 + 6 * u[:, 0]
        beta1 = -0.03 + 0.06 * u[:, 1]
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
):
    if solver_kwargs is None:
        solver_kwargs = {}

    rng = np.random.default_rng(seed)
    idx = np.arange(len(df_obs))
    if use_subset_points is not None and use_subset_points < len(df_obs):
        idx = rng.choice(idx, size=use_subset_points, replace=False)

    W_pts = df_obs.loc[idx, "W_mps"].to_numpy()
    Ts_pts = df_obs.loc[idx, "Ts_K"].to_numpy()

    thetas = sample_theta(n_theta, model_name, seed=seed + 100)

    X_list, y_list = [], []
    total = n_theta * len(idx)
    counter = 0

    for th in thetas:
        for W, Ts in zip(W_pts, Ts_pts):
            Q = compute_Qlc(
                W=float(W),
                Ts=float(Ts),
                material_model=model_name,
                theta=(float(th[0]), float(th[1])),
                **solver_kwargs,
            )

            Q_kW = Q / 1000.0 if Q > 1000 else Q

            X_list.append([W, Ts, th[0], th[1]])
            y_list.append(Q_kW)

            counter += 1
            if counter % 50 == 0:
                print(f"[{model_name}] training evals: {counter}/{total}")

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    return X, y


def fit_gp(X, y):
    xscaler = StandardScaler()
    yscaler = StandardScaler()

    Xs = xscaler.fit_transform(X)
    ys = yscaler.fit_transform(y.reshape(-1, 1)).ravel()

    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=np.ones(X.shape[1]),
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5,
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
):
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
    )

    gp, xscaler, yscaler = fit_gp(X, y)

    bundle = dict(gp=gp, xscaler=xscaler, yscaler=yscaler, model=model_name)
    joblib.dump(bundle, out_path)
    print(f"Saved GP emulator: {out_path}")
    print("Kernel:", gp.kernel_)
