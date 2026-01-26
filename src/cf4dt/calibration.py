"""Bayesian calibration using GP surrogate and emcee."""

import numpy as np
import pandas as pd
import joblib
import emcee

from .gp_utils import gp_predict


def log_prior(theta, model_name):
    beta0, beta1 = theta

    if model_name == "powerlaw":
        if not (-12 <= beta0 <= -6 and -4 <= beta1 <= 4):
            return -np.inf
        return -0.5 * ((beta0 + 9) / 2.0) ** 2 - 0.5 * (beta1 / 2.0) ** 2

    if model_name == "exponential":
        if not (-12 <= beta0 <= -6 and -0.03 <= beta1 <= 0.03):
            return -np.inf
        return -0.5 * ((beta0 + 9) / 2.0) ** 2 - 0.5 * (beta1 / 0.015) ** 2

    raise ValueError(model_name)


def log_likelihood(theta, bundle, W, Ts, y_obs, sigma_meas):
    beta0, beta1 = theta
    X = np.column_stack([W, Ts, np.full_like(W, beta0), np.full_like(W, beta1)])
    mu, std_gp = gp_predict(bundle, X)

    var = sigma_meas**2 + std_gp**2
    r = y_obs - mu
    return -0.5 * np.sum((r**2) / var + np.log(2 * np.pi * var))


def log_posterior(theta, bundle, model_name, W, Ts, y_obs, sigma_meas):
    lp = log_prior(theta, model_name)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, bundle, W, Ts, y_obs, sigma_meas)


def run_mcmc(
    model_name,
    data_csv,
    gp_path,
    nwalkers=32,
    nsteps=6000,
    burn=1500,
    thin=10,
    seed=0,
):
    df = pd.read_csv(data_csv)

    W = df["W_mps"].to_numpy()
    Ts = df["Ts_K"].to_numpy()
    y_obs = df["y_obs_kW"].to_numpy()
    sigma_meas = float(df["sigma_kW"].iloc[0])

    bundle = joblib.load(gp_path)

    ndim = 2
    rng = np.random.default_rng(seed)

    if model_name == "powerlaw":
        init = np.array([-9.0, 0.0])
        spread = np.array([0.3, 0.3])
    else:
        init = np.array([-9.0, 0.0])
        spread = np.array([0.3, 0.003])

    p0 = init + spread * rng.standard_normal(size=(nwalkers, ndim))

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        lambda th: log_posterior(th, bundle, model_name, W, Ts, y_obs, sigma_meas),
    )

    sampler.run_mcmc(p0, nsteps, progress=True)

    samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
    return samples


def calibrate_and_save(
    model_name,
    data_csv,
    gp_path,
    out_path,
    nwalkers=32,
    nsteps=6000,
    burn=1500,
    thin=10,
    seed=0,
):
    samples = run_mcmc(
        model_name=model_name,
        data_csv=data_csv,
        gp_path=gp_path,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burn=burn,
        thin=thin,
        seed=seed,
    )
    np.save(out_path, samples)
    print(f"Saved posterior samples to {out_path}")
    print(model_name, "posterior mean:", samples.mean(axis=0), "std:", samples.std(axis=0))
