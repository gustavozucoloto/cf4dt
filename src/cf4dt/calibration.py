"""Bayesian calibration using GP surrogate and emcee."""

import numpy as np
import pandas as pd
import joblib
import emcee
from multiprocessing import Pool

from .gp_utils import gp_predict


DEFAULT_BOUNDS = {
    "powerlaw": {
        "beta0": (-14.5, -13.5),
        "beta1": (0.1, 0.9),
        "mu": (-14.0, 0.6),
        "sigma": (0.3, 0.2),
    },
    "exponential": {
        "beta0": (-14.5, -13.5),
        "beta1": (0.002, 0.010),
        "mu": (-14.0, 0.006),
        "sigma": (0.3, 0.002),
    },
    "logarithmic": {
        "beta0": (-14.5, -13.5),
        "beta1": (0.1, 1.0),
        "mu": (-14.0, 0.5),
        "sigma": (0.3, 0.3),
    },
}


def log_prior(theta, model_name, beta0_bounds=None, beta1_bounds=None):
    beta0, beta1 = theta
    defaults = DEFAULT_BOUNDS[model_name]
    beta0_min, beta0_max = beta0_bounds or defaults["beta0"]
    beta1_min, beta1_max = beta1_bounds or defaults["beta1"]
    mu0, mu1 = defaults["mu"]
    sig0, sig1 = defaults["sigma"]

    if model_name == "powerlaw":
        # beta0 ~ N(-14.0, 0.3^2), beta1 ~ N(1.1, 0.3^2)
        if not (beta0_min <= beta0 <= beta0_max and beta1_min <= beta1 <= beta1_max):
            return -np.inf
        lp_beta0 = -0.5 * ((beta0 - mu0) / sig0) ** 2
        lp_beta1 = -0.5 * ((beta1 - mu1) / sig1) ** 2
        return lp_beta0 + lp_beta1

    if model_name == "exponential":
        # beta0 ~ N(-14.0, 0.3^2), beta1 ~ N(0.006, 0.002^2)
        if not (beta0_min <= beta0 <= beta0_max and beta1_min <= beta1 <= beta1_max):
            return -np.inf
        lp_beta0 = -0.5 * ((beta0 - mu0) / sig0) ** 2
        lp_beta1 = -0.5 * ((beta1 - mu1) / sig1) ** 2
        return lp_beta0 + lp_beta1

    if model_name == "logarithmic":
        # beta0 ~ N(-14.0, 0.3^2), beta1 ~ N(0.5, 0.3^2)
        if not (beta0_min <= beta0 <= beta0_max and beta1_min <= beta1 <= beta1_max):
            return -np.inf
        lp_beta0 = -0.5 * ((beta0 - mu0) / sig0) ** 2
        lp_beta1 = -0.5 * ((beta1 - mu1) / sig1) ** 2
        return lp_beta0 + lp_beta1

    raise ValueError(model_name)


def log_likelihood(theta, bundle, W, Ts, y_obs, sigma_meas):
    beta0, beta1 = theta
    X = np.column_stack([W, Ts, np.full_like(W, beta0), np.full_like(W, beta1)])
    mu, std_gp = gp_predict(bundle, X)

    var = sigma_meas**2 + std_gp**2
    r = y_obs - mu
    return -0.5 * np.sum((r**2) / var + np.log(2 * np.pi * var))


def log_posterior(theta, bundle, model_name, W, Ts, y_obs, sigma_meas, beta0_bounds, beta1_bounds):
    lp = log_prior(theta, model_name, beta0_bounds=beta0_bounds, beta1_bounds=beta1_bounds)
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
    beta0_bounds=None,
    beta1_bounds=None,
    n_jobs=1,  # Number of parallel processes (1=serial)
):
    df = pd.read_csv(data_csv)

    W = df["W_mph"].to_numpy()
    Ts = df["Ts_K"].to_numpy()
    y_obs = df["y_obs_kW"].to_numpy()
    sigma_meas = float(df["sigma_kW"].iloc[0])

    bundle = joblib.load(gp_path)

    ndim = 2
    rng = np.random.default_rng(seed)

    if model_name == "powerlaw":
        init = np.array([-14.0, 1.1])
        spread = np.array([0.6, 0.4])
    elif model_name == "exponential":
        init = np.array([-14.0, 0.006])
        spread = np.array([0.6, 0.003])
    else:
        init = np.array([-14.0, 0.5])
        spread = np.array([0.6, 0.3])

    p0 = init + spread * rng.standard_normal(size=(nwalkers, ndim))

    log_prob_args = (bundle, model_name, W, Ts, y_obs, sigma_meas, beta0_bounds, beta1_bounds)

    # Use Pool for parallel walkers if n_jobs > 1
    if n_jobs > 1:
        with Pool(processes=n_jobs) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                log_posterior,
                args=log_prob_args,
                moves=emcee.moves.StretchMove(a=2.0),
                pool=pool,
            )
            sampler.run_mcmc(p0, nsteps, progress=True)
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_posterior,
            args=log_prob_args,
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
    beta0_bounds=None,
    beta1_bounds=None,
    n_jobs=1,  # Number of parallel processes (1=serial)
):
    """
    Run Bayesian calibration and save posterior samples.
    
    Parameters
    ----------
    n_jobs : int
        Number of parallel processes. Set to 1 for serial execution.
    """
    samples = run_mcmc(
        model_name=model_name,
        data_csv=data_csv,
        gp_path=gp_path,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burn=burn,
        thin=thin,
        seed=seed,
        beta0_bounds=beta0_bounds,
        beta1_bounds=beta1_bounds,
        n_jobs=n_jobs,
    )
    np.save(out_path, samples)
    print(f"Saved posterior samples to {out_path}")
    print(model_name, "posterior mean:", samples.mean(axis=0), "std:", samples.std(axis=0))
