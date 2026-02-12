"""Bayesian calibration using GP surrogate and emcee."""

import numpy as np
import pandas as pd
import joblib
import emcee
from multiprocessing import Pool
from scipy.special import log_ndtr

from .gp_utils import gp_predict


# These bounds should match the GP emulator training ranges for (beta0, beta1).
DEFAULT_BOUNDS = {
    "powerlaw": {
        "beta0": (-14.3507, -11.7415),
        "beta1": (-2.09343, -1.7128),
        "mu": (-13.0461, -1.90311),
        "sigma": (0.6523, 0.0952),
    },
    "exponential": {
        "beta0": (-14.2971, -11.6977),
        "beta1": (-0.0131714, -0.0107766),
        "mu": (-12.9974, -0.011974),
        "sigma": (0.6499, 0.000599),
    },
    "logarithmic": {
        "beta0": (-14.2804, -11.684),
        "beta1": (-3.74977, -3.06799),
        "mu": (-12.9822, -3.40888),
        "sigma": (0.6491, 0.1704),
    },
}


def get_prior_bounds(model_name, beta0_bounds=None, beta1_bounds=None):
    defaults = DEFAULT_BOUNDS[model_name]
    beta0_min, beta0_max = beta0_bounds or defaults["beta0"]
    beta1_min, beta1_max = beta1_bounds or defaults["beta1"]
    return beta0_min, beta0_max, beta1_min, beta1_max


def _logsubexp(a, b):
    """Compute log(exp(a) - exp(b)) stably for a >= b."""
    if b > a:
        return -np.inf
    return a + np.log1p(-np.exp(b - a))


def _log_truncnorm_1d(x, mu, sigma, lo, hi):
    """Log-pdf of N(mu,sigma^2) truncated to [lo, hi]. Returns -inf outside."""
    if not np.isfinite(x) or not np.isfinite(mu) or not np.isfinite(sigma):
        return -np.inf
    if sigma <= 0:
        return -np.inf
    if x < lo or x > hi:
        return -np.inf

    z = (x - mu) / sigma
    log_pdf = -0.5 * z * z - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)

    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    log_cdf_b = float(log_ndtr(b))
    log_cdf_a = float(log_ndtr(a))
    log_Z = _logsubexp(log_cdf_b, log_cdf_a)
    if not np.isfinite(log_Z):
        return -np.inf

    return log_pdf - log_Z


def log_prior(theta, model_name, beta0_bounds=None, beta1_bounds=None, prior_kind="truncnorm"):
    """
    Log prior on (beta0, beta1).

    prior_kind:
      - "truncnorm": truncated Gaussian within the GP training box (recommended with a GP surrogate)
      - "gaussian": unbounded Gaussian (can lead to GP extrapolation issues)
    """
    beta0, beta1 = theta
    mu0, mu1 = DEFAULT_BOUNDS[model_name]["mu"]
    sig0, sig1 = DEFAULT_BOUNDS[model_name]["sigma"]

    prior_kind = str(prior_kind).lower()

    if prior_kind in {"gaussian", "normal"}:
        # Full Gaussian log-pdf up to an additive constant (kept consistent here)
        z0 = (beta0 - mu0) / sig0
        z1 = (beta1 - mu1) / sig1
        return (
            -0.5 * z0 * z0 - np.log(sig0) - 0.5 * np.log(2.0 * np.pi)
            -0.5 * z1 * z1 - np.log(sig1) - 0.5 * np.log(2.0 * np.pi)
        )

    if prior_kind in {"truncnorm", "truncated", "truncated_gaussian", "truncated_normal"}:
        beta0_min, beta0_max, beta1_min, beta1_max = get_prior_bounds(
            model_name, beta0_bounds=beta0_bounds, beta1_bounds=beta1_bounds
        )
        lp_beta0 = _log_truncnorm_1d(beta0, mu0, sig0, beta0_min, beta0_max)
        lp_beta1 = _log_truncnorm_1d(beta1, mu1, sig1, beta1_min, beta1_max)
        return lp_beta0 + lp_beta1

    raise ValueError(f"Unknown prior_kind={prior_kind!r}")


def log_likelihood(theta, bundle, W, Ts, y_obs, sigma_meas, var_floor=1e-18):
    beta0, beta1 = theta
    X = np.column_stack([W, Ts, np.full_like(W, beta0), np.full_like(W, beta1)])
    mu, std_gp = gp_predict(bundle, X)

    var = sigma_meas**2 + std_gp**2
    var = np.maximum(var, var_floor)

    r = y_obs - mu
    ll = -0.5 * np.sum((r**2) / var + np.log(2.0 * np.pi * var))
    return ll


def log_posterior(
    theta, bundle, model_name, W, Ts, y_obs, sigma_meas,
    beta0_bounds=None, beta1_bounds=None, prior_kind="truncnorm"
):
    lp = log_prior(
        theta, model_name,
        beta0_bounds=beta0_bounds, beta1_bounds=beta1_bounds,
        prior_kind=prior_kind
    )
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(theta, bundle, W, Ts, y_obs, sigma_meas)
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll


def run_mcmc(
    model_name, data_csv, gp_path,
    nwalkers=32, nsteps=6000, burn=1500, thin=10, seed=0,
    beta0_bounds=None, beta1_bounds=None, prior_kind="truncnorm",
    n_jobs=1
):
    df = pd.read_csv(data_csv)
    W = df["W_mph"].to_numpy()
    Ts = df["Ts_K"].to_numpy()
    y_obs = df["y_obs_kW"].to_numpy()
    sigma_meas = float(df["sigma_kW"].iloc[0])

    bundle = joblib.load(gp_path)

    rng = np.random.default_rng(seed)
    ndim = 2

    # Use the GP training box to initialize walkers.
    beta0_min, beta0_max, beta1_min, beta1_max = get_prior_bounds(
        model_name, beta0_bounds=beta0_bounds, beta1_bounds=beta1_bounds
    )
    p0 = np.column_stack([
        rng.uniform(beta0_min, beta0_max, size=nwalkers),
        rng.uniform(beta1_min, beta1_max, size=nwalkers),
    ])

    log_prob_args = (bundle, model_name, W, Ts, y_obs, sigma_meas, beta0_bounds, beta1_bounds, prior_kind)
    move = emcee.moves.StretchMove(a=2.0)

    if n_jobs > 1:
        with Pool(processes=n_jobs) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_posterior, args=log_prob_args, moves=move, pool=pool
            )
            sampler.run_mcmc(p0, nsteps, progress=True)
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior, args=log_prob_args, moves=move
        )
        sampler.run_mcmc(p0, nsteps, progress=True)

    chain = sampler.get_chain(discard=burn, thin=thin, flat=True)
    logp = sampler.get_log_prob(discard=burn, thin=thin, flat=True)

    mask = np.isfinite(logp)
    samples = chain[mask]
    return samples


def calibrate_and_save(
    model_name, data_csv, gp_path, out_path,
    nwalkers=32, nsteps=6000, burn=1500, thin=10, seed=0,
    beta0_bounds=None, beta1_bounds=None, prior_kind="truncnorm",
    n_jobs=1,
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
        beta0_bounds=beta0_bounds,
        beta1_bounds=beta1_bounds,
        prior_kind=prior_kind,
        n_jobs=n_jobs,
    )

    np.save(out_path, samples)
    print(f"Saved posterior samples to {out_path}")
    print(model_name, "posterior mean:", samples.mean(axis=0), "std:", samples.std(axis=0))
    print("num samples saved:", len(samples))

    # Optional diagnostic: should be 0.0 for truncnorm
    b0min, b0max, b1min, b1max = get_prior_bounds(model_name, beta0_bounds, beta1_bounds)
    frac_out = np.mean(
        (samples[:, 0] < b0min) | (samples[:, 0] > b0max) |
        (samples[:, 1] < b1min) | (samples[:, 1] > b1max)
    )
    print("fraction outside GP theta box:", frac_out)
