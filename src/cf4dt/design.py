"""Design utilities (Latin Hypercube sampling for Ts, W)."""

import numpy as np


def lhs_2d(n, rng):
    u = np.zeros((n, 2), dtype=float)
    for j in range(2):
        perm = rng.permutation(n)
        u[:, j] = (perm + rng.random(n)) / n
    return u


def sample_design(
    n_lhs=120,
    seed=1,
    Ts_min=80.0,
    Ts_max=260.0,
    W_mph_min=0.05,
    W_mph_max=10.0,
):
    """Return (W_all [m/s], Ts_all [K]) covering bounds plus edge points."""
    rng = np.random.default_rng(seed)
    u = lhs_2d(n_lhs, rng=rng)

    Ts = Ts_min + (Ts_max - Ts_min) * u[:, 0]

    logWmin = np.log10(W_mph_min)
    logWmax = np.log10(W_mph_max)
    W_mph = 10 ** (logWmin + (logWmax - logWmin) * u[:, 1])
    W = W_mph / 3600.0

    Ts_mid = 0.5 * (Ts_min + Ts_max)
    W_mph_mid = 10 ** (0.5 * (logWmin + logWmax))

    edge = [
        (W_mph_min, Ts_min),
        (W_mph_min, Ts_max),
        (W_mph_max, Ts_min),
        (W_mph_max, Ts_max),
        (W_mph_min, Ts_mid),
        (W_mph_max, Ts_mid),
        (W_mph_mid, Ts_min),
        (W_mph_mid, Ts_max),
    ]
    W_edge = np.array([w for w, _ in edge]) / 3600.0
    Ts_edge = np.array([t for _, t in edge])

    W_all = np.hstack([W, W_edge])
    Ts_all = np.hstack([Ts, Ts_edge])
    return W_all, Ts_all
