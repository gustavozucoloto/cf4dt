"""Noise models for synthetic sensor observations."""

import numpy as np


def add_noise_kW(Qlc_W, sigma_kW=0.1, seed=2):
    rng = np.random.default_rng(seed)
    Q_kW = Qlc_W / 1000.0
    y_kW = Q_kW + rng.normal(0.0, sigma_kW, size=Q_kW.shape)
    return y_kW
