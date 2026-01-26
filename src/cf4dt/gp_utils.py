"""Shared Gaussian Process helper functions."""


def gp_predict(bundle, X):
    gp = bundle["gp"]
    xscaler = bundle["xscaler"]
    yscaler = bundle["yscaler"]

    Xs = xscaler.transform(X)
    mu_s, std_s = gp.predict(Xs, return_std=True)

    mu = yscaler.inverse_transform(mu_s.reshape(-1, 1)).ravel()
    std = std_s * float(yscaler.scale_[0])
    return mu, std
