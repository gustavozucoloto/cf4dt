import numpy as np
s = np.load("data/posterior_powerlaw.npy")
beta0, beta1 = s[:,0], s[:,1]

mask = (
    (beta0 >= -14.8) & (beta0 <= -13.2) &
    (beta1 >= 0.1) & (beta1 <= 0.9)
)

s2 = s[mask]
np.save("data/posterior_powerlaw_clean.npy", s2)
print("kept", len(s2), "of", len(s))
print("mean", s2.mean(axis=0), "std", s2.std(axis=0))
