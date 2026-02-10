"""Design utilities (uniform grid sampling for Ts, W)."""

import numpy as np


def sample_design(
    W_mph_list=None,
    Ts_K_list=None,
    n_Ts=10,
    n_W=12,
    Ts_min=80.0,
    Ts_max=260.0,
    W_mph_min=0.05,
    W_mph_max=10.0,
    **kwargs  # Accept but ignore legacy parameters like seed, n_lhs
):
    """
    Return (W_mph_all [m/h], Ts_all [K]) as grid.
    
    Parameters
    ----------
    W_mph_list : list or array, optional
        Explicit list of velocities in meters per hour. If provided, uses these instead of generating.
    Ts_K_list : list or array, optional
        Explicit list of temperatures in K. If provided, uses these instead of generating.
    n_Ts : int
        Number of uniformly spaced temperature points (used if Ts_K_list not provided)
    n_W : int
        Number of uniformly spaced (log-scale) velocity points in m/h (used if W_mph_list not provided)
    """
    # Use explicit lists if provided, otherwise generate
    if Ts_K_list is not None:
        Ts_grid = np.array(Ts_K_list)
    else:
        Ts_grid = np.linspace(Ts_min, Ts_max, n_Ts)
    
    if W_mph_list is not None:
        W_mph_grid = np.array(W_mph_list)
    else:
        logWmin = np.log10(W_mph_min)
        logWmax = np.log10(W_mph_max)
        W_mph_grid = np.logspace(logWmin, logWmax, n_W)
    
    # Create meshgrid
    W_mesh, Ts_mesh = np.meshgrid(W_mph_grid, Ts_grid)
    
    # Flatten to get all combinations
    W_mph_all = W_mesh.ravel()
    Ts_all = Ts_mesh.ravel()
    return W_mph_all, Ts_all
