"""
fenicsx_forward_qlc.py  (FEniCSx / dolfinx)

Unified forward model for Q_lc computation using thermal diffusivity (alpha).

This module solves the transient heat conduction problem in thermal diffusivity form:
    dT/dt = (1/r) d/dr(alpha(T) * r * dT/dr)
  with t_end = L/W and
    Q_lc = 2π Ro * W * ∫_0^{t_end} q(t) dt
    q(t) = (-alpha ∇T)·n at r = Ro

Supports three material models:
    - "ulamec": Uses Ulamec (2007) correlations, computes alpha(T) = k(T)/(rho(T)*cp(T))
    - "powerlaw": Toy model alpha(T;theta) = exp(beta0) * (T/T0)^beta1
    - "exponential": Toy model alpha(T;theta) = exp(beta0 + beta1*(T - T0))

Key features:
- Direct calibration of thermal diffusivity (eliminates rho, cp uncertainty)
- Transient physics with W-dependence
- UFL-compatible expressions for FEniCS
- MPI-aware solver

Author: (your refactor helper)
"""

from __future__ import annotations

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import NonlinearProblem


# =============================================================================
# Ulamec (2007) material properties (work for UFL Expressions)
# =============================================================================

def rho_ice_ulamec(T):
    """Ice density rho(T) [kg/m^3]. Valid ~ 0–273 K."""
    return 933.31 + 0.037978 * T - 3.6274e-4 * T**2


def k_ice_ulamec(T):
    """Ice thermal conductivity k(T) [W/(m·K)]. Valid ~ 10 K–Tm."""
    return 619.2 / T + 58646.0 / (T**3) + 3.237e-3 * T - 1.382e-5 * T**2


def cp_ice_ulamec(T):
    """Ice heat capacity cp(T) [J/(kg·K)] (rational form)."""
    Tt = 273.16
    x = T / Tt
    c1 = 1.843e5
    c2 = 1.6357e8
    c3 = 3.5519e9
    c4 = 1.667e2
    c5 = 6.465e4
    c6 = 1.6935e6
    num = x**3 * c1 + c2 * x**2 + c3 * x**6
    den = 1.0 + c4 * x**2 + c5 * x**4 + c6 * x**8
    return num / den


# =============================================================================
# Toy alpha(T;theta) models (UFL-safe)
# =============================================================================

def alpha_model_ufl(T, theta, model: str = "powerlaw", T0: float = 200.0):
    """
    Return alpha(T;theta) as a UFL expression (positive by construction).

    theta = (beta0, beta1)

    powerlaw:    alpha = exp(beta0) * (T/T0)^beta1
    exponential: alpha = exp(beta0 + beta1*(T - T0))
    """
    beta0, beta1 = theta
    beta0 = PETSc.ScalarType(beta0)
    beta1 = PETSc.ScalarType(beta1)
    T0 = PETSc.ScalarType(T0)

    if model == "powerlaw":
        return ufl.exp(beta0) * (T / T0) ** beta1
    if model == "exponential":
        return ufl.exp(beta0 + beta1 * (T - T0))
    raise ValueError(f"Unknown alpha model: {model}")


def alpha_model(T, theta, model: str = "powerlaw", T0: float = 200.0):
    """
    Return alpha(T;theta) as a numpy array (for UQ/plotting, not UFL).

    theta = (beta0, beta1)

    powerlaw:    alpha = exp(beta0) * (T/T0)^beta1
    exponential: alpha = exp(beta0 + beta1*(T - T0))
    """
    beta0, beta1 = theta

    if model == "powerlaw":
        return np.exp(beta0) * (T / T0) ** beta1
    if model == "exponential":
        return np.exp(beta0 + beta1 * (T - T0))
    raise ValueError(f"Unknown alpha model: {model}")


# =============================================================================
# Thermal diffusivity models
# =============================================================================

def alpha_ulamec_ufl(T):
    """
    Compute thermal diffusivity alpha = k/(rho*cp) for Ulamec model (UFL).
    
    Returns alpha(T) [m^2/s] as a UFL expression.
    """
    kT = k_ice_ulamec(T)
    rhoT = rho_ice_ulamec(T)
    cpT = cp_ice_ulamec(T)
    return kT / (rhoT * cpT)


def get_alpha_ufl(material_model: str, T, theta=None):
    """
    Return thermal diffusivity alpha(T) as a UFL expression for all models.

    Parameters
    ----------
    material_model : str
        "ulamec", "powerlaw", or "exponential"
    T : dolfinx.fem.Function or UFL expression
        Temperature field
    theta : tuple[float,float] or None
        Required for toy models ("powerlaw", "exponential")

    Returns
    -------
    alphaT : UFL expression
        Thermal diffusivity alpha(T) [m^2/s]

    Notes
    -----
    - "ulamec": alpha = k/(rho*cp) using Ulamec (2007) correlations
    - "powerlaw": alpha = exp(beta0) * (T/T0)^beta1
    - "exponential": alpha = exp(beta0 + beta1*(T - T0))
    """
    if material_model == "ulamec":
        return alpha_ulamec_ufl(T)
    
    if material_model in ("powerlaw", "exponential"):
        if theta is None:
            raise ValueError(f"theta is required for material_model='{material_model}'")
        return alpha_model_ufl(T, theta, model=material_model)
    
    raise ValueError(f"Unknown material_model: {material_model}")


# =============================================================================
# Forward solver: unified transient solver using alpha formulation
# =============================================================================

def compute_Qlc(
    *,
    W: float,
    Ts: float,
    material_model: str = "ulamec",
    theta=None,
    Ro: float = 0.1,
    L: float = 3.7,
    Tm: float = 273.15,
    Rinf: float | None = None,
    # mesh/time
    num_cells: int = 1000,
    p_grade: float = 3.0,
    num_steps: int = 1000,
    dt_ratio: float = 1.03,
    # comm and solver behavior
    comm=None,
    petsc_options: dict | None = None,
    petsc_prefix: str = "heat1d_",
    clamp_alpha_min: float = 1e-12,
) -> float:
    """
    Compute Q_lc using thermal diffusivity (alpha) formulation.

    Solves the transient heat conduction problem:
        dT/dt = (1/r) d/dr(alpha(T) * r * dT/dr)
    
    with t_end = L/W and
        Q_lc = 2π Ro * W * ∫_0^{t_end} q(t) dt
        q(t) = (-alpha ∇T)·n at r = Ro

    Parameters
    ----------
    W : float
        Melt rate [m/s]
    Ts : float
        Surface/ambient temperature [K]
    material_model : str
        "ulamec", "powerlaw", or "exponential"
    theta : tuple[float, float] or None
        Calibration parameters (beta0, beta1) for toy models.
        Not required for "ulamec".
    Ro : float
        Probe radius [m]
    L : float
        Probe length [m]
    Tm : float
        Melting temperature [K]
    Rinf : float or None
        Outer domain radius [m]. Default: Ro + 5.0
    num_cells : int
        Number of mesh cells
    p_grade : float
        Mesh grading parameter (>1 refines near Ro)
    num_steps : int
        Number of time steps
    dt_ratio : float
        Time step growth ratio (1.0 = uniform)
    comm : MPI communicator or None
        Default: MPI.COMM_WORLD
    petsc_options : dict or None
        PETSc solver options
    petsc_prefix : str
        PETSc options prefix
    clamp_alpha_min : float
        Minimum alpha value to prevent solver issues

    Returns
    -------
    Q_lc_total : float
        Total lateral heat loss in Watts (positive).

    Notes
    -----
    - All models use alpha formulation: dT/dt = (1/r) d/dr(alpha r dT/dr)
    - For "ulamec": alpha = k/(rho*cp) using Ulamec (2007) correlations
    - For toy models: alpha parameterized directly by theta
    - This eliminates uncertainty from constant rho, cp assumptions
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    if Rinf is None:
        Rinf = Ro + 5.0

    if W <= 0:
        raise ValueError("W must be > 0 (t_end=L/W).")

    # -------------------------------------------------------------------------
    # Build graded 1D mesh r in [Ro, Rinf]
    # -------------------------------------------------------------------------
    domain = mesh.create_interval(comm, num_cells, [0.0, 1.0])
    x = domain.geometry.x
    x[:, 0] = Ro + (Rinf - Ro) * (x[:, 0] ** p_grade)

    V = fem.functionspace(domain, ("CG", 1))
    T = fem.Function(V)
    v = ufl.TestFunction(V)

    # Boundary tagging
    fdim = domain.topology.dim - 1

    def is_inner(xp):
        return np.isclose(xp[0], Ro, atol=1e-12)

    def is_outer(xp):
        return np.isclose(xp[0], Rinf, atol=1e-12)

    inner_facets = mesh.locate_entities_boundary(domain, fdim, is_inner)
    outer_facets = mesh.locate_entities_boundary(domain, fdim, is_outer)

    facet_indices = np.hstack([inner_facets, outer_facets]).astype(np.int32)
    facet_markers = np.hstack([
        np.full(inner_facets.shape, 1, dtype=np.int32),
        np.full(outer_facets.shape, 2, dtype=np.int32),
    ])

    order = np.argsort(facet_indices)
    facet_tag = mesh.meshtags(domain, fdim, facet_indices[order], facet_markers[order])
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

    # Dirichlet BCs
    inner_dofs = fem.locate_dofs_topological(V, fdim, facet_tag.find(1))
    outer_dofs = fem.locate_dofs_topological(V, fdim, facet_tag.find(2))
    bc_inner = fem.dirichletbc(PETSc.ScalarType(Tm), inner_dofs, V)
    bc_outer = fem.dirichletbc(PETSc.ScalarType(Ts), outer_dofs, V)
    bcs = [bc_inner, bc_outer]

    # Initial condition
    T.x.array[:] = Ts
    T.x.array[inner_dofs] = Tm
    T.x.array[outer_dofs] = Ts

    # Axisymmetric weight and normal
    rr = ufl.SpatialCoordinate(domain)[0]
    n = ufl.FacetNormal(domain)

    # Get alpha(T) for the specified model
    alphaT = get_alpha_ufl(material_model, T, theta=theta)
    # Clamp to avoid near-zero issues
    alphaT = ufl.max_value(alphaT, PETSc.ScalarType(clamp_alpha_min))

    # -------------------------------------------------------------------------
    # Variational form: transient with implicit Euler
    # dT/dt = (1/r) d/dr(alpha * r * dT/dr)
    # -------------------------------------------------------------------------
    Tn = fem.Function(V)
    Tn.x.array[:] = T.x.array[:]

    # Timestep schedule
    t_end = L / W
    rratio = float(dt_ratio)
    if abs(rratio - 1.0) < 1e-14:
        dts = np.full(num_steps, t_end / num_steps)
    else:
        dt0 = t_end * (rratio - 1.0) / (rratio**num_steps - 1.0)
        dts = dt0 * (rratio ** np.arange(num_steps))

    dt_c = fem.Constant(domain, PETSc.ScalarType(dts[0]))

    # Weak form: ∫ (T-Tn)/dt * v * r dr + ∫ alpha ∇T·∇v * r dr = 0
    F = (T - Tn) / dt_c * v * rr * ufl.dx \
        + (alphaT * ufl.dot(ufl.grad(T), ufl.grad(v))) * rr * ufl.dx

    # -------------------------------------------------------------------------
    # PETSc options
    # -------------------------------------------------------------------------
    if petsc_options is None:
        petsc_options = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-5,
            "snes_atol": 1e-7,
            "snes_max_it": 100,
            "snes_error_if_not_converged": False,
            "ksp_type": "gmres",
            "ksp_rtol": 1e-6,
            "ksp_max_it": 300,
            "ksp_error_if_not_converged": False,
        }
        if comm.size == 1:
            petsc_options["pc_type"] = "ilu"
        else:
            petsc_options["pc_type"] = "hypre"
            petsc_options["pc_hypre_type"] = "boomeramg"

    problem = NonlinearProblem(
        F, T, bcs=bcs,
        petsc_options=petsc_options,
        petsc_options_prefix=petsc_prefix
    )

    # -------------------------------------------------------------------------
    # Wall heat flux and Q_lc integration
    # q = (-k ∇T)·n = (-alpha * rho * cp * ∇T)·n  [W/m²]
    # (must include rho*cp to convert from thermal diffusivity to thermal conductivity)
    # -------------------------------------------------------------------------
    rhoT = rho_ice_ulamec(T)
    cpT = cp_ice_ulamec(T)
    q_expr = (-alphaT * rhoT * cpT * ufl.dot(ufl.grad(T), n))

    # Initial flux
    q_prev = fem.assemble_scalar(fem.form(q_expr * ds(1)))
    q_prev = float(q_prev)
    Q_int = 0.0

    # Time stepping
    for i in range(num_steps):
        dt_c.value = PETSc.ScalarType(dts[i])
        problem.solve()
        
        q_now = fem.assemble_scalar(fem.form(q_expr * ds(1)))
        q_now = float(q_now)
        
        # Trapezoidal integration
        Q_int += 0.5 * (q_prev + q_now) * float(dts[i])
        q_prev = q_now
        
        # Update previous solution
        Tn.x.array[:] = T.x.array[:]

    Q_int = abs(Q_int)
    Qlc_total = float(2.0 * np.pi * Ro * W * Q_int)
    return Qlc_total
