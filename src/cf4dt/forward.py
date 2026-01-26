"""
fenicsx_forward_qlc.py  (FEniCSx / dolfinx)

Unified forward model for Q_lc computation.

This module is a corrected + fully functioning refactor of your forward model that:
- Preserves the ORIGINAL transient physics (your artificial-data generator):
    rho(T) cp(T) dT/dt = (1/r) d/dr ( k(T) r dT/dr )
  with t_end = L/W and
    Q_lc = 2π Ro * W * ∫_0^{t_end} q(t) dt
    q(t) = (-k ∇T)·n at r = Ro

- Supports multiple material models:
    material_model="ulamec"      -> full Ulamec k(T), rho(T), cp(T)
    material_model="powerlaw"    -> toy alpha(T;theta) with k = (rho0*cp0)*alpha
    material_model="exponential" -> toy alpha(T;theta) with k = (rho0*cp0)*alpha

- Optionally supports a steady conduction mode (W-independent) for quick tests:
    mode="steady" -> solves ∇·(k(T)∇T)=0 and returns Q = 2π Ro L q_wall

Key fixes relative to your refactor:
- Restores time dependence + W dependence by default (mode="transient").
- Uses UFL math (ufl.exp) for expressions involving the FEniCS function T.
- Makes the toy alpha-model physically consistent by converting alpha -> conductivity:
      k = (rho0 * cp0) * alpha
- Uses consistent heat-flux definition q = (-k ∇T)·n and ensures positivity.
- Keeps MPI compatibility (comm defaults to MPI.COMM_WORLD), but you can pass MPI.COMM_SELF.

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


# =============================================================================
# Material property accessor
# =============================================================================

def get_material_properties(material_model: str,
                            T,
                            theta=None,
                            *,
                            rho0: float = 917.0,
                            cp0: float = 2000.0,
                            C0: float | None = None):
    """
    Return (k(T), cp(T), rho(T)) as UFL expressions.

    Parameters
    ----------
    material_model : str
        "ulamec", "powerlaw", "exponential"
    T : dolfinx.fem.Function or UFL expression
        Temperature field
    theta : tuple[float,float] or None
        Required for toy models ("powerlaw", "exponential")
    rho0, cp0 : float
        Constant density and heat capacity used for toy models.
    C0 : float or None
        If provided, use k = C0 * alpha. Otherwise k = (rho0*cp0) * alpha.

    Notes
    -----
    - For toy models we interpret alpha as *thermal diffusivity* [m^2/s],
      and convert it to conductivity via k = (rho*cp)*alpha.
    - cp and rho are constants in the toy models, so rho*cp is constant as well.
    """
    if material_model == "ulamec":
        kT = k_ice_ulamec(T)
        cpT = cp_ice_ulamec(T)
        rhoT = rho_ice_ulamec(T)
        return kT, cpT, rhoT

    if material_model in ("powerlaw", "exponential"):
        if theta is None:
            raise ValueError(f"theta is required for material_model='{material_model}'")

        alphaT = alpha_model_ufl(T, theta, model=material_model)

        rhoT = PETSc.ScalarType(rho0)
        cpT = PETSc.ScalarType(cp0)

        if C0 is None:
            C0 = float(rho0 * cp0)  # [J/(m^3 K)]
        kT = PETSc.ScalarType(C0) * alphaT

        return kT, cpT, rhoT

    raise ValueError(f"Unknown material_model: {material_model}")


# =============================================================================
# Forward solver: transient (default) or steady
# =============================================================================

def compute_Qlc(
    *,
    W: float,
    Ts: float,
    material_model: str = "ulamec",
    theta=None,
    mode: str = "transient",
    Ro: float = 0.1,
    L: float = 3.7,
    Tm: float = 273.15,
    Rinf: float | None = None,
    # mesh/time
    num_cells: int = 1000,
    p_grade: float = 3.0,
    num_steps: int = 1000,
    dt_ratio: float = 1.03,
    # toy-model constants (only used when material_model != "ulamec")
    rho0: float = 917.0,
    cp0: float = 2000.0,
    C0: float | None = None,
    # comm and solver behavior
    comm=None,
    petsc_options: dict | None = None,
    petsc_prefix: str = "heat1d_",
    clamp_k_min: float = 1e-12,
) -> float:
    """
    Compute Q_lc.

    mode="transient" (default): reproduces your original artificial-data physics:
        rho(T)cp(T) (T-Tn)/dt + ∇·(k(T)∇T) = 0
        t_end = L/W
        Q_lc = 2π Ro * W * ∫_0^{t_end} q(t) dt
        q = (-k∇T)·n at r=Ro

    mode="steady": quick steady conduction approximation (W independent):
        ∇·(k(T)∇T)=0
        Q_lc = 2π Ro * L * q_wall

    Returns
    -------
    Q_lc_total : float
        Total lateral heat loss in Watts (positive).
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    if Rinf is None:
        Rinf = Ro + 5.0

    if mode not in ("transient", "steady"):
        raise ValueError("mode must be 'transient' or 'steady'")

    if mode == "transient":
        if W <= 0:
            raise ValueError("W must be > 0 in transient mode (t_end=L/W).")

    # -------------------------------------------------------------------------
    # Build graded 1D mesh r in [Ro, Rinf] by mapping [0,1] -> Ro+(Rinf-Ro)x^p
    # -------------------------------------------------------------------------
    domain = mesh.create_interval(comm, num_cells, [0.0, 1.0])
    x = domain.geometry.x
    x[:, 0] = Ro + (Rinf - Ro) * (x[:, 0] ** p_grade)

    V = fem.functionspace(domain, ("CG", 1))
    T = fem.Function(V)   # unknown at current step (or steady)
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

    # Initial condition: ambient Ts everywhere (enforce BC values explicitly)
    T.x.array[:] = Ts
    T.x.array[inner_dofs] = Tm
    T.x.array[outer_dofs] = Ts

    # Axisymmetric weight and normal
    rr = ufl.SpatialCoordinate(domain)[0]
    n = ufl.FacetNormal(domain)

    # Material properties (UFL)
    kT, cpT, rhoT = get_material_properties(
        material_model, T, theta,
        rho0=rho0, cp0=cp0, C0=C0
    )
    # Clamp k to avoid near-zero/negative issues
    kT = ufl.max_value(kT, PETSc.ScalarType(clamp_k_min))

    # -------------------------------------------------------------------------
    # Variational form
    # -------------------------------------------------------------------------
    if mode == "steady":
        # ∫ k ∇T·∇v * r dx = 0
        F = (kT * ufl.dot(ufl.grad(T), ufl.grad(v))) * rr * ufl.dx
        dt_c = None
        Tn = None
        dts = None
    else:
        # transient: implicit Euler
        Tn = fem.Function(V)
        Tn.x.array[:] = T.x.array[:]  # start from the initialized state

        # timestep schedule for t_end = L/W
        t_end = L / W
        rratio = float(dt_ratio)
        if abs(rratio - 1.0) < 1e-14:
            dts = np.full(num_steps, t_end / num_steps)
        else:
            dt0 = t_end * (rratio - 1.0) / (rratio**num_steps - 1.0)
            dts = dt0 * (rratio ** np.arange(num_steps))

        dt_c = fem.Constant(domain, PETSc.ScalarType(dts[0]))

        F = (rhoT * cpT * (T - Tn) / dt_c) * v * rr * ufl.dx \
            + (kT * ufl.dot(ufl.grad(T), ufl.grad(v))) * rr * ufl.dx

    # -------------------------------------------------------------------------
    # PETSc options
    # -------------------------------------------------------------------------
    if petsc_options is None:
        petsc_options = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-8,
            "snes_atol": 1e-10,
            "snes_max_it": 30,
            "snes_error_if_not_converged": True,

            "ksp_type": "gmres",
            "ksp_rtol": 1e-8,
            "ksp_max_it": 200,
            "ksp_error_if_not_converged": True,
        }
        # Preconditioner choice
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
    # -------------------------------------------------------------------------
    # Use the physically explicit definition:
    #   q = (-k ∇T)·n
    q_expr = (-kT * ufl.dot(ufl.grad(T), n))

    if mode == "steady":
        # Solve once
        problem.solve()

        q_wall = fem.assemble_scalar(fem.form(q_expr * ds(1)))
        q_wall = float(q_wall)
        q_wall = abs(q_wall)

        # Q = 2π Ro L q_wall
        Qlc_total = float(2.0 * np.pi * Ro * L * q_wall)
        return Qlc_total

    # transient integration
    # assemble flux at initial state:
    q_prev = fem.assemble_scalar(fem.form(q_expr * ds(1)))
    q_prev = float(q_prev)

    Q_int = 0.0  # integral of q dt

    for i in range(num_steps):
        dt_c.value = PETSc.ScalarType(dts[i])

        # Solve nonlinear system for this time step
        problem.solve()

        q_now = fem.assemble_scalar(fem.form(q_expr * ds(1)))
        q_now = float(q_now)

        # trapezoidal integration
        Q_int += 0.5 * (q_prev + q_now) * float(dts[i])
        q_prev = q_now

        # update previous solution
        Tn.x.array[:] = T.x.array[:]

    # Make sure q integral yields positive power
    Q_int = abs(Q_int)

    # Q_lc = 2π Ro * W * ∫ q dt
    Qlc_total = float(2.0 * np.pi * Ro * W * Q_int)
    return Qlc_total
