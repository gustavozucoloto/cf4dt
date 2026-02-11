"""Parameter-space factability checks for toy alpha(T) models.

This module provides utilities to explore the (beta0, beta1) parameter
space of the simplified thermal diffusivity models used in the project:

- powerlaw:    alpha(T; beta0, beta1) = exp(beta0) * (T/T0)**beta1
- exponential: alpha(T; beta0, beta1) = exp(beta0 + beta1*(T - T0))

The idea is to check which parameter combinations are *physically
acceptable* based on a known reference range for thermal diffusivity
alpha(T), without running the full PDE forward model.

Core functionality:

- Efficient sampling of (beta0, beta1) using Latin Hypercube Sampling
  (via SciPy, falling back to uniform random if unavailable).
- For each sampled parameter pair, we evaluate alpha(T; beta) on a
  temperature interval [T_min, T_max] and check whether alpha(T) stays
  within a user-defined range [alpha_min, alpha_max].
- From the accepted samples we infer a tighter, data-driven bounding
  box for beta0 and beta1.

You can run a simple command-line check from the project root, e.g.::

	python -m cf4dt.check_model_factability \
		--model powerlaw \
		--alpha-min 1e-7 --alpha-max 5e-6 \
		--n-samples 2000

Adjust the alpha range to your known physical range for thermal
diffusivity. The script will report how many samples are acceptable and
print an estimated feasible range for (beta0, beta1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Iterable, Dict
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

try:  # Prefer relative import when used as a package
	from .forward import alpha_model
except Exception:  # Fallback for direct script execution
	# __file__ = .../project/src/cf4dt/check_model_factability.py
	# parents[1] is the src/ directory which contains the cf4dt package.
	SRC = Path(__file__).resolve().parents[1]
	if str(SRC) not in sys.path:
		sys.path.insert(0, str(SRC))
	from cf4dt.forward import alpha_model



BetaPair = Tuple[float, float]


@dataclass
class SamplingConfig:
	"""Configuration for parameter-space sampling."""

	model_name: str  # "powerlaw" or "exponential"
	beta0_range: Tuple[float, float]
	beta1_range: Tuple[float, float]
	alpha_range: Tuple[float, float]
	T_range: Tuple[float, float] = (80.0, 273.15)
	n_T: int = 100
	n_samples: int = 2000
	sampler: str = "lhs"  # or "uniform"
	random_state: int | None = None


def _lhs_unit(n_samples: int, dim: int, rng: np.random.Generator) -> np.ndarray:
	"""Latin Hypercube in [0,1]^dim using SciPy if available, else manual.

	Returns an array of shape (n_samples, dim).
	"""

	try:
		from scipy.stats import qmc  # type: ignore

		sampler = qmc.LatinHypercube(d=dim, seed=rng.integers(0, 2**32 - 1))
		return sampler.random(n=n_samples)
	except Exception:  # pragma: no cover - fallback path
		# Simple manual LHS: divide [0,1] into n bins in each dim and
		# randomly shuffle samples within each bin.
		u = (np.arange(n_samples)[:, None] + rng.random((n_samples, dim))) / n_samples
		for j in range(dim):
			rng.shuffle(u[:, j])
		return u


def _sample_betas(config: SamplingConfig) -> np.ndarray:
	"""Sample (beta0, beta1) in their specified ranges.

	Returns
	-------
	betas : ndarray, shape (n_samples, 2)
	"""

	rng = np.random.default_rng(config.random_state)

	if config.sampler == "lhs":
		u = _lhs_unit(config.n_samples, 2, rng)
	else:
		u = rng.random((config.n_samples, 2))

	beta0 = config.beta0_range[0] + (config.beta0_range[1] - config.beta0_range[0]) * u[:, 0]
	beta1 = config.beta1_range[0] + (config.beta1_range[1] - config.beta1_range[0]) * u[:, 1]
	return np.column_stack([beta0, beta1])


def _is_feasible(
	beta0: float,
	beta1: float,
	*,
	model_name: str,
	alpha_min: float,
	alpha_max: float,
	T_min: float,
	T_max: float,
	n_T: int,
	T0: float = 200.0,
) -> bool:
	"""Check if alpha(T; beta) stays in [alpha_min, alpha_max] on [T_min, T_max]."""

	T = np.linspace(T_min, T_max, n_T)
	alpha_vals = alpha_model(T, (beta0, beta1), model=model_name, T0=T0)
	amin = float(np.min(alpha_vals))
	amax = float(np.max(alpha_vals))
	return (amin >= alpha_min) and (amax <= alpha_max)


def sample_parameter_space(config: SamplingConfig) -> Dict[str, np.ndarray]:
	"""Sample parameter space and evaluate feasibility against alpha-range.

	Parameters
	----------
	config : SamplingConfig
		Configuration describing model name, parameter ranges, alpha
		range, temperature range, and sampling details.

	Returns
	-------
	result : dict
		Dictionary with keys:

		- "betas": ndarray of shape (n_samples, 2)
		- "feasible": boolean mask of shape (n_samples,)
	"""

	betas = _sample_betas(config)
	feasible_mask = np.zeros(config.n_samples, dtype=bool)

	alpha_min, alpha_max = config.alpha_range
	T_min, T_max = config.T_range

	for i, (b0, b1) in enumerate(betas):
		feasible_mask[i] = _is_feasible(
			b0,
			b1,
			model_name=config.model_name,
			alpha_min=alpha_min,
			alpha_max=alpha_max,
			T_min=T_min,
			T_max=T_max,
			n_T=config.n_T,
		)

	return {"betas": betas, "feasible": feasible_mask}


def estimate_feasible_box(config: SamplingConfig) -> Tuple[Tuple[float, float], Tuple[float, float]]:
	"""Estimate a bounding box for (beta0, beta1) from feasible samples.

	Returns
	-------
	(beta0_min, beta0_max), (beta1_min, beta1_max)
	"""

	result = sample_parameter_space(config)
	betas = result["betas"]
	mask = result["feasible"]

	if not np.any(mask):
		raise RuntimeError("No feasible samples found for the given configuration.")

	feasible_betas = betas[mask]
	beta0_min = float(np.min(feasible_betas[:, 0]))
	beta0_max = float(np.max(feasible_betas[:, 0]))
	beta1_min = float(np.min(feasible_betas[:, 1]))
	beta1_max = float(np.max(feasible_betas[:, 1]))

	return (beta0_min, beta0_max), (beta1_min, beta1_max)


def _default_beta_ranges(model_name: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
	"""Return the prior/theory-informed default ranges for (beta0, beta1)."""

	if model_name == "powerlaw":
		# From THEORY.md
		return (-14.5, -13.5), (0.5, 1.7)
	if model_name == "exponential":
		return (-14.5, -13.5), (0.002, 0.010)
	raise ValueError(f"Unknown model_name: {model_name}")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Sample (beta0, beta1) for toy alpha(T) models and check "
			"feasibility against a thermal diffusivity range."
		)
	)

	parser.add_argument(
		"--model",
		choices=["powerlaw", "exponential"],
		required=True,
		help="Toy model to analyse.",
	)
	parser.add_argument("--alpha-min", type=float, required=True, help="Minimum alpha(T) [m^2/s].")
	parser.add_argument("--alpha-max", type=float, required=True, help="Maximum alpha(T) [m^2/s].")

	parser.add_argument("--beta0-min", type=float, help="Lower bound for beta0 (override default).")
	parser.add_argument("--beta0-max", type=float, help="Upper bound for beta0 (override default).")
	parser.add_argument("--beta1-min", type=float, help="Lower bound for beta1 (override default).")
	parser.add_argument("--beta1-max", type=float, help="Upper bound for beta1 (override default).")

	parser.add_argument("--T-min", type=float, default=80.0, help="Minimum temperature [K].")
	parser.add_argument("--T-max", type=float, default=273.15, help="Maximum temperature [K].")
	parser.add_argument("--n-T", type=int, default=100, help="Number of points in temperature grid.")

	parser.add_argument("--n-samples", type=int, default=2000, help="Number of parameter samples.")
	parser.add_argument(
		"--sampler",
		choices=["lhs", "uniform"],
		default="lhs",
		help="Sampling strategy (Latin Hypercube or uniform).",
	)
	parser.add_argument("--seed", type=int, default=None, help="Random seed.")
	parser.add_argument(
		"--plot",
		action="store_true",
		help=(
			"If set, save a scatter plot of sampled (beta0, beta1) "
			"colored by feasibility into outputs/."
		),
	)

	return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
	"""CLI entry point to run a quick parameter-space factability check."""

	args = _parse_args(argv)

	beta0_range_default, beta1_range_default = _default_beta_ranges(args.model)
	beta0_range = (
		args.beta0_min if args.beta0_min is not None else beta0_range_default[0],
		args.beta0_max if args.beta0_max is not None else beta0_range_default[1],
	)
	beta1_range = (
		args.beta1_min if args.beta1_min is not None else beta1_range_default[0],
		args.beta1_max if args.beta1_max is not None else beta1_range_default[1],
	)

	config = SamplingConfig(
		model_name=args.model,
		beta0_range=beta0_range,
		beta1_range=beta1_range,
		alpha_range=(args.alpha_min, args.alpha_max),
		T_range=(args.T_min, args.T_max),
		n_T=args.n_T,
		n_samples=args.n_samples,
		sampler=args.sampler,
		random_state=args.seed,
	)

	print("Sampling configuration:")
	print(f"  model       : {config.model_name}")
	print(f"  beta0 range : {config.beta0_range}")
	print(f"  beta1 range : {config.beta1_range}")
	print(f"  alpha range : {config.alpha_range} [m^2/s]")
	print(f"  T range     : {config.T_range} [K]")
	print(f"  n_T         : {config.n_T}")
	print(f"  n_samples   : {config.n_samples}")
	print(f"  sampler     : {config.sampler}")

	result = sample_parameter_space(config)
	betas = result["betas"]
	mask = result["feasible"]
	n_feasible = int(mask.sum())

	print("")
	print(f"Feasible samples: {n_feasible} / {config.n_samples} " f"({n_feasible / config.n_samples:.1%})")

	if n_feasible == 0:
		print("No feasible samples found. Consider widening the alpha range or beta ranges.")
		return

	(beta0_min, beta0_max), (beta1_min, beta1_max) = estimate_feasible_box(config)

	print("\nEstimated feasible parameter box (from feasible samples):")
	print(f"  beta0 in [{beta0_min:.4f}, {beta0_max:.4f}]")
	print(f"  beta1 in [{beta1_min:.4f}, {beta1_max:.4f}]")

	if args.plot:
		out_dir = Path("outputs")
		out_dir.mkdir(parents=True, exist_ok=True)
		out_path = out_dir / f"factability_{config.model_name}_samples.png"

		plt.figure(figsize=(6, 5))
		# Plot infeasible points lightly, feasible points highlighted
		if np.any(~mask):
			plt.scatter(
				betas[~mask, 0],
				betas[~mask, 1],
				color="lightgray",
				s=8,
				label="infeasible",
				alpha=0.5,
			)
		plt.scatter(
			betas[mask, 0],
			betas[mask, 1],
			color="tab:green",
			s=18,
			label="feasible",
			alpha=0.9,
		)
		plt.xlabel("beta0")
		plt.ylabel("beta1")
		plt.title(f"Parameter-space feasibility: {config.model_name}")
		plt.grid(True, alpha=0.3)
		plt.legend()
		plt.tight_layout()
		plt.savefig(out_path, dpi=200, bbox_inches="tight")
		plt.close()
		print(f"Saved sample scatter plot to: {out_path}")


if __name__ == "__main__":  # pragma: no cover - manual CLI use
	main()

