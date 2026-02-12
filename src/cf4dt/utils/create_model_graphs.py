"""Utility helpers to create alpha(T) graphs for different toy models.

This module focuses on the existing toy thermal diffusivity models used in
the project:

- powerlaw:    alpha(T; beta0, beta1) = exp(beta0) * (T/T0)**beta1
- exponential: alpha(T; beta0, beta1) = exp(beta0 + beta1*(T - T0))

The main entry point ``main()`` generates a set of illustrative plots for
different beta combinations for both models, overlaid with the Ulamec (2007)
reference alpha(T), and saves them under ``outputs/``.

You can run it from the project root with::

	python -m cf4dt.utils.create_model_graphs

or equivalently:

	python src/cf4dt/utils/create_model_graphs.py

This is intended for quick visual exploration of how changing (beta0, beta1)
affects the toy models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, Dict
import sys

import numpy as np
import matplotlib.pyplot as plt

# Support both "python -m cf4dt.utils.create_model_graphs" (package mode)
# and "python src/cf4dt/utils/create_model_graphs.py" (script mode).
try:  # pragma: no cover - import robustness
	from cf4dt.forward import (
		alpha_model,
		k_ice_ulamec,
		rho_ice_ulamec,
		cp_ice_ulamec,
	)
except ModuleNotFoundError:  # likely running as a plain script
	ROOT = Path(__file__).resolve().parents[2]  # .../project/src
	if str(ROOT) not in sys.path:
		sys.path.insert(0, str(ROOT))
	from cf4dt.forward import (
		alpha_model,
		k_ice_ulamec,
		rho_ice_ulamec,
		cp_ice_ulamec,
	)


BetaPair = Tuple[float, float]


def _default_beta_sets() -> Dict[str, list[BetaPair]]:
	"""Provide a small, illustrative set of beta combinations.

	These are chosen to live inside the theory/prior ranges described in
	THEORY.md so that the resulting curves are physically plausible but
	visually distinct.
	"""

	# Powerlaw: beta0 in [-14.5, -13.5], beta1 in [0.5, 1.7]
	powerlaw_betas: list[BetaPair] = [
		(-14.3, 0.7),
		(-14.0, 1.1),  # around prior mean
		(-13.7, 1.5),
	]

	# Exponential: beta0 in [-14.5, -13.5], beta1 in [0.002, 0.010]
	exponential_betas: list[BetaPair] = [
		(-16.0759, -0.0354),
		(-14.6568, -0.0265),
		(-13.2377, -0.0176),
		(-11.8186, -0.0087),
		(-9.2378, 0.0),
	]

	return {"powerlaw": powerlaw_betas, "exponential": exponential_betas}


def _compute_ulamec(T: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
	"""Return (T, alpha_ulamec(T)) on a default or provided grid."""

	if T is None:
		T = np.linspace(80.0, 273.15, 200)
	k = k_ice_ulamec(T)
	rho = rho_ice_ulamec(T)
	cp = cp_ice_ulamec(T)
	alpha_ulamec = k / (rho * cp)
	return T, alpha_ulamec


def _plot_param_sweep(
	*,
	model_name: str,
	T: np.ndarray,
	alpha_ulamec: np.ndarray | None,
	T0: float,
	vary: str,
	fixed_value: float,
	values: Iterable[float],
	out_path: Path,
	use_loglog: bool = True,
)	-> Path:
	"""Plot alpha(T) for a single model varying one parameter.

	Parameters
	----------
	model_name : {"powerlaw", "exponential"}
	vary : {"beta0", "beta1"}
	fixed_value : float
		Value of the other beta kept fixed.
	values : iterable of float
		Values for the parameter being varied.
	out_path : Path
		Output path for the PNG file.
	"""

	fig, ax = plt.subplots(figsize=(7, 5))

	if alpha_ulamec is not None:
		if use_loglog:
			ax.loglog(T, alpha_ulamec, "k--", linewidth=2.5, label="Ulamec (2007)")
		else:
			ax.plot(T, alpha_ulamec, "k--", linewidth=2.5, label="Ulamec (2007)")

	values = list(values)
	for val in values:
		if vary == "beta0":
			theta = (val, fixed_value)
			lbl = fr"β₀={val:.3f} (β₁={fixed_value:.3f})"
		else:
			theta = (fixed_value, val)
			lbl = fr"β₁={val:.3f} (β₀={fixed_value:.3f})"

		alpha_vals = alpha_model(T, theta, model=model_name, T0=T0)
		if use_loglog:
			ax.loglog(T, alpha_vals, label=lbl)
		else:
			ax.plot(T, alpha_vals, label=lbl)

	ax.set_xlabel("Temperature T (K)")
	ax.set_ylabel("α(T) [m²/s]")
	title_model = "Powerlaw" if model_name == "powerlaw" else "Exponential"
	if vary == "beta0":
		ax.set_title(f"{title_model} model: vary β₀")
	else:
		ax.set_title(f"{title_model} model: vary β₁")

	if not use_loglog:
		ax.set_yscale("log")
	ax.grid(True, which="both", alpha=0.3)
	ax.legend(fontsize=9)
	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	return out_path


def generate_param_sweep_plots(
	*,
	T: np.ndarray | None = None,
	out_dir: str | Path = "outputs",
	T0: float = 200.0,
	use_loglog: bool = True,
	include_ulamec: bool = True,
)	-> dict[str, Path]:
	"""Create two graphs per model: vary β₀ and vary β₁.

	For each of the toy models (powerlaw, exponential), this function
	creates two PNG files:

	- one where β₀ varies and β₁ is fixed
	- one where β₁ varies and β₀ is fixed

	Returns a dict mapping a short key to each output path.
	"""

	defaults = _default_beta_sets()
	T, alpha_ulamec = _compute_ulamec(T) if include_ulamec else _compute_ulamec(T)
	if not include_ulamec:
		alpha_ulamec = None

	out_dir = Path(out_dir)
	paths: dict[str, Path] = {}

	# Powerlaw defaults
	pw_betas = defaults["powerlaw"]
	# choose central values as "fixed" references
	beta0_vals_pw = [b[0] for b in pw_betas]
	beta1_vals_pw = [b[1] for b in pw_betas]
	fixed_beta1_pw = float(np.median(beta1_vals_pw))
	fixed_beta0_pw = float(np.median(beta0_vals_pw))

	paths["powerlaw_vary_beta0"] = _plot_param_sweep(
		model_name="powerlaw",
		T=T,
		alpha_ulamec=alpha_ulamec,
		T0=T0,
		vary="beta0",
		fixed_value=fixed_beta1_pw,
		values=beta0_vals_pw,
		out_path=out_dir / "alpha_powerlaw_vary_beta0.png",
		use_loglog=use_loglog,
	)

	paths["powerlaw_vary_beta1"] = _plot_param_sweep(
		model_name="powerlaw",
		T=T,
		alpha_ulamec=alpha_ulamec,
		T0=T0,
		vary="beta1",
		fixed_value=fixed_beta0_pw,
		values=beta1_vals_pw,
		out_path=out_dir / "alpha_powerlaw_vary_beta1.png",
		use_loglog=use_loglog,
	)

	# Exponential defaults
	exp_betas = defaults["exponential"]
	beta0_vals_exp = [b[0] for b in exp_betas]
	beta1_vals_exp = [b[1] for b in exp_betas]
	fixed_beta1_exp = float(np.median(beta1_vals_exp))
	fixed_beta0_exp = float(np.median(beta0_vals_exp))

	paths["exponential_vary_beta0"] = _plot_param_sweep(
		model_name="exponential",
		T=T,
		alpha_ulamec=alpha_ulamec,
		T0=T0,
		vary="beta0",
		fixed_value=fixed_beta1_exp,
		values=beta0_vals_exp,
		out_path=out_dir / "alpha_exponential_vary_beta0.png",
		use_loglog=use_loglog,
	)

	paths["exponential_vary_beta1"] = _plot_param_sweep(
		model_name="exponential",
		T=T,
		alpha_ulamec=alpha_ulamec,
		T0=T0,
		vary="beta1",
		fixed_value=fixed_beta0_exp,
		values=beta1_vals_exp,
		out_path=out_dir / "alpha_exponential_vary_beta1.png",
		use_loglog=use_loglog,
	)

	return paths


def main() -> None:
	"""Command-line entry: generate two graphs per model.

	For each toy model (powerlaw and exponential) this will create:

	- outputs/alpha_powerlaw_vary_beta0.png
	- outputs/alpha_powerlaw_vary_beta1.png
	- outputs/alpha_exponential_vary_beta0.png
	- outputs/alpha_exponential_vary_beta1.png
	"""

	paths = generate_param_sweep_plots()
	for key, path in paths.items():
		print(f"Saved {key} plot to: {path}")


if __name__ == "__main__":  # pragma: no cover - manual plotting entry
	main()

