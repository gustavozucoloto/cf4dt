# Theory: Cryobot Digital Twin Framework

## 1. Artificial Data Generation

### 1.1 The Forward Problem

The thermal diffusivity formulation for transient heat conduction in a cylindrical geometry (cryobot borehole) is solved:

$$\frac{\partial T}{\partial t} = \frac{1}{r}\frac{\partial}{\partial r}\left(\alpha(T) \, r \frac{\partial T}{\partial r}\right)$$

with:
- Spatial domain: $r \in [R_o, R_\infty]$ (inner to outer radius)
- Time domain: $t \in [0, L/W]$ (penetration time)
- Boundary conditions: $T(R_o, t) = T_s$ (constant surface), $T(R_\infty, t) = T_\infty$ (ambient far-field)
- Heat flux at inner boundary: $q(t) = -\alpha(T) \left.\frac{\partial T}{\partial r}\right|_{r=R_o}$

The problem is solved using the Finite Element Method (FEniCSx/dolfinx) with implicit time stepping. The melting power is then computed:

$$Q_{lc} = 2\pi R_o W \int_0^{L/W} q(t) \, dt$$

where $W$ is the penetration velocity and $L$ is the borehole length.

### 1.2 Material Model: Ulamec (2007)

The "real" synthetic truth data is generated using the Ulamec (2007) correlations for ice:

$$\alpha(T) = \frac{k(T)}{\rho(T) \, c_p(T)}$$

where:
- **Density** (kg/m³): $\rho(T) = 933.31 + 0.037978 \, T - 3.6274 \times 10^{-4} \, T^2$
- **Thermal conductivity** (W/(m·K)): $k(T) = \frac{619.2}{T} + \frac{58646}{T^3} + 3.237 \times 10^{-3} \, T - 1.382 \times 10^{-5} \, T^2$
- **Heat capacity** (J/(kg·K)): $c_p(T) = \frac{x^3 c_1 + c_2 x^2 + c_3 x^6}{1 + c_4 x^2 + c_5 x^4 + c_6 x^8}$, $x = T/273.16$

### 1.3 Parameterized Models

Three simplified toy models are proposed to approximate the thermal diffusivity:

**Powerlaw model:**
$$\alpha(T; \theta) = \exp(\beta_0) \left(\frac{T}{T_0}\right)^{\beta_1}$$

**Exponential model:**
$$\alpha(T; \theta) = \exp(\beta_0 + \beta_1(T - T_0))$$

**Logarithmic model:**
$$\alpha(T; \theta) = \exp(\beta_0)\left(1 + \beta_1\log(T/T_0)\right)$$

where $\theta = (\beta_0, \beta_1)$ are calibration parameters and $T_0 = 200$ K is a reference temperature. The parameter ranges are informed by matching the Ulamec (2007) model:
- Powerlaw: $\beta_0 \in [-14.5, -13.5]$, $\beta_1 \in [0.1, 0.9]$ (concave for $0 < \beta_1 < 1$)
- Exponential: $\beta_0 \in [-14.5, -13.5]$, $\beta_1 \in [0.002, 0.010]$
- Logarithmic: $\beta_0 \in [-14.5, -13.5]$, $\beta_1 \in [0.1, 1.0]$ (concave for $\beta_1 > 0$)

**Figure:** The figure below shows 50 random samples from the prior distributions of both toy models, overlaid with the Ulamec reference alpha(T). Generate it with:
```bash
python scripts/generate_theory_plots.py
```

![Alpha models comparison](outputs/alpha_models_prior_samples.png)

---

## 2. Gaussian Process Emulator

### 2.1 Training Data Generation

A training set is constructed by evaluating the forward model on a Design of Experiments (DoE) grid:

- **Input space**: $(W, T_s, \beta_0, \beta_1) \in \mathcal{D}$, with $W$ in m/hour
  - $(W, T_s)$ sampled from artificial data (with optional subsetting)
  - $(\beta_0, \beta_1)$ sampled uniformly using Latin Hypercube Sampling (LHS)
  
- **Output**: $Q_{lc}$ (kW) from solving the PDE for each design point

Default configuration (workflow):
- $n_{\text{design}} = 32$ spatial points (subsampled from full dataset)
- $n_\theta = 20$ parameter samples (LHS)
- Total training points: $32 \times 20 = 640$

Parameter sampling uses LHS within the tightened bounds to ensure dense coverage in regions explored during calibration.

### 2.2 GP Kernel and Fitting

The Gaussian Process uses a simplified composite kernel:

$$K = \sigma_\ell^2 \cdot \text{RBF}(\ell_i) + \sigma_{\text{white}}^2$$

**Why RBF (Radial Basis Function)?**
- **Simplicity**: RBF assumes smooth, infinitely-differentiable functions—appropriate for emulating smooth PDE solutions
- **Efficiency**: Fewer hyperparameters and faster to optimize than Matérn, ideal for exploratory surrogate modeling
- **Per-dimension length scales**: Each input (W, T_s, β₀, β₁) gets its own length scale, learning the importance of each dimension

**Why White Kernel (noise)?**
- **Model discrepancy**: The GP surrogate is an approximation to expensive PDE solves; White noise captures this systematic error
- **Numerical stability**: Small noise prevents ill-conditioning when fitting; variance goes from near-zero to ~1e-5 as learned
- **Realistic uncertainty**: Real observations have sensor noise (~0.1 kW); GP noise term absorbs both measurement and model error

**Why combine them?**
The composite kernel RBF + White provides:
1. A smooth representation of the underlying physics
2. Robustness through white noise capturing model error and numerical precision
3. Hyperparameters (amplitudes, length scales, noise) learned automatically from data via maximum likelihood

The GP is trained on standardized input features ($X_s$) and standardized output targets ($y_s$) using maximum likelihood estimation with $n_{\text{restarts}} = 3$ random initializations of hyperparameters.

### 2.3 Prediction

For a new point $(W, T_s, \beta_0, \beta_1)$, the GP provides:
$$Q_\text{pred} \sim \mathcal{N}(\mu(x), \sigma^2(x))$$

where $\mu(x)$ is the posterior mean and $\sigma(x)$ is the posterior standard deviation, accounting for both model discrepancy and parametric uncertainty.

---

## 3. Bayesian Calibration

### 3.1 Problem Setup

Given synthetic observations $\{(W_i, T_{s,i}, y_i)\}$ with measurement noise $\sigma_{\text{meas}}$, infer the posterior distribution of parameters $\theta = (\beta_0, \beta_1)$ for a given toy model.

### 3.2 Prior Distribution

Gaussian priors informed by matching the Ulamec (2007) model:

**Powerlaw model:**
$$p(\theta) = \mathcal{N}(\beta_0; -14, 0.3^2) \times \mathcal{N}(\beta_1; 0.6, 0.2^2)$$
trun­cated to $\beta_0 \in [-14.5, -13.5]$, $\beta_1 \in [0.1, 0.9]$

**Exponential model:**
$$p(\theta) = \mathcal{N}(\beta_0; -14, 0.3^2) \times \mathcal{N}(\beta_1; 0.006, 0.002^2)$$
trun­cated to $\beta_0 \in [-14.5, -13.5]$, $\beta_1 \in [0.002, 0.010]$

**Logarithmic model:**
$$p(\theta) = \mathcal{N}(\beta_0; -14, 0.3^2) \times \mathcal{N}(\beta_1; 0.5, 0.3^2)$$
trun­cated to $\beta_0 \in [-14.5, -13.5]$, $\beta_1 \in [0.1, 1.0]$

The priors constrain the search to physically plausible regions near the Ulamec approximations.

**Why normal priors here?**
- The parameters are real-valued and centered on Ulamec-fit estimates, so Gaussians capture symmetric uncertainty around those estimates.
- Truncation enforces physical bounds (e.g., concavity/positivity), while keeping a smooth preference for values near the reference fit.
- Using the same family across models keeps calibration comparisons consistent.

**Alternative priors (not used here):**
- **Lognormal** priors are useful for strictly positive, multiplicative-scale parameters. We did not use them because the calibrated parameters are naturally real-valued and centered around Ulamec-fit values, so a normal prior is a simpler and more direct representation of uncertainty.
- **Beta** priors are appropriate for parameters constrained to $[0,1]$ (e.g., probabilities or proportions). Our parameters are not naturally bounded to $[0,1]$, so a beta prior would require an additional re-parameterization step without clear benefit.

### 3.3 Likelihood

The likelihood combines GP prediction uncertainty and measurement noise:

$$p(y_i | W_i, T_{s,i}, \theta) = \mathcal{N}\left(y_i; \mu(W_i, T_{s,i}, \theta), \sigma_\text{total}^2\right)$$

where:
$$\sigma_\text{total}^2 = \sigma_\text{meas}^2 + \sigma_{\text{GP}}^2(W_i, T_{s,i}, \theta)$$

### 3.4 MCMC Sampling

The posterior is explored using the Affine Invariant Ensemble MCMC sampler (`emcee`):

- **Walkers**: $n_{\text{walkers}} = 32$
- **Iterations**: $n_{\text{steps}} = 6000$
- **Burn-in**: 1500 iterations discarded
- **Thinning**: every 10th sample retained

Initial positions are drawn from:
$$p_0 = \theta_{\text{init}} + \mathcal{N}(0, \Sigma_{\text{init}})$$

where $\theta_{\text{init}}$ and $\Sigma_{\text{init}}$ are model-dependent:
- Powerlaw: $\theta_{\text{init}} = (-14, 0.6)$, $\Sigma_{\text{init}} = \text{diag}(0.3^2, 0.2^2)$
- Exponential: $\theta_{\text{init}} = (-14, 0.006)$, $\Sigma_{\text{init}} = \text{diag}(0.3^2, 0.002^2)$
- Logarithmic: $\theta_{\text{init}} = (-14, 0.5)$, $\Sigma_{\text{init}} = \text{diag}(0.3^2, 0.3^2)$

Workflow bounds used in calibration:
- $\beta_0 \in [-14.8, -13.2]$
- $\beta_1$ per model: powerlaw $[0.1, 0.9]$, exponential $[0.001, 0.012]$, logarithmic $[0.1, 1.0]$

The MCMC sampler supports parallel execution via multiprocessing, with the number of parallel processes controlled by the `n_jobs` parameter.

---

## 4. Uncertainty Quantification

### 4.1 Posterior Predictive Distribution

For any $(W, T_s)$, the posterior distribution of $Q_{lc}$ is approximated by:

1. Sample $n_{\text{post}} = 400$ parameter vectors from the MCMC chain
2. For each $\theta^{(j)}$, evaluate the GP mean: $\mu_j = \mu(W, T_s, \theta^{(j)})$
3. Aggregate 95% credible intervals from the empirical distribution of $\{\mu_j\}$: [$Q_{0.025}$, $Q_{0.5}$, $Q_{0.975}$]

This reflects parameter uncertainty propagated through the GP surrogate (without adding GP predictive noise).

### 4.2 Posterior Alpha Distribution

Similarly, the posterior distribution of the thermal diffusivity function $\alpha(T; \theta)$ is:

$$\alpha^{(j)}(T) = \begin{cases}
\exp(\beta_0^{(j)}) (T / 200)^{\beta_1^{(j)}} & \text{(powerlaw)} \\
\exp(\beta_0^{(j)} + \beta_1^{(j)}(T - 200)) & \text{(exponential)} \\
\exp(\beta_0^{(j)})\left(1 + \beta_1^{(j)}\log(T/200)\right) & \text{(logarithmic)}
\end{cases}$$

Credible bands are computed element-wise over the posterior samples.

### 4.4 PCE-Based UQ (optional)

As an alternative to direct GP evaluation, a Polynomial Chaos Expansion (PCE) can be fit to the GP outputs as a low-order surrogate in $(\beta_0, \beta_1)$. For two parameters and low polynomial degree, the number of required training evaluations is modest (tens of runs), and the resulting surrogate enables fast credible band computation over the full $(W, T_s)$ grid.

### 4.3 UQ Scenario Grid (workflow)

The workflow UQ uses a dense linear grid:
- $W \in [0.05, 5.0]$ m/hour, 50 points (linear spacing)
- $T_s \in [80, 260]$ K, 30 points (linear spacing)

---

## References

- **Ulamec, S.** (2007). Thermal properties and processes in planetary ices. In *Europa* (pp. 427–457). University of Arizona Press.
- **Rasmussen, C. E., & Williams, C. K. I.** (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- **Foreman-Mackey, D., et al.** (2013). *emcee*: The MCMC Hammer. *PASP*, 125(925), 306–312.
- **Gelman, A., et al.** (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
- **Johnson, N. L., Kotz, S., & Balakrishnan, N.** (1994). *Continuous Univariate Distributions, Vol. 1* (2nd ed.). Wiley.
