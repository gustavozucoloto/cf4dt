# Cryobot Digital Twin

Unified FEniCSx forward solver with Bayesian calibration and GP emulator.

## Setup

### Local Development
Create/activate your environment using conda (or mamba):
```bash
conda create -n fenicsx -c conda-forge python=3.11 fenics-dolfinx mpi4py petsc4py numpy scipy pandas matplotlib scikit-learn emcee joblib
conda activate fenicsx
```

Alternatively, install from the frozen requirements:
```bash
pip install -r requirements.txt
```

Ensure `mpirun`/MPI is available (dolfinx requires it even in serial mode).

### HPC Cluster Setup (via Micromamba)
Since our HPC cluster do not have conda available, use **micromamba** instead:

1. **Install micromamba** (if not already installed):
   ```bash
   "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. **Create the fenicsx environment**:
   ```bash
   micromamba create -n fenicsx -c conda-forge python=3.11 fenics-dolfinx mpi4py petsc4py numpy scipy pandas matplotlib scikit-learn emcee joblib
   micromamba activate fenicsx
   ```

3. **Verify the environment**:
   ```bash
   python -c "from dolfinx import __version__; print('dolfinx:', __version__)"
   python -c "from mpi4py import MPI; print('MPI vendor:', MPI.get_vendor())"
   ```

**Note:** Do not load system OpenMPI when using conda-forge dolfinx (which uses MPICH). The SLURM sbatch file handles this automatically.

## Workflow
### Running Locally
Execute the workflow steps manually:

1. Generate synthetic truth data (Ulamec material model):
   
   Edit velocity and temperature lists in `scripts/generate_artificial_data.py`:
   ```python
   # Define your custom velocities (in meters per hour)
   W_mph_list = [0.05, 0.5, 2.0, 5.0]
   
   # Define your custom temperatures (in Kelvin)
   Ts_K_list = [80.0, 125.0, 170.0, 215.0, 260.0]
   ```
   
   Then run:
   ```bash
   python scripts/generate_artificial_data.py
   ```
   
   This creates a full grid: 4 velocities × 5 temperatures = 20 simulation points.
   To use parallel execution, edit `n_jobs` parameter in the script.

2. Train GP emulator on reduced model (choose powerlaw or exponential):
   ```bash
   python scripts/build_gp_emulator.py --model powerlaw \
       --data data/artificial_Qlc_data.csv \
       --out data/gp_powerlaw.joblib

      # Try a Matern(ν=2.5) kernel (often more flexible than RBF):
      python scripts/build_gp_emulator.py --model logarithmic --kernel matern_2p5 \
         --data data/artificial_Qlc_data.csv --out data/gp_logarithmic_matern25.joblib
   ```
   For exponential: `--model exponential --out data/gp_exponential.joblib`.
   Tune cost via `--n-theta` and `--subset`.

3. Run Bayesian calibration (emcee) using the GP:
   ```bash
      python scripts/run_bayesian_calibration.py --model powerlaw \
       --data data/artificial_Qlc_data.csv \
       --gp data/gp_powerlaw.joblib \
       --out data/posterior_powerlaw.npy

      # To keep MCMC samples inside the GP training box (recommended):
      python scripts/run_bayesian_calibration.py --model powerlaw --prior truncnorm \
         --data data/artificial_Qlc_data.csv --gp data/gp_powerlaw.joblib --out data/posterior_powerlaw.npy

      # Legacy behavior (can query GP outside training box):
      python scripts/run_bayesian_calibration.py --model powerlaw --prior gaussian \
         --data data/artificial_Qlc_data.csv --gp data/gp_powerlaw.joblib --out data/posterior_powerlaw.npy
   ```

4. Generate UQ plots:
   ```bash
   python scripts/run_uq.py --model powerlaw \
       --gp data/gp_powerlaw.joblib \
       --posterior data/posterior_powerlaw.npy \
       --out-prefix outputs/uq_powerlaw
   ```

### Running on HPC Cluster via SLURM
This repo ships a 2-stage, parallel SLURM workflow that keeps only **quick** and **medium** calibration cases:

1) Train 6 GPs in parallel (powerlaw/exponential/logarithmic × RBF/Matern)
2) After GP training succeeds, run calibration + UQ for quick + medium (12 tasks total)

Submit from the login node:

```bash
./submit_parallel_workflow.sh
```

**How the SLURM workflow works:**

1. **Stage 1 (GP training)**: `run_batch_train_gps.sbatch` is submitted as a 6-task array (max 6 in flight).
2. **Stage 2 (calibration + UQ)**: `run_batch_calib_uq.sbatch` is submitted as a 12-task array (quick + medium) with an `afterok` dependency on Stage 1.

2. **Environment Setup**:
   - Loads GCC module (required for dolfinx)
   - Configures micromamba paths and activates the `fenicsx` environment
   - Sets thread limits for BLAS/LAPACK libraries (numpy, scipy, sklearn)
   - Validates that the micromamba environment exists before proceeding

3. **MPI Configuration**:
   - Uses `srun --mpi=none` to avoid SLURM PMI/PMIx conflicts with conda-forge MPI
   - The micromamba environment provides its own MPICH

4. **Workflow Execution**:
   - Runs all 4 workflow steps (data generation, GP training, Bayesian calibration, UQ plots)
   - Loops over both `powerlaw` and `exponential` material models
   - Creates output directories (`logs/`, `data/`, `outputs/`) if they don't exist
   - Captures stdout/stderr to `logs/cryobot_dt_<jobid>.out` and `.err`

**Monitor and debug:**

```bash
# Check job status
squeue --me

# View real-time output (replace JOBID)
tail -f logs/cryobot_dt_<JOBID>.out

# View error log if job fails
cat logs/cryobot_dt_<JOBID>.err
```

## Parallel Execution

All workflow steps support parallel execution via the `--n-jobs` parameter to accelerate high-throughput tasks on multi-core systems:

### Basic Usage

Edit the `n_jobs` parameter in each script:
```python
# In scripts/generate_artificial_data.py
generate_artificial_data(
    ...,
    n_jobs=1,  # Change to 16 for parallel execution
)
```

or set it during workflow execution (see scripts for details).

### Parallel Steps in Workflow

1. **Data Generation** (independent forward simulations):
   - Edit `n_jobs` parameter in `scripts/generate_artificial_data.py`
   - Parallelizes forward model evaluations over design points (W, Ts)
   - Each worker uses isolated MPI communicator to avoid conflicts
   - Expected speedup: ~4-8x on 16 cores (I/O and initialization overhead)

2. **GP Training** (building training set):
   ```bash
   python scripts/build_gp_emulator.py --model powerlaw \
       --data data/artificial_Qlc_data.csv \
       --out data/gp_powerlaw.joblib \
       --n-jobs 16
   ```
   - Parallelizes forward evaluations over (W, Ts, theta) grid
   - GP training itself is serial (fast fitting step)
   - Expected speedup: ~6-10x on 16 cores

3. **Bayesian Calibration** (MCMC with emcee):
   ```bash
   python scripts/run_bayesian_calibration.py --model powerlaw \
       --data data/artificial_Qlc_data.csv \
       --gp data/gp_powerlaw.joblib \
       --out data/posterior_powerlaw.npy \
       --n-jobs 16
   ```
   - Uses emcee's built-in multiprocessing for parallel walkers
   - Each walker evaluates GP likelihood independently
   - Expected speedup: ~10-15x on 16 cores (minimal overhead)

### Cluster Integration
The SLURM scripts read `$SLURM_CPUS_PER_TASK` and pass it to `--n-jobs` for GP training and calibration.
To adjust parallelism of *how many array tasks run at once*, set:

```bash
TRAIN_PARALLELISM=6 CALIB_PARALLELISM=12 ./submit_parallel_workflow.sh
```

### Performance Notes
- Use `n_jobs=1` (default) for serial execution or small problem sizes
- Optimal `n_jobs` ≈ number of physical cores (not hyperthreads)
- Monitor with `htop` to verify core utilization
- Each parallel worker uses MPI.COMM_SELF to isolate dolfinx communicators

## Notes
- Unified forward solver: `src/cf4dt/forward.py` with `compute_Qlc(material_model='ulamec'|'powerlaw'|'exponential', theta=...)` (transient axisymmetric heat conduction using thermal diffusivity formulation).
- All models use alpha (thermal diffusivity) formulation: `dT/dt = (1/r) d/dr(alpha(T) * r * dT/dr)`.
- Truth data uses `material_model='ulamec'` with alpha computed from k/(rho*cp); emulator/calibration use the parameterized toy models.
- **Priors and GP training**: Tightened to explore regions near Ulamec fits (powerlaw: β₀∈[-14.5,-13.5], β₁∈[0.5,1.7]; exponential: β₀∈[-14.5,-13.5], β₁∈[0.002,0.010]).
- **GP kernel**: Supports `--kernel rbf` (default) and `--kernel matern_2p5` (Matern $\nu=2.5$) for the emulator.
- **White noise**: Fixed across runs (shared for RBF/Matern) to keep comparisons apples-to-apples.
