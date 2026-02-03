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
Since most HPC clusters do not have conda available, use **micromamba** instead:

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
   ```bash
   python scripts/generate_artificial_data.py --out data/artificial_Qlc_data.csv
   ```
   Useful flags: `--n-lhs`, `--num-cells`, `--p-grade`, `--Ts-min/max`, `--W-min/max`.

2. Train GP emulator on reduced model (choose powerlaw or exponential):
   ```bash
   python scripts/build_gp_emulator.py --model powerlaw \
       --data data/artificial_Qlc_data.csv \
       --out data/gp_powerlaw.joblib
   ```
   For exponential: `--model exponential --out data/gp_exponential.joblib`.
   Tune cost via `--n-theta` and `--subset`.

3. Run Bayesian calibration (emcee) using the GP:
   ```bash
   python scripts/run_bayesian_calibration.py --model powerlaw \
       --data data/artificial_Qlc_data.csv \
       --gp data/gp_powerlaw.joblib \
       --out data/posterior_powerlaw.npy
   ```

4. Generate UQ plots:
   ```bash
   python scripts/run_uq.py --model powerlaw \
       --gp data/gp_powerlaw.joblib \
       --posterior data/posterior_powerlaw.npy \
       --out-prefix outputs/uq_powerlaw
   ```

### Running on HPC Cluster via SLURM
Use the `run_workflow.sbatch` script to submit the entire workflow to the cluster:

```bash
sbatch run_workflow.sbatch
```

**How the sbatch script works:**

1. **SLURM Configuration** (`#SBATCH` directives):
   - `--job-name=cryobot_dt`: Job identifier
   - `--cpus-per-task=16`: Request 16 CPU cores (adjust as needed)
   - `--mem=5GB`: Memory allocation
   - `--time=12:00:00`: Maximum runtime (12 hours)
   - `-A thes2143`: Charge to account `thes2143`
   - `-p c23ms`: Partition/queue (adjust to your cluster)

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

## Notes
- Unified forward solver: `src/cf4dt/forward.py` with `compute_Qlc(material_model='ulamec'|'powerlaw'|'exponential', theta=...)` (steady-state axisymmetric conduction).
- Truth data uses `material_model='ulamec'`; emulator/calibration use the parameterized models.
