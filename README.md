# Cryobot Digital Twin

Unified FEniCSx forward solver with Bayesian calibration and GP emulator.

## Setup
1. Create/activate your environment (example):
   ```bash
   conda create -n fenicsx -c conda-forge python=3.11 fenics-dolfinx mpi4py petsc4py numpy scipy pandas matplotlib scikit-learn emcee joblib
   conda activate fenicsx
   ```
   Or install from the frozen list:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure `mpirun`/MPI is available (dolfinx requires it even in serial mode).

## Workflow
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

## Notes
- Unified forward solver: `src/cf4dt/forward.py` with `compute_Qlc(material_model='ulamec'|'powerlaw'|'exponential', theta=...)` (steady-state axisymmetric conduction).
- Truth data uses `material_model='ulamec'`; emulator/calibration use the parameterized models.
