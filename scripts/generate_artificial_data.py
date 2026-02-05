#!/usr/bin/env python
"""Generate artificial data with custom velocity and temperature lists."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cf4dt.data_generation import generate_artificial_data


def main():
    W_mph_list = [0.05, 0.5, 2.0, 5.0]
    
    Ts_K_list = [80.0, 125.0, 170.0, 215.0, 260.0]
    # ==================================================================
    
    # This will create a n_W × n_Ts grid with all combinations
    # Current: 4 velocities × 5 temperatures = 20 simulation points
    generate_artificial_data(
        out_csv="data/artificial_Qlc_data.csv",
        W_mph_list=W_mph_list,
        Ts_K_list=Ts_K_list,
        seed_noise=2,
        sigma_kW=1e-7,  # ~10% noise level (signal is ~1e-6 kW)
        Ro=0.1,
        L=3.7,
        Tm=273.15,
        Rinf_offset=1.0,
        num_cells=400,
        p_grade=3.0,
        num_steps=1000,
        dt_ratio=1.03,
        n_jobs=1,  # Set to match SLURM_CPUS_PER_TASK for parallel execution
    )


if __name__ == "__main__":
    main()
