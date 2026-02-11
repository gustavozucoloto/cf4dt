#!/usr/bin/env python
"""Convenience script to generate alpha(T) graphs for toy models.

This script simply delegates to cf4dt.utils.create_model_graphs.main(),
so it should be run from the project root:

    python scripts/run_create_model_graphs.py

It will produce four PNG files under ``outputs/``:

- outputs/alpha_powerlaw_vary_beta0.png
- outputs/alpha_powerlaw_vary_beta1.png
- outputs/alpha_exponential_vary_beta0.png
- outputs/alpha_exponential_vary_beta1.png
"""

from pathlib import Path
import sys

# Ensure src/ is on sys.path so cf4dt can be imported when running as a script.
ROOT = Path.cwd()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cf4dt.utils.create_model_graphs import main


if __name__ == "__main__":
    main()
