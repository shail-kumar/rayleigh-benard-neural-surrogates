"""Locations of generated trajectories and figures.

Paths resolve from the installed package rather than the working directory.
Override either one by environment variable:

    RBCS_DATA_DIR=/mnt/scratch/rbc  rbcs-simulate
"""

import os
from pathlib import Path


def _find_project_root() -> Path:
    """Locate the repo root by its pyproject.toml, falling back to the current
    directory when the package is installed without a repo around it."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


PROJECT_ROOT = _find_project_root()

DATA_DIR = Path(os.environ.get("RBCS_DATA_DIR") or PROJECT_ROOT / "data")
RESULTS_DIR = Path(os.environ.get("RBCS_RESULTS_DIR") or PROJECT_ROOT / "results")


def data_path(name: str) -> Path:
    """Absolute path for a trajectory file, creating DATA_DIR on demand."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / name


def result_path(name: str) -> Path:
    """Absolute path for a generated figure or results file, creating
    RESULTS_DIR on demand."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name
