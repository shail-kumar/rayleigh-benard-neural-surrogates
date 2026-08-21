"""Generate and load convection trajectories.

Snapshots are stored as (n_traj, n_snapshots, 2, nx, nz) float32, channel 0
temperature perturbation and channel 1 vorticity. Those two determine the
state; streamfunction and velocities follow from the vorticity.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import h5py
import numpy as np

from rbcs import config
from rbcs.solver import RayleighBenard

CHANNELS = ("theta", "omega")


def default_workers(n_jobs):
    """One process per trajectory, capped by core count."""
    return max(1, min(n_jobs, os.cpu_count() or 1))


def simulate(seed, n_snapshots=None, sample_interval=None, spinup=None, progress=None):
    """Run one trajectory past its transient and return (fields, nusselt)."""
    n_snapshots = n_snapshots or config.N_SNAPSHOTS
    sample_interval = sample_interval or config.SAMPLE_INTERVAL
    spinup = config.SPINUP_TIME if spinup is None else spinup

    s = RayleighBenard(
        ra=config.RA,
        pr=config.PR,
        nx=config.NX,
        nz=config.NZ,
        aspect=config.ASPECT,
        dt=config.DT,
        seed=seed,
    )
    s.advance(spinup)
    if not s.is_finite():
        raise RuntimeError(f"trajectory seed={seed} diverged during spinup")

    fields = np.empty((n_snapshots, 2, config.NX, config.NZ), dtype=np.float32)
    nusselt = np.empty(n_snapshots, dtype=np.float32)
    for i in range(n_snapshots):
        fields[i, 0] = s.theta
        fields[i, 1] = s.omega
        nusselt[i] = s.nusselt()
        if progress and i % progress == 0:
            progress_msg(seed, i, n_snapshots, nusselt[i])
        s.advance(sample_interval)

    if not np.isfinite(fields).all():
        raise RuntimeError(f"trajectory seed={seed} produced non-finite fields")
    return fields, nusselt


def progress_msg(seed, i, n, nu):
    print(f"  seed {seed}: snapshot {i + 1}/{n}, Nu = {nu:.2f}", flush=True)


def write_dataset(path, seeds, workers=None, **kwargs):
    """Simulate one trajectory per seed, in parallel, and write them to HDF5.

    Renamed into place on success, so an interrupted run leaves nothing that
    looks complete.
    """
    workers = default_workers(len(seeds)) if workers is None else workers
    tmp = Path(path).with_suffix(".partial")
    with h5py.File(tmp, "w") as f:
        f.attrs.update(
            ra=config.RA,
            pr=config.PR,
            nx=config.NX,
            nz=config.NZ,
            aspect=config.ASPECT,
            dt=config.DT,
            sample_interval=kwargs.get("sample_interval") or config.SAMPLE_INTERVAL,
            spinup=config.SPINUP_TIME if kwargs.get("spinup") is None else kwargs["spinup"],
            channels=list(CHANNELS),
            seeds=list(seeds),
        )
        fields_ds = nusselt_ds = None
        worker = partial(simulate, **kwargs)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = pool.map(worker, seeds)
        for j, ((fields, nusselt), seed) in enumerate(zip(results, seeds)):
            if fields_ds is None:
                # Uncompressed: gzip barely shrinks turbulent float32 fields.
                fields_ds = f.create_dataset(
                    "fields",
                    shape=(len(seeds), *fields.shape),
                    dtype=np.float32,
                    chunks=(1, 1, 2, config.NX, config.NZ),
                )
                nusselt_ds = f.create_dataset(
                    "nusselt", shape=(len(seeds), len(nusselt)), dtype=np.float32
                )
            fields_ds[j] = fields
            nusselt_ds[j] = nusselt
            print(f"  seed {seed}: done, Nu = {nusselt.mean():.2f} +- {nusselt.std():.2f}", flush=True)
    os.replace(tmp, path)
    return path


def load_dataset(path):
    """Return (fields, nusselt, attrs) with fields read fully into memory."""
    with h5py.File(path, "r") as f:
        return f["fields"][...], f["nusselt"][...], dict(f.attrs)


def split_by_trajectory(n_traj, n_val=1, n_test=1):
    """Index split. Splitting by time step instead would leak the test set."""
    if n_traj <= n_val + n_test:
        raise ValueError(f"{n_traj} trajectories cannot fill a {n_val}/{n_test} val/test split")
    idx = np.arange(n_traj)
    return idx[: -(n_val + n_test)], idx[-(n_val + n_test) : -n_test], idx[-n_test:]
