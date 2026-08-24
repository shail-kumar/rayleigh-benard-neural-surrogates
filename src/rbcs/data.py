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


def dataset_name(ra=None, batch=1):
    """Dataset filename, labelled by Rayleigh number and batch."""
    ra = config.RA if ra is None else ra
    exp = f"{ra:.0e}" if float(f"{ra:.0e}") == ra else f"{ra:.1e}"
    exp = exp.replace("e+0", "e").replace("e+", "e")
    suffix = "" if batch == 1 else f"_b{batch}"
    return f"trajectories_Ra{exp}{suffix}.h5"


def find_datasets(ra=None, data_dir=None):
    """All batches for one Rayleigh number, in batch order."""
    from rbcs.paths import DATA_DIR

    d = Path(data_dir or DATA_DIR)
    stem = dataset_name(ra).removesuffix(".h5")
    return sorted(d.glob(f"{stem}*.h5"), key=lambda p: len(p.name))


def batch_seeds(batch, n):
    """Seeds for a batch, so batches never overlap."""
    return list(range((batch - 1) * n, batch * n))


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


def load_dataset(paths):
    """Return (fields, nusselt, attrs), concatenating batches along trajectory.

    Raises if batches disagree on the physics or the grid.
    """
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    fields, nusselt, seeds, attrs = [], [], [], None
    for p in paths:
        with h5py.File(p, "r") as f:
            a = dict(f.attrs)
            if attrs is None:
                attrs = a
            else:
                for k in ("ra", "pr", "nx", "nz", "aspect", "dt", "sample_interval"):
                    if a[k] != attrs[k]:
                        raise ValueError(f"{Path(p).name} has {k}={a[k]}, expected {attrs[k]}")
            seeds.extend(np.atleast_1d(a["seeds"]).tolist())
            fields.append(f["fields"][...])
            nusselt.append(f["nusselt"][...])
    attrs["seeds"] = seeds
    return np.concatenate(fields), np.concatenate(nusselt), attrs


def split_by_trajectory(n_traj, n_val=1, n_test=1):
    """Index split. Splitting by time step instead would leak the test set."""
    if n_traj <= n_val + n_test:
        raise ValueError(f"{n_traj} trajectories cannot fill a {n_val}/{n_test} val/test split")
    idx = np.arange(n_traj)
    return idx[: -(n_val + n_test)], idx[-(n_val + n_test) : -n_test], idx[-n_test:]
