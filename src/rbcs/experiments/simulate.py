"""Generate the training trajectories."""

import argparse
import time

from rbcs import config
from rbcs.data import batch_seeds, dataset_name, default_workers, write_dataset
from rbcs.paths import data_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", "--n-trajectories", type=int, default=8)
    p.add_argument("-o", "--out", default=None, help="default: named by Ra")
    p.add_argument("--snapshots", type=int, default=config.N_SNAPSHOTS)
    p.add_argument("--interval", type=float, default=config.SAMPLE_INTERVAL)
    p.add_argument("--spinup", type=float, default=config.SPINUP_TIME)
    p.add_argument("-j", "--workers", type=int, default=None, help="default: one per trajectory")
    p.add_argument("-b", "--batch", type=int, default=1, help="seeds and filename are offset per batch")
    args = p.parse_args()

    out = data_path(args.out or dataset_name(batch=args.batch))
    seeds = batch_seeds(args.batch, args.n_trajectories)
    workers = default_workers(len(seeds)) if args.workers is None else args.workers
    print(
        f"batch {args.batch}: {len(seeds)} trajectories (seeds {seeds[0]}-{seeds[-1]}) on "
        f"{workers} workers, {args.snapshots} snapshots "
        f"every {args.interval} time units, spinup {args.spinup}\n-> {out}"
    )

    t0 = time.perf_counter()
    write_dataset(
        out,
        seeds,
        workers=workers,
        n_snapshots=args.snapshots,
        sample_interval=args.interval,
        spinup=args.spinup,
    )
    print(f"\n{out} written in {(time.perf_counter() - t0) / 60:.1f} min")
    print(f"{out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
