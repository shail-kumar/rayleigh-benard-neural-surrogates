"""Fit the reservoir-computing baseline: POD reduction plus an ESN."""

import argparse
import json
import time

import numpy as np

from rbcs.baselines import POD, ReservoirSurrogate
from rbcs.data import find_datasets, load_dataset, split_by_trajectory
from rbcs.dataset import channel_stats
from rbcs.paths import result_path


def relative_mse(pred, target):
    """Same metric as the FNO: per-channel MSE over channel variance."""
    num = ((pred - target) ** 2).mean(axis=(0, 2, 3))
    den = target.var(axis=(0, 2, 3)) + 1e-12
    return float((num / den).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--shifts", type=int, default=8)
    p.add_argument("--reservoir", type=int, default=800)
    p.add_argument("--spectral-radius", type=float, default=0.95)
    p.add_argument("--leak", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--stride", type=int, default=6, help="subsample for the POD fit")
    p.add_argument("--train-shifts", type=int, default=1,
                   help="shifted copies of each training trajectory for the ESN")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    fields, _, _ = load_dataset(find_datasets())
    tr, va, te = split_by_trajectory(fields.shape[0])
    mean, std = channel_stats(fields[tr])

    t0 = time.perf_counter()
    pod = POD(rank=args.rank, shifts=args.shifts, seed=args.seed)
    pod.fit(fields[tr][:, :: args.stride], mean, std)
    floor_val = pod.reconstruction_error(fields[va])
    floor_test = pod.reconstruction_error(fields[te])
    print(
        f"POD rank {args.rank}, {args.shifts} shifts, {time.perf_counter() - t0:.0f}s\n"
        f"  reconstruction floor: val {floor_val:.4f}  test {floor_test:.4f}",
        flush=True,
    )

    coeffs = [pod.encode(fields[i]) for i in range(fields.shape[0])]

    # The ESN learns dynamics in POD coordinates, which are phase-specific:
    # shifting a training trajectory doubles its error. Shifted copies teach
    # phase-invariant dynamics, the same fix the basis itself needed.
    train_seqs = [coeffs[i] for i in tr]
    if args.train_shifts > 1:
        rng = np.random.default_rng(args.seed)
        for i in tr:
            for _ in range(args.train_shifts - 1):
                sh = int(rng.integers(fields.shape[-2]))
                train_seqs.append(pod.encode(np.roll(fields[i], sh, axis=-2)))
        print(f"ESN training sequences: {len(train_seqs)}", flush=True)

    t0 = time.perf_counter()
    esn = ReservoirSurrogate(
        n_inputs=args.rank,
        n_reservoir=args.reservoir,
        spectral_radius=args.spectral_radius,
        leak_rate=args.leak,
        ridge_alpha=args.alpha,
        seed=args.seed,
    )
    esn.fit(train_seqs)
    print(f"ESN {args.reservoir} units fitted in {time.perf_counter() - t0:.0f}s", flush=True)

    out = {"pod_floor_val": floor_val, "pod_floor_test": floor_test, "args": vars(args)}
    for name, idx in (("val", va), ("test", te)):
        errs = []
        for i in idx:
            c = coeffs[i]
            w = 20
            pred_c = esn.predict_next(c)[w:-1]      # predicts c[t+1] from c[t]
            pred = pod.decode(pred_c)
            errs.append(relative_mse(pred, fields[i][w + 1 :]))
        out[f"rc_{name}"] = float(np.mean(errs))
        print(f"  RC one-step {name}: {out[f'rc_{name}']:.4f}", flush=True)

    # persistence, measured here so the harness agrees with the dataset analysis
    for name, idx in (("val", va), ("test", te)):
        e = [relative_mse(fields[i][20:-1], fields[i][21:]) for i in idx]
        out[f"persistence_{name}"] = float(np.mean(e))
        print(f"  persistence {name}: {out[f'persistence_{name}']:.4f}", flush=True)

    path = result_path("rc_results.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"\n{path}")


if __name__ == "__main__":
    main()
