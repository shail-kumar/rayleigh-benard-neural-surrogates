"""Compare FNO, reservoir computing and persistence: accuracy, physics, speed."""

import argparse
import json
import time

import numpy as np
import torch

from rbcs.baselines import POD, ReservoirSurrogate
from rbcs.data import find_datasets, load_dataset, split_by_trajectory
from rbcs.dataset import channel_stats
from rbcs.fno import FNO2d
from rbcs.paths import result_path
from rbcs.solver import RayleighBenard


def relative_mse(pred, target):
    num = ((pred - target) ** 2).mean(axis=(0, 2, 3))
    den = target.var(axis=(0, 2, 3)) + 1e-12
    return float((num / den).mean())


def nusselt(fields, solver, ra, pr):
    """Volume-averaged Nu for a stack of (theta, omega) frames."""
    out = []
    for f in fields:
        _, w = solver.velocities(f[1].astype(np.float64))
        out.append(1.0 + np.sqrt(ra * pr) * float(np.mean(w * f[0])))
    return np.array(out)


def x_spectrum(fields):
    """Temperature variance per horizontal wavenumber, averaged over z and time."""
    amp = np.abs(np.fft.rfft(fields[:, 0], axis=-2)) ** 2
    return amp.mean(axis=(0, 2))


def load_fno(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    m = FNO2d(width=a["width"], modes_x=a["modes_x"], modes_z=a["modes_z"], depth=a["depth"])
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


@torch.no_grad()
def fno_rollout(model, x0, n):
    x = torch.from_numpy(x0[None].copy())
    out = []
    for _ in range(n):
        x = model(x)
        out.append(x[0].numpy().copy())
    return np.array(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="fno.pt")
    p.add_argument("--rank", type=int, default=512)
    p.add_argument("--shifts", type=int, default=8)
    p.add_argument("--train-shifts", type=int, default=16)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--stride", type=int, default=6)
    args = p.parse_args()

    fields, _, attrs = load_dataset(find_datasets())
    ra, pr = float(attrs["ra"]), float(attrs["pr"])
    tr, va, te = split_by_trajectory(fields.shape[0])
    mean, std = channel_stats(fields[tr])
    solver = RayleighBenard(
        ra=ra, pr=pr, nx=int(attrs["nx"]), nz=int(attrs["nz"]),
        aspect=float(attrs["aspect"]), dt=float(attrs["dt"]),
    )
    res = {"ra": ra, "horizon": args.horizon}

    model, ck = load_fno(result_path(args.checkpoint))
    res["fno"] = {"params": sum(q.numel() for q in model.parameters()),
                  "epoch": ck["epoch"], "val_at_checkpoint": ck["val"]}
    print(f"FNO: {res['fno']['params']:,} params, best epoch {ck['epoch']}", flush=True)

    pod = POD(rank=args.rank, shifts=args.shifts).fit(fields[tr][:, :: args.stride], mean, std)
    coeffs = [pod.encode(fields[i]) for i in range(fields.shape[0])]
    seqs = [coeffs[i] for i in tr]
    rng = np.random.default_rng(0)
    for i in tr:
        for _ in range(args.train_shifts - 1):
            seqs.append(pod.encode(np.roll(fields[i], int(rng.integers(fields.shape[-2])), axis=-2)))
    esn = ReservoirSurrogate(n_inputs=args.rank, n_reservoir=800).fit(seqs)
    res["rc"] = {"rank": args.rank, "pod_params": args.rank * int(np.prod(fields.shape[2:]))}
    print("RC fitted", flush=True)

    # ---- single step -------------------------------------------------------
    w = 20
    for name, idx in (("val", va), ("test", te)):
        f = fields[idx[0]]
        with torch.no_grad():
            pf = model(torch.from_numpy(f[w:-1].copy())).numpy()
        res.setdefault("single_step", {})[name] = {
            "fno": relative_mse(pf, f[w + 1 :]),
            "rc": relative_mse(pod.decode(esn.predict_next(coeffs[idx[0]])[w:-1]), f[w + 1 :]),
            "persistence": relative_mse(f[w:-1], f[w + 1 :]),
        }
        print(f"single-step {name}: {res['single_step'][name]}", flush=True)

    # ---- rollout -----------------------------------------------------------
    f = fields[te[0]]
    start = 100
    truth = f[start + 1 : start + 1 + args.horizon]
    roll = {
        "fno": fno_rollout(model, f[start], args.horizon),
        "rc": pod.decode(esn.rollout(coeffs[te[0]][: start + 1], args.horizon)),
        "persistence": np.repeat(f[start][None], args.horizon, axis=0),
    }
    res["rollout"] = {
        k: [relative_mse(v[i : i + 1], truth[i : i + 1]) for i in range(args.horizon)]
        for k, v in roll.items()
    }
    for k, v in res["rollout"].items():
        print(f"rollout {k:>11}: step1 {v[0]:.4f}  step5 {v[4]:.4f}  step{args.horizon} {v[-1]:.4f}", flush=True)

    # ---- physics -----------------------------------------------------------
    res["nusselt"] = {"truth": nusselt(truth, solver, ra, pr).tolist()}
    for k, v in roll.items():
        res["nusselt"][k] = nusselt(v, solver, ra, pr).tolist()
    res["spectrum"] = {"truth": x_spectrum(truth).tolist()}
    for k, v in roll.items():
        res["spectrum"][k] = x_spectrum(v).tolist()
    print(f"Nu truth mean {np.mean(res['nusselt']['truth']):.2f}, "
          f"fno {np.mean(res['nusselt']['fno']):.2f}, rc {np.mean(res['nusselt']['rc']):.2f}", flush=True)

    # ---- speed -------------------------------------------------------------
    # Both sides on one thread. The solver advances on a single core, so timing
    # the FNO against torch's default pool would not be like for like.
    x1 = torch.from_numpy(f[start][None].copy())
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.no_grad():
        for _ in range(3):
            model(x1)
        t0 = time.perf_counter()
        for _ in range(20):
            model(x1)
        fno_ms = (time.perf_counter() - t0) / 20 * 1000
    torch.set_num_threads(threads)
    si = float(attrs["sample_interval"])
    solver.advance(si)
    t0 = time.perf_counter()
    for _ in range(4):
        solver.advance(si)
    solver_s = (time.perf_counter() - t0) / 4
    res["speed"] = {"fno_ms": fno_ms, "solver_s_per_step": solver_s,
                    "speedup": solver_s / (fno_ms / 1000)}
    print(f"speed: FNO {fno_ms:.0f} ms vs solver {solver_s:.1f} s -> {res['speed']['speedup']:.0f}x", flush=True)

    out = result_path("comparison.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"\n{out}")


if __name__ == "__main__":
    main()
