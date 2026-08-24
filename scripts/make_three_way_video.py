"""Solver, FNO and reservoir computing side by side, with their Nusselt numbers.

    python social_media/make_three_way_video.py [--steps 150] [--fps 6]
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from rbcs.baselines import POD, ReservoirSurrogate
from rbcs.data import find_datasets, load_dataset, split_by_trajectory
from rbcs.dataset import channel_stats
from rbcs.experiments.evaluate import fno_rollout, load_fno, nusselt
from rbcs.paths import result_path
from rbcs.solver import RayleighBenard

# Single-thread costs per 0.5 time units, measured on one core (see WRITEUP section 5).
SOLVER_S_PER_STEP = 5.22
FNO_S_PER_STEP = 0.057


def rc_rollout(fields, tr, traj_idx, start, steps, rank, shifts, train_shifts, stride):
    mean, std = channel_stats(fields[tr])
    pod = POD(rank=rank, shifts=shifts).fit(fields[tr][:, ::stride], mean, std)
    seqs = [pod.encode(fields[i]) for i in tr]
    rng = np.random.default_rng(0)
    for i in tr:
        for _ in range(train_shifts - 1):
            seqs.append(pod.encode(np.roll(fields[i], int(rng.integers(fields.shape[-2])), axis=-2)))
    esn = ReservoirSurrogate(n_inputs=rank, n_reservoir=800).fit(seqs)
    warm = pod.encode(fields[traj_idx])[: start + 1]
    t0 = time.perf_counter()
    coeffs = esn.rollout(warm, steps)
    out = pod.decode(coeffs)
    return out, (time.perf_counter() - t0) / steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--start", type=int, default=100)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--rank", type=int, default=512)
    p.add_argument("--shifts", type=int, default=8)
    p.add_argument("--train-shifts", type=int, default=16)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--refit", action="store_true", help="ignore the cached RC rollout")
    p.add_argument("-o", "--out", default="three_way.mp4")
    args = p.parse_args()

    fields, _, attrs = load_dataset(find_datasets())
    tr, _, te = split_by_trajectory(fields.shape[0])
    traj = te[0]
    ra, pr = float(attrs["ra"]), float(attrs["pr"])
    nx, nz, aspect = int(attrs["nx"]), int(attrs["nz"]), float(attrs["aspect"])
    dt_s = float(attrs["sample_interval"])

    truth = fields[traj][args.start + 1 : args.start + 1 + args.steps]
    n = len(truth)

    model, _ = load_fno(result_path("fno.pt"))
    fno = fno_rollout(model, fields[traj][args.start], n)

    cache = result_path(f"rc_rollout_t{traj}_s{args.start}_n{n}.npz")
    if cache.exists() and not args.refit:
        z_ = np.load(cache)
        rc, rc_s_per_step = z_["rc"], float(z_["s_per_step"])
    else:
        print("fitting the reservoir baseline, this takes a few minutes", flush=True)
        rc, rc_s_per_step = rc_rollout(fields, tr, traj, args.start, n,
                                       args.rank, args.shifts, args.train_shifts, args.stride)
        np.savez(cache, rc=rc, s_per_step=rc_s_per_step)

    solver = RayleighBenard(ra=ra, pr=pr, nx=nx, nz=nz, aspect=aspect, dt=float(attrs["dt"]))
    nu = {k: nusselt(v, solver, ra, pr) for k, v in
          (("Solver", truth), ("FNO", fno), ("Reservoir computing", rc))}

    z = np.linspace(0.0, 1.0, nz)
    base = (1.0 - z)[None, None, :]
    panels = {"Solver": base + truth[:, 0], "FNO": base + fno[:, 0],
              "Reservoir computing": base + rc[:, 0]}
    cost = {"Solver": SOLVER_S_PER_STEP, "FNO": FNO_S_PER_STEP,
            "Reservoir computing": rc_s_per_step}
    t = np.arange(n) * dt_s

    fig = plt.figure(figsize=(1000 / args.dpi, 1020 / args.dpi), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.0, 2.5],
                          width_ratios=[60.0, 1.2], hspace=0.34, wspace=0.03,
                          left=0.085, right=0.90, top=0.925, bottom=0.075)

    colours = {"Solver": "0.25", "FNO": "#b2182b", "Reservoir computing": "#2166ac"}
    ims, clocks = {}, {}
    for i, name in enumerate(panels):
        ax = fig.add_subplot(gs[i, 0])
        ims[name] = ax.imshow(panels[name][0].T, origin="lower", extent=[0, aspect, 0, 1],
                              cmap="RdBu_r", vmin=0.0, vmax=1.0, interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(name, fontsize=11, color=colours[name], labelpad=8)
        clocks[name] = ax.text(0.986, 0.90, "", transform=ax.transAxes, fontsize=9.5,
                               ha="right", va="top", color="0.15",
                               bbox={"boxstyle": "round,pad=0.28", "fc": "white",
                                     "ec": "none", "alpha": 0.8})

    cax = fig.add_subplot(gs[0:3, 1])
    cb = fig.colorbar(ims["Solver"], cax=cax)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["cold", "hot"])
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(labelsize=9)

    axn = fig.add_subplot(gs[3, 0])
    traces, dots = {}, {}
    for name in panels:
        style = "-" if name == "Solver" else "--"
        (traces[name],) = axn.plot([], [], style, lw=1.6, color=colours[name], label=name)
        (dots[name],) = axn.plot([], [], "o", ms=5, color=colours[name])
    axn.set_xlim(t[0], t[-1]); axn.set_ylim(8, 20)
    axn.set_xlabel("time (free-fall units)", fontsize=11)
    axn.set_ylabel("Nusselt number", fontsize=11)
    axn.tick_params(labelsize=9)
    axn.spines[["top", "right"]].set_visible(False)
    axn.legend(loc="lower left", fontsize=9, frameon=False, ncol=3)

    fig.suptitle("Rayleigh-B\u00e9nard convection,  $Ra = 10^{7}$", fontsize=14, y=0.975)
    stamp = fig.text(0.085, 0.947, "", fontsize=10, color="0.35")

    def update(i):
        for name in panels:
            ims[name].set_data(panels[name][i].T)
            traces[name].set_data(t[: i + 1], nu[name][: i + 1])
            dots[name].set_data([t[i]], [nu[name][i]])
            clocks[name].set_text(f"{(i + 1) * cost[name]:,.1f} s of compute")
        stamp.set_text(f"t = {t[i]:5.1f}    step {i + 1} / {n}")
        return (list(ims.values()) + list(traces.values())
                + list(dots.values()) + list(clocks.values()) + [stamp])

    out = result_path(args.out)
    writer = FFMpegWriter(fps=args.fps, bitrate=3600,
                          extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
                          metadata={"title": "Solver vs FNO vs reservoir computing"})
    FuncAnimation(fig, update, frames=n, blit=False).save(out, writer=writer, dpi=args.dpi)
    plt.close(fig)
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB, {n} frames, {n / args.fps:.0f} s)")


if __name__ == "__main__":
    main()
