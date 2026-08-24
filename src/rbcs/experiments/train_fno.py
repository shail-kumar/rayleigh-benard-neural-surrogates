"""Train the FNO for one-step prediction."""

import argparse
import csv
import time

import torch
from torch.utils.data import DataLoader

from rbcs.dataset import build
from rbcs.fno import FNO2d
from rbcs.paths import result_path


def relative_mse(pred, target):
    """Per-channel MSE scaled by channel variance, then averaged.

    Plain MSE would let vorticity (std ~4) swamp temperature (std ~0.26).
    """
    num = ((pred - target) ** 2).mean(dim=(0, 2, 3))
    den = target.var(dim=(0, 2, 3), unbiased=False) + 1e-12
    return (num / den).mean()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total = n = 0.0
    for x, y in loader:
        total += relative_mse(model(x), y).item() * x.shape[0]
        n += x.shape[0]
    model.train()
    return total / n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--modes-x", type=int, default=16)
    p.add_argument("--modes-z", type=int, default=12)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="fno")
    p.add_argument("--threads", type=int, default=None, help="torch intra-op threads")
    p.add_argument("--workers", type=int, default=4, help="dataloader processes")
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    train_ds, val_ds, _, (mean, std), _ = build(seed=args.seed)
    train = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.workers, persistent_workers=args.workers > 0,
    )
    val = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers)

    model = FNO2d(
        width=args.width, modes_x=args.modes_x, modes_z=args.modes_z, depth=args.depth
    )
    model.set_normalisation(mean, std)
    n_par = sum(q.numel() for q in model.parameters())

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    ckpt = result_path(f"{args.tag}.pt")
    metrics = result_path(f"{args.tag}_metrics.csv")
    with open(metrics, "w", newline="") as fh:
        csv.writer(fh).writerow(["epoch", "train", "val", "lr", "minutes"])

    print(
        f"{n_par:,} params | {len(train_ds)} train pairs | {len(val_ds)} val | "
        f"{torch.get_num_threads()} threads | {args.workers} workers",
        flush=True,
    )
    best, best_epoch, t0 = float("inf"), -1, time.perf_counter()

    for epoch in range(args.epochs):
        run = n = 0.0
        for x, y in train:
            opt.zero_grad()
            loss = relative_mse(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}; stopping")
            loss.backward()
            opt.step()
            run += loss.item() * x.shape[0]
            n += x.shape[0]
        sched.step()

        tr_loss, va_loss = run / n, evaluate(model, val)
        mins = (time.perf_counter() - t0) / 60
        with open(metrics, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [epoch, f"{tr_loss:.6f}", f"{va_loss:.6f}", f"{sched.get_last_lr()[0]:.2e}", f"{mins:.1f}"]
            )
        flag = ""
        if va_loss < best:
            best, best_epoch, flag = va_loss, epoch, "  *best"
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "val": va_loss, "epoch": epoch},
                ckpt,
            )
        print(f"epoch {epoch:>3}  train {tr_loss:.5f}  val {va_loss:.5f}  {mins:>5.1f}m{flag}", flush=True)

        if epoch - best_epoch >= args.patience:
            print(f"no improvement in {args.patience} epochs; stopping", flush=True)
            break

    print(f"\nbest val {best:.5f} at epoch {best_epoch}")
    print(f"persistence bar is 0.216; ratio {best / 0.216:.3f}")
    print(f"{ckpt}\n{metrics}")


if __name__ == "__main__":
    main()
