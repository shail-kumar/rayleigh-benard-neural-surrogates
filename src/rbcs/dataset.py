"""Consecutive-snapshot pairs with the domain's exact symmetries as augmentation."""

import numpy as np
import torch
from torch.utils.data import Dataset

from rbcs.data import find_datasets, load_dataset, split_by_trajectory

# Vorticity is a pseudoscalar: reflecting x flips its sign, and the horizontal
# velocity's, while temperature is unchanged.
REFLECTION_SIGN = (1.0, -1.0)


def reflect_x(field):
    """Mirror in x, negating vorticity. Exact symmetry of the equations."""
    out = np.flip(field, axis=-2).copy()
    for c, s in enumerate(REFLECTION_SIGN):
        out[..., c, :, :] *= s
    return out


class PairDataset(Dataset):
    """(field_t, field_t+1) pairs from a set of trajectories."""

    def __init__(self, fields, augment=False, seed=0):
        self.fields = fields
        self.augment = augment
        self.n_traj, self.n_snap = fields.shape[:2]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_traj * (self.n_snap - 1)

    def __getitem__(self, i):
        traj, t = divmod(i, self.n_snap - 1)
        x = self.fields[traj, t]
        y = self.fields[traj, t + 1]
        if self.augment:
            # x is periodic, so every circular shift is a valid sample. Both
            # frames must shift together or the pair stops being a time step.
            shift = int(self.rng.integers(self.fields.shape[-2]))
            x = np.roll(x, shift, axis=-2)
            y = np.roll(y, shift, axis=-2)
            if self.rng.random() < 0.5:
                x, y = reflect_x(x), reflect_x(y)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(
            np.ascontiguousarray(y)
        )


def channel_stats(fields):
    """Per-channel mean and std, over training trajectories only."""
    return fields.mean(axis=(0, 1, 3, 4)), fields.std(axis=(0, 1, 3, 4))


def build(ra=None, n_val=1, n_test=1, seed=0):
    """Return (train, val, test) datasets plus the training channel statistics."""
    fields, _, attrs = load_dataset(find_datasets(ra))
    train_i, val_i, test_i = split_by_trajectory(fields.shape[0], n_val, n_test)
    mean, std = channel_stats(fields[train_i])
    return (
        PairDataset(fields[train_i], augment=True, seed=seed),
        PairDataset(fields[val_i]),
        PairDataset(fields[test_i]),
        (mean, std),
        attrs,
    )
