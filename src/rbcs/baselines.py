"""Reference models: persistence, POD reduction, and an ESN in the reduced space."""

import numpy as np


def persistence(x):
    """Next frame equals this one."""
    return x


def ridge_fit(X, Y, alpha):
    """Closed-form multi-output ridge. Rows are samples."""
    n = X.shape[1]
    return np.linalg.solve(X.T @ X + alpha * np.eye(n), X.T @ Y)


def randomized_svd(X, rank, oversample=10, power=1, seed=0):
    """Top-`rank` right singular vectors of X (samples x features).

    Randomized range finder: cheaper than a full SVD when rank << features,
    which is the case here (64 modes out of ~10^5 grid points).
    """
    rng = np.random.default_rng(seed)
    k = rank + oversample
    omega = rng.standard_normal((X.shape[1], k)).astype(X.dtype)
    Y = X @ omega
    for _ in range(power):  # power iterations sharpen the spectrum decay
        Y = X @ (X.T @ Y)
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ X
    _, s, Vt = np.linalg.svd(B, full_matrices=False)
    return s[:rank], Vt[:rank]


class POD:
    """Proper orthogonal decomposition of normalised fields."""

    def __init__(self, rank=64, shifts=8, seed=0):
        self.rank, self.shifts, self.seed = rank, shifts, seed

    def fit(self, fields, mean, std):
        """fields: (n_traj, n_snap, C, nx, nz).

        The basis is fitted on circularly shifted copies. Without them POD
        learns phase-specific structures and fails to transfer: measured
        reconstruction error on a held-out trajectory was 0.148 plain against
        0.023 shift-augmented, at rank 128.
        """
        self.shape = fields.shape[2:]
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, -1, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, -1, 1, 1)
        x = fields.reshape(-1, *self.shape)
        if self.shifts > 1:
            rng = np.random.default_rng(self.seed)
            x = np.concatenate(
                [x] + [
                    np.roll(x, int(rng.integers(x.shape[-2])), axis=-2)
                    for _ in range(self.shifts - 1)
                ]
            )
        flat = self._flatten(x)
        self.singular_values, self.modes = randomized_svd(flat, self.rank, seed=self.seed)
        self.energy = np.cumsum(self.singular_values**2) / np.sum(self.singular_values**2)
        return self

    def _flatten(self, x):
        return ((x - self.mean) / self.std).reshape(x.shape[0], -1)

    def encode(self, x):
        return self._flatten(x) @ self.modes.T

    def decode(self, c):
        x = (c @ self.modes).reshape(-1, *self.shape)
        return x * self.std + self.mean

    def reconstruction_error(self, fields):
        """Relative error from projection alone. Bounds what RC can achieve."""
        x = fields.reshape(-1, *self.shape)
        r = self.decode(self.encode(x))
        num = ((r - x) ** 2).mean(axis=(0, 2, 3))
        den = x.var(axis=(0, 2, 3)) + 1e-12
        return float((num / den).mean())


class ReservoirSurrogate:
    """ESN predicting the next POD coefficient vector from the current one.

    Same recipe as the classical ESN: a fixed random reservoir, a ridge
    readout, and nothing else trained.
    """

    def __init__(
        self,
        n_inputs,
        n_reservoir=800,
        spectral_radius=0.95,
        sparsity=0.9,
        leak_rate=0.3,
        ridge_alpha=1.0,
        seed=0,
    ):
        rng = np.random.default_rng(seed)
        self.n_reservoir, self.leak_rate = n_reservoir, leak_rate
        self.W_in = rng.uniform(-1, 1, (n_reservoir, n_inputs)).astype(np.float32)
        W = rng.uniform(-1, 1, (n_reservoir, n_reservoir)).astype(np.float32)
        W[rng.random((n_reservoir, n_reservoir)) < sparsity] = 0.0
        W *= spectral_radius / np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W.astype(np.float32)
        self.ridge_alpha = ridge_alpha
        self.W_out = None
        # POD coefficients span orders of magnitude across modes; fed raw they
        # saturate tanh on the leading modes and the rest are ignored.
        self.c_mean = None
        self.c_scale = None

    def states(self, U, x0=None):
        """Reservoir trajectory driven by U: (T, n_inputs) -> (T, n_reservoir)."""
        x = np.zeros(self.n_reservoir, dtype=np.float32) if x0 is None else x0.copy()
        out = np.empty((U.shape[0], self.n_reservoir), dtype=np.float32)
        for t in range(U.shape[0]):
            x = (1 - self.leak_rate) * x + self.leak_rate * np.tanh(
                self.W_in @ U[t] + self.W @ x
            )
            out[t] = x
        return out

    def _features(self, states, U):
        # Readout sees the reservoir, the current input, and a bias.
        return np.hstack([states, U, np.ones((len(U), 1), dtype=np.float32)])

    def _standardise(self, U):
        return (U - self.c_mean) / self.c_scale

    def _destandardise(self, U):
        return U * self.c_scale + self.c_mean

    def fit(self, sequences, washout=20):
        """sequences: list of (T, n_inputs) coefficient trajectories."""
        allc = np.concatenate(sequences)
        self.c_mean = allc.mean(axis=0).astype(np.float32)
        self.c_scale = (allc.std(axis=0) + 1e-8).astype(np.float32)
        sequences = [self._standardise(U) for U in sequences]
        F, Y = [], []
        for U in sequences:
            s = self.states(U)
            F.append(self._features(s, U)[washout:-1])
            Y.append(U[washout + 1 :])
        self.W_out = ridge_fit(np.vstack(F), np.vstack(Y), self.ridge_alpha)
        return self

    def predict_next(self, U, washout=20):
        """One-step predictions. The first `washout` entries are unusable
        because the reservoir starts from zero, so they are returned as NaN."""
        Us = self._standardise(U)
        s = self.states(Us)
        out = self._destandardise(self._features(s, Us) @ self.W_out)
        out[:washout] = np.nan
        return out

    def rollout(self, U_warm, n_steps):
        """Free-run after warming the reservoir on U_warm."""
        Uw = self._standardise(U_warm)
        s = self.states(Uw)
        x, u = s[-1], Uw[-1]
        preds = []
        for _ in range(n_steps):
            u = self._features(x[None], u[None])[0] @ self.W_out
            x = (1 - self.leak_rate) * x + self.leak_rate * np.tanh(
                self.W_in @ u + self.W @ x
            )
            preds.append(u)
        return self._destandardise(np.array(preds))
