"""2D Rayleigh-Benard convection, Boussinesq, vorticity-streamfunction form.

Periodic in x, fixed-temperature walls in z, no-slip by default. Fourier in x,
second-order finite differences in z, SSP-RK3 in time. Free-fall units.
"""

import numpy as np
from scipy.fft import dst, idst


class RayleighBenard:
    """State is `theta` (temperature perturbation) and `omega` (vorticity),
    both shaped (nx, nz) with walls at z indices 0 and -1."""

    def __init__(self, ra, pr, nx, nz, aspect, dt, seed=0, no_slip=True):
        self.ra, self.pr, self.dt = ra, pr, dt
        self.no_slip = no_slip
        self.nx, self.nz = nx, nz
        self.lx = aspect

        self.nu = np.sqrt(pr / ra)
        self.kappa = 1.0 / np.sqrt(ra * pr)

        self.dx = self.lx / nx
        self.dz = 1.0 / (nz - 1)
        self.x = np.arange(nx) * self.dx
        self.z = np.arange(nz) * self.dz

        self.kx = 2.0 * np.pi * np.fft.rfftfreq(nx, d=self.dx)
        self.k2 = self.kx**2

        # DST-I diagonalises the Dirichlet d2/dz2 operator exactly.
        m = np.arange(1, nz - 1)
        self.lam_z = -(4.0 / self.dz**2) * np.sin(m * np.pi / (2 * (nz - 1))) ** 2

        # Positive for every k, so k=0 needs no special case.
        self.poisson_denom = self.k2[:, None] - self.lam_z[None, :]

        rng = np.random.default_rng(seed)
        self.theta = 1e-2 * rng.standard_normal((nx, nz))
        self.omega = np.zeros((nx, nz))
        self._apply_bc()
        self.time = 0.0

    def _apply_bc(self):
        self.theta[:, 0] = 0.0
        self.theta[:, -1] = 0.0
        self._set_wall_vorticity(self.omega)

    def _set_wall_vorticity(self, omega):
        """Impose the wall condition on omega, in place.

        Zero under free-slip; Woods' formula under no-slip."""
        if not self.no_slip:
            omega[:, 0] = 0.0
            omega[:, -1] = 0.0
            return omega
        psi = self.streamfunction(omega)
        omega[:, 0] = -0.5 * omega[:, 1] - 3.0 * psi[:, 1] / self.dz**2
        omega[:, -1] = -0.5 * omega[:, -2] - 3.0 * psi[:, -2] / self.dz**2
        return omega

    def _ddx(self, f):
        return np.fft.irfft(1j * self.kx[:, None] * np.fft.rfft(f, axis=0), n=self.nx, axis=0)

    def _d2dx2(self, f):
        return np.fft.irfft(-self.k2[:, None] * np.fft.rfft(f, axis=0), n=self.nx, axis=0)

    def _ddz(self, f):
        out = np.empty_like(f)
        out[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * self.dz)
        # One-sided at the walls, for u = dpsi/dz.
        out[:, 0] = (-3.0 * f[:, 0] + 4.0 * f[:, 1] - f[:, 2]) / (2.0 * self.dz)
        out[:, -1] = (3.0 * f[:, -1] - 4.0 * f[:, -2] + f[:, -3]) / (2.0 * self.dz)
        return out

    def _d2dz2(self, f):
        out = np.zeros_like(f)
        out[:, 1:-1] = (f[:, 2:] - 2.0 * f[:, 1:-1] + f[:, :-2]) / self.dz**2
        return out

    def _lap(self, f):
        return self._d2dx2(f) + self._d2dz2(f)

    def streamfunction(self, omega):
        """Solve lap(psi) = -omega with psi = 0 on both walls."""
        rhs = np.fft.rfft(omega[:, 1:-1], axis=0)
        rhs = dst(rhs.real, type=1, axis=1, norm="ortho") + 1j * dst(
            rhs.imag, type=1, axis=1, norm="ortho"
        )
        psi_hat = rhs / self.poisson_denom
        psi_i = idst(psi_hat.real, type=1, axis=1, norm="ortho") + 1j * idst(
            psi_hat.imag, type=1, axis=1, norm="ortho"
        )
        psi = np.zeros((self.nx, self.nz))
        psi[:, 1:-1] = np.fft.irfft(psi_i, n=self.nx, axis=0)
        return psi

    def velocities(self, omega=None):
        """Return (u, w) from the streamfunction."""
        psi = self.streamfunction(self.omega if omega is None else omega)
        return self._ddz(psi), -self._ddx(psi)

    def _rhs(self, theta, omega):
        omega = self._set_wall_vorticity(omega.copy())
        psi = self.streamfunction(omega)
        u, w = self._ddz(psi), -self._ddx(psi)

        dtheta_dx = self._ddx(theta)

        # +w is the base-state advection term, -w * dT_base/dz.
        dtheta = (
            -u * dtheta_dx - w * self._ddz(theta) + w + self.kappa * self._lap(theta)
        )
        # Buoyancy dT/dx reduces to dtheta/dx.
        domega = (
            -u * self._ddx(omega)
            - w * self._ddz(omega)
            + dtheta_dx
            + self.nu * self._lap(omega)
        )
        return dtheta, domega

    def step(self):
        """Advance one step with SSP-RK3."""
        dt = self.dt
        t0, o0 = self.theta, self.omega

        kt, ko = self._rhs(t0, o0)
        t1, o1 = t0 + dt * kt, o0 + dt * ko
        t1[:, [0, -1]] = 0.0

        kt, ko = self._rhs(t1, o1)
        t2 = 0.75 * t0 + 0.25 * (t1 + dt * kt)
        o2 = 0.75 * o0 + 0.25 * (o1 + dt * ko)
        t2[:, [0, -1]] = 0.0

        kt, ko = self._rhs(t2, o2)
        self.theta = (t0 + 2.0 * (t2 + dt * kt)) / 3.0
        self.omega = (o0 + 2.0 * (o2 + dt * ko)) / 3.0
        self._apply_bc()
        self.time += dt

    def advance(self, duration):
        """Step until `duration` of simulation time has elapsed."""
        for _ in range(round(duration / self.dt)):
            self.step()

    @property
    def temperature(self):
        """Full temperature field, base state included."""
        return (1.0 - self.z)[None, :] + self.theta

    def nusselt(self):
        """Volume-averaged Nusselt number: 1 for conduction, ~8 at Ra=1e6."""
        _, w = self.velocities()
        return 1.0 + np.sqrt(self.ra * self.pr) * float(np.mean(w * self.theta))

    def kinetic_energy(self):
        u, w = self.velocities()
        return 0.5 * float(np.mean(u**2 + w**2))

    def is_finite(self):
        return bool(np.isfinite(self.theta).all() and np.isfinite(self.omega).all())
