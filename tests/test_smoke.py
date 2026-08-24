"""Sanity checks on the solver, its physics, and the dataset round-trip.

Tests touching generated trajectories skip when data/ is empty (see README).
"""

import numpy as np
import pytest

from rbcs.data import CHANNELS, dataset_name, find_datasets, split_by_trajectory
from rbcs.paths import DATA_DIR
from rbcs.solver import RayleighBenard

TRAJECTORIES = DATA_DIR / dataset_name()

requires_data = pytest.mark.skipif(
    not TRAJECTORIES.is_file(),
    reason=f"{TRAJECTORIES.name} not generated - run rbcs-simulate",
)


@pytest.fixture
def solver():
    """Small, cheap solver instance. Too coarse for physics, fine for shapes."""
    return RayleighBenard(ra=1e6, pr=1.0, nx=32, nz=17, aspect=2.0, dt=2.5e-3, seed=0)


def test_poisson_recovers_a_known_streamfunction(solver):
    x, z = np.meshgrid(solver.x, solver.z, indexing="ij")
    k = 2 * np.pi / solver.lx
    psi = np.sin(k * x) * np.sin(np.pi * z)
    recovered = solver.streamfunction((k**2 + np.pi**2) * psi)
    assert np.abs(recovered - psi).max() / np.abs(psi).max() < 2e-3


def test_poisson_is_second_order_in_z():
    k = 2 * np.pi / 2.0

    def error(nz):
        s = RayleighBenard(ra=1e6, pr=1.0, nx=32, nz=nz, aspect=2.0, dt=2.5e-3)
        x, z = np.meshgrid(s.x, s.z, indexing="ij")
        psi = np.sin(k * x) * np.sin(np.pi * z)
        return np.abs(s.streamfunction((k**2 + np.pi**2) * psi) - psi).max()

    # Halving dz must cut the error by about four.
    assert 3.5 < error(33) / error(65) < 4.5


def test_velocity_field_is_divergence_free(solver):
    u, w = solver.velocities(np.ones_like(solver.omega))
    div = solver._ddx(u) + solver._ddz(w)
    assert np.abs(div[:, 1:-1]).max() < 1e-10


def test_walls_hold_after_stepping(solver):
    for _ in range(5):
        solver.step()
    assert np.abs(solver.theta[:, 0]).max() == 0.0
    assert np.abs(solver.theta[:, -1]).max() == 0.0
    _, w = solver.velocities()
    assert np.abs(w[:, 0]).max() < 1e-12
    assert np.abs(w[:, -1]).max() < 1e-12


def test_conduction_state_has_unit_nusselt():
    s = RayleighBenard(ra=1e6, pr=1.0, nx=32, nz=17, aspect=2.0, dt=2.5e-3)
    s.theta[:] = 0.0
    s.omega[:] = 0.0
    assert s.nusselt() == pytest.approx(1.0)


def test_temperature_includes_the_base_state(solver):
    solver.theta[:] = 0.0
    t = solver.temperature
    assert t[:, 0] == pytest.approx(1.0)
    assert t[:, -1] == pytest.approx(0.0)


def test_stepping_is_deterministic_for_a_fixed_seed():
    states = []
    for _ in range(2):
        s = RayleighBenard(ra=1e6, pr=1.0, nx=32, nz=17, aspect=2.0, dt=2.5e-3, seed=3)
        for _ in range(10):
            s.step()
        states.append(s.theta.copy())
    np.testing.assert_allclose(states[0], states[1])


def test_free_slip_leaves_zero_wall_vorticity():
    s = RayleighBenard(ra=1e6, pr=1.0, nx=32, nz=17, aspect=2.0, dt=2.5e-3, no_slip=False)
    s.step()
    assert np.abs(s.omega[:, 0]).max() == 0.0
    assert np.abs(s.omega[:, -1]).max() == 0.0


def test_split_is_by_trajectory_and_disjoint():
    train, val, test = split_by_trajectory(8)
    assert len(train) == 6 and len(val) == 1 and len(test) == 1
    assert not (set(train) & set(val) & set(test))


def test_split_rejects_too_few_trajectories():
    with pytest.raises(ValueError):
        split_by_trajectory(2)


@requires_data
def test_dataset_shape_and_metadata():
    from rbcs.data import load_dataset

    fields, nusselt, attrs = load_dataset(find_datasets())
    assert fields.ndim == 5
    assert fields.shape[2] == len(CHANNELS)
    assert fields.shape[:2] == nusselt.shape
    assert np.isfinite(fields).all()
    assert attrs["nx"] == fields.shape[3]


@requires_data
def test_generated_trajectories_are_convecting():
    from rbcs.data import load_dataset

    _, nusselt, attrs = load_dataset(find_datasets())
    # Nu ~ 0.16 Ra^0.28 for 2D no-slip: 7.7 at Ra=1e6, 14.6 at Ra=1e7.
    expected = 0.16 * float(attrs["ra"]) ** 0.28
    assert 0.5 * expected < nusselt.mean() < 2.0 * expected


@requires_data
def test_generated_trajectories_respect_the_walls():
    from rbcs.data import load_dataset

    fields, _, _ = load_dataset(find_datasets())
    theta = fields[:, :, 0]
    assert np.abs(theta[..., 0]).max() == 0.0
    assert np.abs(theta[..., -1]).max() == 0.0


def _stack(s):
    return np.stack([s.theta, s.omega])


def _set(s, f):
    s.theta, s.omega = f[0].copy(), f[1].copy()
    return s


def _seeded(seed=1):
    return RayleighBenard(ra=1e6, pr=1.0, nx=32, nz=17, aspect=2.0, dt=2.5e-3, seed=seed)


def test_x_translation_is_a_symmetry_of_the_solver():
    """Shifting in x then stepping equals stepping then shifting.

    This is what makes circular-shift augmentation exact rather than a trick.
    """
    a = _seeded()
    for _ in range(3):
        a.step()
    s = _set(_seeded(), np.roll(_stack(_seeded()), 7, axis=-2))
    for _ in range(3):
        s.step()
    np.testing.assert_allclose(_stack(s), np.roll(_stack(a), 7, axis=-2), atol=1e-10)


def test_x_reflection_is_a_symmetry_of_the_solver():
    """Mirroring x and negating vorticity commutes with time stepping."""
    from rbcs.dataset import reflect_x

    a = _seeded()
    for _ in range(3):
        a.step()
    s = _set(_seeded(), reflect_x(_stack(_seeded())))
    for _ in range(3):
        s.step()
    np.testing.assert_allclose(_stack(s), reflect_x(_stack(a)), atol=1e-10)
