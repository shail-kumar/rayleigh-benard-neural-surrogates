# Neural surrogates for Rayleigh-Bénard convection

Can a neural operator replace a fluid solver, and does it beat a much cheaper
classical method at the same job?

This compares a **Fourier Neural Operator** against **reservoir computing** and a
persistence reference, predicting the evolution of 2D Rayleigh-Bénard
convection. Accuracy is measured not only as field error but as whether the
surrogate preserves the physics: heat transport and the energy spectrum.


## The flow

Boussinesq convection in a layer heated from below, periodic horizontally, with
no-slip isothermal walls. Vorticity-streamfunction formulation, Fourier in x,
second-order finite differences in z, SSP-RK3 in time.

| | |
|---|---|
| Rayleigh number | 1e6 |
| Prandtl number | 1 |
| Grid | 128 x 96, aspect ratio 2 |
| Measured Nusselt number | 7.69 ± 1.50 |
| Published value, 2D no-slip | 8 to 9 |

The solver is verified against a manufactured solution, produces a
divergence-free velocity field to machine precision, and reproduces the
published heat transport.

## Quickstart

```bash
pip install -e ".[dev]"

rbcs-simulate -n 8 -j 8          # 8 trajectories on 8 cores, about 7 minutes
pytest                            # tests needing data skip if there is none
```

Trajectories are independent, so generation runs one worker per seed. Data
lands in `data/` and is not tracked; regenerate it as needed. Override
locations with `RBCS_DATA_DIR` and `RBCS_RESULTS_DIR`.

## Dataset

`data/trajectories.h5`, shaped `(n_traj, n_snapshots, 2, nx, nz)` float32, with
channel 0 the temperature perturbation and channel 1 the vorticity. Those two
determine the state; streamfunction and velocities follow from the vorticity.

Splits are by trajectory, never by time step: neighbouring snapshots are
strongly correlated, so a time-step split leaks the test set into training.
