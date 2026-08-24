# Neural surrogates for Rayleigh-Bénard convection

Neural operators can stand in for a fluid solver. This project asks what is lost
when they do, and whether the usual error metrics would show it.

I compare a Fourier Neural Operator (FNO), written from scratch in PyTorch, against
reservoir computing on a POD basis and a persistence baseline, on 2D
Rayleigh-Bénard convection at Ra = 1e7. The three are compared on field error, on
heat transport, and on the energy spectrum band by band.

## Results

Single-step prediction on held-out trajectories:

| model | test rel. MSE | parameters |
|---|---|---|
| Persistence | 0.0268 | 0 |
| Reservoir computing | 0.0182 | 51.0M |
| FNO | 0.00052 | 617k |

The FNO is 35x more accurate than reservoir computing with 83x fewer parameters,
and 52x more accurate than persistence.

Field error alone is not enough to judge a surrogate. After a 20-step rollout,
the energy retained relative to truth is:

| band | FNO | RC | Persistence |
|---|---|---|---|
| large, k 1-4 | 0.99 | 1.03 | 0.96 |
| small, k 16-48 | 1.03 | 0.35 | 0.79 |
| finest, k 48+ | 0.90 | 0.14 | 0.66 |

Reservoir computing retains 14% of the energy at the finest scales, because
truncating to 512 POD modes cannot represent the thermal filaments. The heat
transport shows the same thing: RC reproduces the mean Nusselt number to within
4.4%, but its fluctuation falls from 2.06 to 0.44. The FNO reproduces both the
mean and the fluctuation.

Timing, both single-threaded on the same machine: the solver takes 5.22 s to
advance 0.5 time units, the FNO 57 ms. That is a 90x speedup. Kochkov et al.
report 40-80x for matched-accuracy comparisons.

The rollout is not stable indefinitely. The FNO tracks the solver for roughly 40
steps, is visibly wrong by 80, and by 120 its error exceeds that of predicting
the mean field. Each step feeds its own error back as input, and nothing in
single-step training penalises that compounding.

## The flow

Boussinesq convection in a layer heated from below, periodic horizontally, with
no-slip isothermal walls. Vorticity-streamfunction formulation, Fourier in x,
second-order finite differences in z, SSP-RK3 in time.

| | |
|---|---|
| Rayleigh number | 1e7 |
| Prandtl number | 1 |
| Grid | 384 x 128, aspect ratio 4 |
| Measured Nusselt number | 14.10 +- 2.43 |
| Published, 2D no-slip | ~14.6, from Nu = 0.16 Ra^0.28 |

The Poisson solve is verified against a known streamfunction and confirmed
second-order in z under grid refinement. The full solver reproduces the domain's
translation and reflection symmetries to 1e-10, gives a divergence-free velocity
field to the same tolerance, and matches the published heat transport.

## Quickstart

```bash
pip install -e ".[dev]"

rbcs-simulate -n 8 -j 8    # 8 trajectories on 8 cores, about 100 minutes
pytest                     # tests needing data skip if there is none
```

Trajectories are independent, so generation runs one worker per seed. Data lands
in `data/` and is not tracked; regenerate it as needed. Override locations with
`RBCS_DATA_DIR` and `RBCS_RESULTS_DIR`.

## Dataset

`data/trajectories_Ra1e7.h5`, shaped `(n_traj, n_snapshots, 2, nx, nz)` float32,
1.89 GB for 8 trajectories. Channel 0 is the temperature perturbation and channel
1 the vorticity. Those two determine the state; streamfunction and velocities
follow from the vorticity.

Splits are by trajectory, never by time step. Neighbouring snapshots are strongly
correlated, so a time-step split leaks the test set into training.

Nothing varies the Rayleigh number, aspect ratio or boundary conditions between
trajectories, so the trained model works at Ra = 1e7 in this geometry and should
not be expected to work elsewhere.

## Limitations

The rollout is usable over roughly 40 steps and diverges numerically past 160.
Injecting noise during training, or training on a multi-step loss, are the usual
remedies and are not attempted here.

The surrogate is differentiable end to end, but its gradients are not reliable
for optimisation. Ascending them produces states the solver disagrees with: in
one test the surrogate predicted a 48% rise in heat transport where the solver
measured a 14% fall.

A fuller writeup of the method and the evaluation follows.
