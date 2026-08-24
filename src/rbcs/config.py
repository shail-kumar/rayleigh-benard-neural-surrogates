"""Physical and numerical parameters for the convection runs."""

RA = 1.0e7
PR = 1.0

# Domain: ASPECT wide by 1 tall, periodic in x, no-slip walls in z.
ASPECT = 4.0
NX = 384
NZ = 128

# Advection-limited, not diffusion-limited. CFL stays under 0.22.
DT = 2.0e-3

# Discard the initial transient before recording. Kinetic energy is still
# drifting at 60.
SPINUP_TIME = 80.0

# Sampling much finer than this makes each step a near-identity map.
SAMPLE_INTERVAL = 0.5
N_SNAPSHOTS = 600

SEED = 0
