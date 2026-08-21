"""Physical and numerical parameters for the convection runs."""

RA = 1.0e6
PR = 1.0

# Domain: ASPECT wide by 1 tall, periodic in x, no-slip walls in z.
ASPECT = 2.0
NX = 128
NZ = 96

# Advection-limited, not diffusion-limited.
DT = 2.5e-3

# Discard the initial transient before recording.
SPINUP_TIME = 60.0

# Sampling much finer than this makes each step a near-identity map.
SAMPLE_INTERVAL = 0.5
N_SNAPSHOTS = 600

SEED = 0
