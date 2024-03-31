import numpy as np


# =============================================================================
# DATATYPE
# =============================================================================


SEED_STATE = np.dtype([('seed', np.uint64)])

PARTICLE_TYPE = np.dtype([
    ('seed',  np.uint64),
    ('x',  np.float64),
    ('y',  np.float64),
    ('z',  np.float64),
    ('u', np.float64),
    ('v', np.float64),
    ('w', np.float64),
    ('wgt', np.float64),
    ('ncoll', np.uint64),
    ('keff', np.float64),
])
