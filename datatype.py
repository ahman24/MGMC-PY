import numpy as np


# =============================================================================
# DATATYPE
# =============================================================================


ESTIMATOR_TYPE = np.dtype([
    ('KEFF_TL_SUM', np.float64),
    ('KEFF_CURRENT', np.float64),
    ('KEFF_SUM', np.float64),
    ('KEFF_SUMSQ', np.float64),
    ('IDX_FISS_BANK', np.float64),
])


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
