from input import SEED, N_PARTICLE, N_GENERATION
from input import CONVERGENCE_METRIC, SE_NX, SE_NY, SE_NZ
from datatype import ESTIMATOR_TYPE
import kernel
from prng import set_seed

import numpy as np


# =============================================================================
# RUN SIMULATION
# =============================================================================


def main():

    # Init simulation seed
    set_seed(SEED)

    # Init eigen estimator
    ESTIMATOR = np.zeros(1, dtype=ESTIMATOR_TYPE)[0]
    ESTIMATOR['KEFF_CURRENT'] = 1.0

    # Init convergence metrics
    if CONVERGENCE_METRIC:
        METRIC_SE = np.zeros(N_GENERATION, dtype=np.float64)
        METRIC_SE_MESH = np.zeros([SE_NX*SE_NY*SE_NZ], dtype=np.float64)
        METRIC_COM = np.zeros((N_GENERATION, 3), dtype=np.float64)

    # Loop until the last fission generations
    for idx_gen in range(N_GENERATION):

        # Loop until the last particle of current generation
        print(f"Gen {idx_gen+1} : ", end="")
        for idx_src in range(N_PARTICLE):

            # Init particle
            P = kernel.init_particle(idx_src, idx_gen)

            # Begin random walks until terminated
            while True:
                kernel.collision(P, ESTIMATOR)
                if P['wgt'] == 0.0:
                    break

            # Accumulate track-length estimator
            ESTIMATOR['KEFF_TL_SUM'] += P['keff']

        # Current generation completed: Calculate average keff
        KEFF, STD = kernel.generation_closeout(
            idx_gen, ESTIMATOR, METRIC_SE, METRIC_SE_MESH, METRIC_COM)

        # Report keff
        kernel.report(KEFF, STD, METRIC_SE[idx_gen], METRIC_COM[idx_gen, :])


if __name__ == "__main__":
    main()
