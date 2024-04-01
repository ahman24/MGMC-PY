from input import N_PARTICLE, N_GENERATION
from datatype import ESTIMATOR_TYPE
import kernel

import numpy as np


# =============================================================================
# RUN SIMULATION
# =============================================================================


def main():

    # Init eigen estimator
    ESTIMATOR = np.zeros(1, dtype=ESTIMATOR_TYPE)[0]
    ESTIMATOR['KEFF_CURRENT'] = 1.0

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
        KEFF, STD = kernel.generation_closeout(idx_gen, ESTIMATOR)

        # Report keff
        if STD != 0.0:
            print(f"{KEFF:.5f} +/- {STD:.5f}")
            continue
        print(f"{KEFF:.5f}")


if __name__ == "__main__":
    main()
