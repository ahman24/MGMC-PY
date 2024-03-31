from input import N_PARTICLE, N_GENERATION
import kernel


# =============================================================================
# RUN SIMULATION
# =============================================================================


# @njit
def main():

    # Loop until the last fission generations
    for idx_gen in range(N_GENERATION):

        # Loop until the last particle of current generation
        print(f"Gen {idx_gen+1} : ", end="")
        KEFF_TL_SUM = 0.0
        for idx_src in range(N_PARTICLE):

            # Init particle
            P = kernel.init_particle(idx_src, idx_gen)

            # Begin random walks until terminated
            while True:
                kernel.collision(P)
                if P['wgt'] == 0.0:
                    KEFF_TL_SUM += P['keff']
                    break

        # Current generation completed: Calculate average keff
        kernel.generation_closeout(idx_gen, KEFF_TL_SUM)

        # Calc SE

        # Perform UFS

        # Sync fiss bank to src bank


if __name__ == "__main__":
    main()
