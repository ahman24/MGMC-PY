from input import NU_XSF, XS_A, XS_T
from input import N_INACTIVE, N_PARTICLE
from input import get_surfs
from datatype import PARTICLE_TYPE
from prng import lcg, skip_ahead

import random  # for synching bank
import math
import numba as nb
from numba import njit
import numpy as np
from numpy import ndarray

# =============================================================================
# CONSTANS
# =============================================================================
PI_DOUBLE = nb.float64(2.0*math.pi)
IDX_FISS_BANK = 0

SRC_BANK = np.zeros(N_PARTICLE, dtype=PARTICLE_TYPE)
FISS_BANK = np.zeros(3*N_PARTICLE, dtype=PARTICLE_TYPE)

# =============================================================================
# SIMULATION
# =============================================================================


KEFF_CURRENT = nb.float64(1.0)
KEFF_SUM = nb.float64(0.0)
KEFF_SUMSQ = nb.float64(0.0)


# cant be jitted because accessing global vars
@njit
def generation_closeout(idx_gen: int, KEFF_TL_SUM: float) -> None:

    # Update current generation keff
    KEFF_CURRENT = KEFF_TL_SUM / N_PARTICLE

    # Synchronize particle banks
    sync_bank()

    # Calculate average keff
    CURRENT_GEN = idx_gen + 1
    IS_INACTIVE = CURRENT_GEN <= N_INACTIVE
    if IS_INACTIVE:
        print(round(KEFF_CURRENT, 5))
        return

    # Active cycle: Accumulate keff and print std
    global KEFF_SUM, KEFF_SUMSQ
    KEFF_SUM += KEFF_CURRENT
    KEFF_SUMSQ += KEFF_CURRENT**2

    # Calculate current average and absolute std
    N_ACTIVE = CURRENT_GEN - N_INACTIVE
    KEFF_AVG = KEFF_SUM / N_ACTIVE
    KEFF_ABS_STD = 1.0 / \
        math.sqrt(N_ACTIVE) * \
        math.sqrt(KEFF_SUMSQ/N_ACTIVE - KEFF_AVG**2)
    print(str(round(KEFF_AVG, 5)) + " +/- " + str(round(KEFF_ABS_STD, 5)))


# =============================================================================
# REACTION CHANNEL
# =============================================================================

@njit
def sample_fission(P, IDX_FISS_BANK, FISS_BANK) -> int:

    # Calculate expected num of fiss neutrons
    NU_EXP = P['wgt'] / KEFF_CURRENT * NU_XSF / XS_T

    # Determine how many fission neutrons actually sampled
    N_SAMPLE = int(NU_EXP)
    if lcg(P) < (NU_EXP - N_SAMPLE):
        N_SAMPLE += 1

    # Early returns
    if N_SAMPLE == 0:
        return IDX_FISS_BANK

    # Sample fission neutrons
    for _ in range(N_SAMPLE):
        FISS_BANK[IDX_FISS_BANK]['x'] = P['x']
        FISS_BANK[IDX_FISS_BANK]['y'] = P['y']
        FISS_BANK[IDX_FISS_BANK]['z'] = P['z']
        FISS_BANK[IDX_FISS_BANK]['wgt'] = 1.0
        IDX_FISS_BANK += 1
    return IDX_FISS_BANK


@njit
def sample_absorption(P) -> None:
    if lcg(P)*XS_T < XS_A:
        P['wgt'] = 0.0
    return


# =============================================================================
# PARTICLE PHYSICS
# =============================================================================


@njit
def sample_isotropic(P) -> None:
    AZIM = PI_DOUBLE * lcg(P)
    MU = 2.0 * lcg(P) - 1.0
    SQRT_TERM = (1.0 - MU**2) ** 0.5

    P['u'] = MU
    P['v'] = math.cos(AZIM) * SQRT_TERM
    P['w'] = math.sin(AZIM) * SQRT_TERM


@njit
def init_particle(IDX_SRC: int, IDX_GEN: int) -> ndarray:
    """
    Note:
    We cant modify directly by doing

    P = SRC_BANK[IDX_SRC]
    P['u'] = U
    ...

    So instead we just inherit attrs to new particle

    """

    # Init particle and seed
    P = np.zeros(1, dtype=PARTICLE_TYPE)[0]
    IDX_P = (IDX_SRC + (IDX_GEN * N_PARTICLE))
    seed = skip_ahead(IDX_P)
    P['seed'] = seed

    # Inherit other attributes from source
    SRC = SRC_BANK[IDX_SRC]
    P['x'] = SRC['x']
    P['y'] = SRC['y']
    P['z'] = SRC['z']
    sample_isotropic(P)
    P['wgt'] = SRC['wgt']
    if P['wgt'] != 0:
        return
    P['wgt'] = 1

    return P


@njit
def sample_dts(P) -> float:

    # Get constants
    PLANES = get_surfs()
    dts = 1e10

    for idx_plane, plane in enumerate(PLANES):

        # relevant position and dir
        IDX = idx_plane // 3 + 1
        if IDX == 0:
            pos = P['x']
            dir = P['u']
        elif IDX == 1:
            pos = P['y']
            dir = P['v']
        else:
            pos = P['z']
            dir = P['w']

        # calc plane dts
        curr_dts = (plane - pos) / dir

        # get lowest non-negative distance
        if curr_dts > 0 and curr_dts < dts:
            SURF_ID = idx_plane
            dts = curr_dts
    return dts, SURF_ID


@njit
def move(P) -> bool:
    DTC = -math.log(lcg(P)) / XS_T
    DTS, SURF_ID = sample_dts(P)
    IS_REFLECTED = DTS < DTC
    IS_COLLISION = not IS_REFLECTED
    if IS_REFLECTED:
        DTC = DTS
        P['x'] += P['u']*DTC
        P['y'] += P['v']*DTC
        P['z'] += P['w']*DTC
        IDX_POS = SURF_ID % 3
        if IDX_POS == 0:
            P['u'] *= -1
        elif IDX_POS == 1:
            P['v'] *= -1
        else:
            P['w'] *= -1
    elif IS_COLLISION:
        P['x'] += P['u']*DTC
        P['y'] += P['v']*DTC
        P['z'] += P['w']*DTC

    P['keff'] += P['wgt'] * DTC * NU_XSF
    P['ncoll'] += 1
    return IS_REFLECTED


# cant be jitted to update idx fiss bank
def collision(P):

    # Move particle
    IS_REFLECTED = move(P)
    if IS_REFLECTED:
        return

    # Sample fission neutrons implicitly
    global IDX_FISS_BANK, FISS_BANK
    IDX_FISS_BANK = sample_fission(P, IDX_FISS_BANK, FISS_BANK)

    # Sample absorption
    sample_absorption(P)
    if P['wgt'] == 0.0:
        return

    # Sample scattering
    sample_isotropic(P)


# =============================================================================
# PARTICLE BANK OPERATIONS
# =============================================================================

def sync_bank():

    # Determine how many samples needed
    global IDX_FISS_BANK
    IDX_FISS_BANK -= 1
    if IDX_FISS_BANK-1 < N_PARTICLE:
        SITES_NEEDED = N_PARTICLE % IDX_FISS_BANK
    else:
        SITES_NEEDED = N_PARTICLE
    P_SAMPLE = SITES_NEEDED / IDX_FISS_BANK

    # Population control
    TEMP_BANK = np.zeros(5*N_PARTICLE, dtype=PARTICLE_TYPE)
    idx_temp = 0
    for P in FISS_BANK:

        # Duplicate multiple times if less than the bank
        if (IDX_FISS_BANK < N_PARTICLE):
            for idx in range(N_PARTICLE//IDX_FISS_BANK):
                TEMP_BANK[idx_temp] = P
                idx_temp += 1

        # Duplicate randomly
        if random.random() < P_SAMPLE:
            TEMP_BANK[idx_temp] = P
            idx_temp += 1

    # Makes sure the temp bank consists of N_PARTICLE
    if idx_temp < N_PARTICLE:
        SITES_NEEDED = N_PARTICLE-idx_temp
        for idx in range(SITES_NEEDED):
            idx_fb = IDX_FISS_BANK - SITES_NEEDED + idx
            TEMP_BANK[idx_temp] = FISS_BANK[idx_fb]
            idx_temp += 1

    # Now copy the temp bank to new source bank
    for idx in range(N_PARTICLE):
        SRC_BANK[idx] = TEMP_BANK[idx]

    # Normalize the weight
    SRC_BANK['wgt'] /= SRC_BANK['wgt'].sum()

    # Reset index for next generation
    IDX_FISS_BANK = 0
