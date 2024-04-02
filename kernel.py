from input import NU_XSF, XS_A, XS_T
from input import N_INACTIVE, N_PARTICLE
from input import CONVERGENCE_METRIC, SE_NX, SE_NY, SE_NZ, BIN_X, BIN_Y, BIN_Z
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


# =============================================================================
# SIMULATION
# =============================================================================

@njit
def init_generation(FISS_BANK: ndarray, ESTIMATOR: ndarray):

    # Reset fission bank and index
    IDX = int(ESTIMATOR['IDX_FISS_BANK'])
    FISS_BANK[:IDX] = np.zeros(IDX, dtype=PARTICLE_TYPE)
    ESTIMATOR['IDX_FISS_BANK'] = 0.0

    # Reset keff generation-wise running-sum
    ESTIMATOR['KEFF_TL_SUM'] = 0.0


@njit
def generation_closeout(idx_gen: int, SRC_BANK: ndarray, FISS_BANK: ndarray, ESTIMATOR: ndarray, METRIC_SE: ndarray, METRIC_COM: ndarray) -> list[float]:

    # Update current generation keff
    ESTIMATOR['KEFF_CURRENT'] = ESTIMATOR['KEFF_TL_SUM'] / N_PARTICLE

    # Calculate convergence metrics
    if CONVERGENCE_METRIC:
        IDX_FISS_BANK = int(ESTIMATOR['IDX_FISS_BANK'])
        SE, COM_X, COM_Y, COM_Z = calculate_convergence_metrics(
            FISS_BANK[:IDX_FISS_BANK])
        METRIC_SE[idx_gen] = SE
        METRIC_COM[idx_gen, 0] = COM_X
        METRIC_COM[idx_gen, 1] = COM_Y
        METRIC_COM[idx_gen, 2] = COM_Z

    # Perform UFS

    # Synchronize particle banks
    sync_bank(SRC_BANK, FISS_BANK, ESTIMATOR)

    # Inactive generation: Report current generation keff
    CURRENT_GEN = idx_gen + 1
    IS_INACTIVE = CURRENT_GEN <= N_INACTIVE
    if IS_INACTIVE:
        return ESTIMATOR['KEFF_CURRENT'], 0.0

    # Active cycle: Accumulate keff and print std
    ESTIMATOR['KEFF_SUM'] += ESTIMATOR['KEFF_CURRENT']
    ESTIMATOR['KEFF_SUMSQ'] += ESTIMATOR['KEFF_CURRENT']**2

    # Calculate current average and absolute std
    N_ACTIVE = CURRENT_GEN - N_INACTIVE
    KEFF_AVG = ESTIMATOR['KEFF_SUM'] / N_ACTIVE
    KEFF_ABS_STD = 1.0 / \
        math.sqrt(N_ACTIVE) * \
        math.sqrt(ESTIMATOR['KEFF_SUMSQ']/N_ACTIVE - KEFF_AVG**2)

    # Report average keff
    return KEFF_AVG, KEFF_ABS_STD


def report(KEFF, STD, SE, COM) -> None:
    print(
        f"{KEFF:.5f} +/- {STD:.5f} {SE:.2f} ({COM[0]:.2f}, {COM[1]:.2f}, {COM[2]:.2f})")


@njit
def sync_bank(SRC_BANK: ndarray, FISS_BANK: ndarray, ESTIMATOR: ndarray):

    # Convert into integer
    IDX_FISS_BANK = int(ESTIMATOR['IDX_FISS_BANK'])

    # Determine how many samples needed
    if IDX_FISS_BANK < N_PARTICLE:
        SITES_NEEDED = N_PARTICLE % IDX_FISS_BANK
    else:
        SITES_NEEDED = N_PARTICLE
    P_SAMPLE = SITES_NEEDED / IDX_FISS_BANK

    # Population control
    TEMP_BANK = np.zeros(3*N_PARTICLE, dtype=PARTICLE_TYPE)
    idx_temp = 0
    for P in FISS_BANK[:IDX_FISS_BANK]:

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
    RATIO = N_PARTICLE / TEMP_BANK[:N_PARTICLE]['wgt'].sum()
    for idx in range(N_PARTICLE):
        SRC_BANK[idx] = TEMP_BANK[idx]
        SRC_BANK[idx]['wgt'] /= RATIO


@njit
def calculate_convergence_metrics(FISS_BANK: ndarray) -> list[float]:

    # Calculate idx corresponding to SE mesh
    IDX_X = np.searchsorted(BIN_X, FISS_BANK['x'])
    IDX_Y = np.searchsorted(BIN_Y, FISS_BANK['y'])
    IDX_Z = np.searchsorted(BIN_Z, FISS_BANK['z'])
    IDX = IDX_X-1 + SE_NX*(IDX_Y-1) + (SE_NX*SE_NY) * (IDX_Z-1)

    # Init SE mesh bin
    METRIC_SE_MESH = np.zeros(SE_NX*SE_NY*SE_NZ)

    # Calculate SE
    TOT_WGT = FISS_BANK['wgt'].sum()
    for loc, P in zip(IDX, FISS_BANK[IDX]):
        METRIC_SE_MESH[int(loc)] += P['wgt'] / TOT_WGT
    NON_ZERO_ENTRY = METRIC_SE_MESH[METRIC_SE_MESH != 0.0]
    SE = -(NON_ZERO_ENTRY * np.log2(NON_ZERO_ENTRY)).sum()

    # Calculate CoM
    COM_X = (FISS_BANK['wgt'] * FISS_BANK['x']).mean()
    COM_Y = (FISS_BANK['wgt'] * FISS_BANK['y']).mean()
    COM_Z = (FISS_BANK['wgt'] * FISS_BANK['z']).mean()

    # Return values
    return SE, COM_X, COM_Y, COM_Z


# =============================================================================
# REACTION
# =============================================================================

@njit
def sample_fission(P, FISS_BANK, ESTIMATOR):

    # Calculate expected num of fiss neutrons
    NU_EXP = P['wgt'] / ESTIMATOR['KEFF_CURRENT'] * NU_XSF / XS_T

    # Determine how many fission neutrons actually sampled
    N_SAMPLE = int(NU_EXP)
    if lcg(P) < (NU_EXP - N_SAMPLE):
        N_SAMPLE += 1

    # Early returns
    if N_SAMPLE == 0:
        return

    # Sample fission neutrons
    for _ in range(N_SAMPLE):
        idx = int(ESTIMATOR['IDX_FISS_BANK'])
        FISS_BANK[idx]['x'] = P['x']
        FISS_BANK[idx]['y'] = P['y']
        FISS_BANK[idx]['z'] = P['z']
        FISS_BANK[idx]['wgt'] = 1.0
        ESTIMATOR['IDX_FISS_BANK'] += 1


@njit
def sample_absorption(P) -> None:
    if lcg(P)*XS_T < XS_A:
        P['wgt'] = 0.0
    return


# =============================================================================
# PARTICLE PHYSICS
# =============================================================================


@njit
def sample_isotropic(P: ndarray) -> None:
    AZIM = PI_DOUBLE * lcg(P)
    MU = 2.0 * lcg(P) - 1.0
    SQRT_TERM = (1.0 - MU**2) ** 0.5

    P['u'] = MU
    P['v'] = math.cos(AZIM) * SQRT_TERM
    P['w'] = math.sin(AZIM) * SQRT_TERM


@njit
def init_particle(IDX_SRC: int, IDX_GEN: int, SRC_BANK: ndarray) -> ndarray:
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
    if IDX_GEN == 0:
        P['wgt'] = 1.0
    return P


@njit
def sample_dts(P: ndarray, PLANES: list[float]) -> float:

    # Get constants
    dts = 1e10
    IDX_PLANE = [0, 1, 2]*2
    POSITIONS = [P['x'], P['y'], P['z']]*2
    DIRECTIONS = [P['u'], P['v'], P['w']]*2

    for idx, plane, pos, dir in zip(IDX_PLANE, PLANES, POSITIONS, DIRECTIONS):

        # calc plane dts
        curr_dts = (plane - pos) / dir

        # get lowest non-negative distance
        if curr_dts > 0 and curr_dts < dts:
            surf_id = idx
            dts = curr_dts
    return dts, surf_id


@njit
def move(P: ndarray, PLANES: list[float]) -> bool:
    DTC = -math.log(lcg(P)) / XS_T
    DTS, IDX_PLANE = sample_dts(P, PLANES)
    IS_REFLECTED = DTS < DTC

    # Reflect direction
    if IS_REFLECTED:
        DTC = DTS

    # Move particle
    P['x'] += P['u']*DTC
    P['y'] += P['v']*DTC
    P['z'] += P['w']*DTC

    # Update outgoing direction and prevent surface coincidence
    if IS_REFLECTED:
        if IDX_PLANE == 0:
            P['u'] *= -1
        elif IDX_PLANE == 1:
            P['v'] *= -1
        else:
            P['w'] *= -1

        P['x'] += P['u']*1e-13
        P['y'] += P['v']*1e-13
        P['z'] += P['w']*1e-13

    # Accumulate track-length tallies
    P['keff'] += P['wgt'] * DTC * NU_XSF
    P['ncoll'] += 1
    return IS_REFLECTED


@njit
def collision(P: ndarray, PLANES: list[float], ESTIMATOR: ndarray, FISS_BANK: ndarray):

    # Move particle
    IS_REFLECTED = move(P, PLANES)
    if IS_REFLECTED:
        return

    # Sample fission neutrons implicitly
    sample_fission(P, FISS_BANK, ESTIMATOR)

    # Sample absorption
    sample_absorption(P)
    if P['wgt'] == 0.0:
        return

    # Sample scattering
    sample_isotropic(P)
