from input import NU_XSF, XS_A, XS_S, XS_T
from input import N_INACTIVE, N_PARTICLE
from input import SE_NX, SE_NY, SE_BIN_X, SE_BIN_Y, SE_BIN_Z
from input import UFS_CONVENTIONAL, UFS_VOL_FRAC, UFS_NX, UFS_NY, UFS_NZ, UFS_BIN_X, UFS_BIN_Y, UFS_BIN_Z
from input import RUSSIAN_ROULETTE, ROULETTE_WGT_THRESHOLD, ROULETTE_WGT_SURVIVE
from input import BRANCHLESS_POP_CTRL
from datatype import PARTICLE_TYPE
from prng import lcg, skip_ahead

import random  # for synching bank
import math
import numba as nb
from numba import njit, prange
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
def init_generation(IDX_GEN: int, SRC_BANK: ndarray, FISS_BANK: ndarray, ESTIMATOR: ndarray, METRIC_SE_MESH: ndarray, UFS_MESH: ndarray):

    # Reset fission bank and index
    IDX = int(ESTIMATOR['IDX_FISS_BANK'])
    FISS_BANK[:IDX] = np.zeros(IDX, dtype=PARTICLE_TYPE)
    ESTIMATOR['IDX_FISS_BANK'] = 0.0

    # Reset keff generation-wise running-sum
    ESTIMATOR['KEFF_TL_SUM'] = 0.0

    # Reset SE mesh src_frac of previous generation
    METRIC_SE_MESH.fill(0.0)

    # Calculate the source fraction in system
    if UFS_CONVENTIONAL and IDX_GEN != 0:
        UFS_MESH.fill(0.0)
        IDX_X = np.searchsorted(UFS_BIN_X, SRC_BANK['x'])
        IDX_Y = np.searchsorted(UFS_BIN_Y, SRC_BANK['y'])
        IDX_Z = np.searchsorted(UFS_BIN_Z, SRC_BANK['z'])
        IDX = IDX_X-1 + UFS_NX*(IDX_Y-1) + (UFS_NX*UFS_NY) * (IDX_Z-1)
        for loc, P in zip(IDX, SRC_BANK):
            UFS_MESH[loc] += P['wgt'] / N_PARTICLE
    elif UFS_CONVENTIONAL and IDX_GEN == 0:
        UNIFORM = 1 / (UFS_NX * UFS_NY * UFS_NZ)
        UFS_MESH.fill(UNIFORM)


@njit
def generation_closeout(idx_gen: int, SRC_BANK: ndarray, FISS_BANK: ndarray, ESTIMATOR: ndarray, METRIC_SE: ndarray, METRIC_SE_MESH: ndarray, METRIC_COM: ndarray) -> list[float]:

    # Update current generation keff
    ESTIMATOR['KEFF_CURRENT'] = ESTIMATOR['KEFF_TL_SUM'] / N_PARTICLE

    # Calculate convergence metrics
    IDX_FISS_BANK = int(ESTIMATOR['IDX_FISS_BANK'])
    calculate_convergence_metrics(idx_gen, FISS_BANK[:IDX_FISS_BANK],
                                  METRIC_SE, METRIC_SE_MESH, METRIC_COM)

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


@njit(parallel=True)
def sync_bank(SRC_BANK: ndarray, FISS_BANK: ndarray, ESTIMATOR: ndarray):
    """
    Perform population control to ensure SRC_BANK size of the next generation 
    is equal to N_PARTICLE.

    VERY IMPORTANT
    --------------
    Note that the sampling process MUST BE WEIGHTED since fission neutrons weights
    are not uniform (especially in branchless simulation). 

    Here we can use shortcut by using available function in numpy, but at the cost
    of not being able to use numba for both generation_closeout and sync_bank functions.
    For now, lets juts do it manually until we figured a solution.
    """

    # Calculate probability of each elem to be selected during weighted sampling
    IDX_FISS_BANK = int(ESTIMATOR['IDX_FISS_BANK'])
    P_SAMPLE = FISS_BANK[:IDX_FISS_BANK]['wgt'] / \
        FISS_BANK[:IDX_FISS_BANK]['wgt'].sum()

    # Weighted sampling: Numpy
    # SRC_BANK = np.random.choice(
    #     FISS_BANK, N_PARTICLE, replace=False, p=P_SAMPLE).astype(PARTICLE_TYPE)

    # Weighted sampling: Manual
    idx_selected = 0
    TOT_WGT = 0
    while idx_selected != N_PARTICLE:
        for idx_p, prob in enumerate(P_SAMPLE):
            if random.random() < prob:
                SRC_BANK[idx_selected]['x'] = FISS_BANK[idx_p]['x']
                SRC_BANK[idx_selected]['y'] = FISS_BANK[idx_p]['y']
                SRC_BANK[idx_selected]['z'] = FISS_BANK[idx_p]['z']
                SRC_BANK[idx_selected]['wgt'] = FISS_BANK[idx_p]['wgt']
                TOT_WGT += FISS_BANK[idx_p]['wgt']
                idx_selected += 1
            if idx_selected == N_PARTICLE:
                break

    # Normalize source bank weights
    WGT_MULT = N_PARTICLE / TOT_WGT
    for idx in prange(N_PARTICLE):
        SRC_BANK[idx]['wgt'] *= WGT_MULT


@njit
def calculate_convergence_metrics(IDX_GEN: int, FISS_BANK: ndarray, METRIC_SE: ndarray, METRIC_SE_MESH: ndarray, METRIC_COM: ndarray) -> list[float]:

    # Calculate idx corresponding to SE mesh
    IDX_X = np.searchsorted(SE_BIN_X, FISS_BANK['x'])
    IDX_Y = np.searchsorted(SE_BIN_Y, FISS_BANK['y'])
    IDX_Z = np.searchsorted(SE_BIN_Z, FISS_BANK['z'])
    IDX = IDX_X-1 + SE_NX*(IDX_Y-1) + (SE_NX*SE_NY) * (IDX_Z-1)

    # Calculate SE
    # Note that the lowest entropy value is 0.0
    TOT_WGT = FISS_BANK['wgt'].sum()
    for loc, P in zip(IDX, FISS_BANK[IDX]):
        METRIC_SE_MESH[int(loc)] += P['wgt'] / TOT_WGT
    NON_ZERO_ENTRY = METRIC_SE_MESH[METRIC_SE_MESH != 0.0]
    METRIC_SE[IDX_GEN] = -(NON_ZERO_ENTRY * np.log2(NON_ZERO_ENTRY)).sum()
    if METRIC_SE[IDX_GEN] < 0.0:
        METRIC_SE[IDX_GEN] = 0.0

    # Calculate CoM
    METRIC_COM[IDX_GEN, 0] = (FISS_BANK['wgt'] * FISS_BANK['x']).mean()
    METRIC_COM[IDX_GEN, 1] = (FISS_BANK['wgt'] * FISS_BANK['y']).mean()
    METRIC_COM[IDX_GEN, 2] = (FISS_BANK['wgt'] * FISS_BANK['z']).mean()


# =============================================================================
# REACTION
# =============================================================================

@njit
def sample_fission(P: ndarray, FISS_BANK: ndarray, ESTIMATOR: ndarray, UFS_MESH: ndarray) -> None:

    # Get UFS weight
    UFS_WGT = 1.0
    if UFS_CONVENTIONAL:
        UFS_WGT = ufs_get_wgt(P, UFS_MESH)

    # Branchless: explicitly sample fission reaction
    BC_OUTGOING_WGT = 1.0
    if BRANCHLESS_POP_CTRL:

        # No fission reaction
        if lcg(P) > bc_calc_prob_fission():
            return

        # Fission reaction
        N_SAMPLE = 1
        BC_OUTGOING_WGT = bc_calc_outgoing_wgt(P)
        P['wgt'] = 0.0
        if BC_OUTGOING_WGT > ROULETTE_WGT_SURVIVE:
            N_SAMPLE += int(P['wgt'] / ROULETTE_WGT_SURVIVE)
            BC_OUTGOING_WGT /= N_SAMPLE

    # Calculate expected num of fiss neutrons
    else:
        NU_EXP = P['wgt'] / ESTIMATOR['KEFF_CURRENT'] * UFS_WGT * NU_XSF / XS_T

        # Determine how many fission neutrons actually sampled
        N_SAMPLE = int(NU_EXP)
        if lcg(P) < (NU_EXP - N_SAMPLE):
            N_SAMPLE += 1

        # Early returns
        if N_SAMPLE == 0:
            return

    # Handle population explosion in UFS or Branchless
    SAMPLE_LIM = 4
    if N_SAMPLE > SAMPLE_LIM:
        RATIO = N_SAMPLE // SAMPLE_LIM + 1
        NEW_N_SAMPLE = int(N_SAMPLE/RATIO)
        UFS_WGT /= (N_SAMPLE / NEW_N_SAMPLE)
        N_SAMPLE = NEW_N_SAMPLE

    # Sample fission neutrons
    for _ in range(N_SAMPLE):
        idx = int(ESTIMATOR['IDX_FISS_BANK'])
        FISS_BANK[idx]['x'] = P['x']
        FISS_BANK[idx]['y'] = P['y']
        FISS_BANK[idx]['z'] = P['z']
        FISS_BANK[idx]['wgt'] = BC_OUTGOING_WGT / UFS_WGT
        ESTIMATOR['IDX_FISS_BANK'] += 1


@njit
def sample_absorption(P) -> None:

    # Branchless Collision: do not sample absorption
    if BRANCHLESS_POP_CTRL:
        return

    # Otherwise, sample
    if lcg(P)*XS_T < XS_A:
        P['wgt'] = 0.0


# =============================================================================
# PARTICLE PHYSICS
# =============================================================================


@njit
def sample_isotropic(P: ndarray) -> None:

    # Sample isotropic direction
    AZIM = PI_DOUBLE * lcg(P)
    MU = 2.0 * lcg(P) - 1.0
    SQRT_TERM = (1.0 - MU**2) ** 0.5

    P['u'] = MU
    P['v'] = math.cos(AZIM) * SQRT_TERM
    P['w'] = math.sin(AZIM) * SQRT_TERM


@njit
def init_particle(IDX_SRC: int, IDX_GEN: int, SRC_BANK: ndarray, SECONDARY_BANK: ndarray) -> ndarray:
    """
    Note:
    We cant modify directly by doing

    P = SRC_BANK[IDX_SRC]
    P['u'] = U
    ...

    So instead we just inherit attrs to new particle

    """

    # Clear particle secondary bank
    SECONDARY_BANK[:] = np.zeros(3*N_PARTICLE, dtype=PARTICLE_TYPE)

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
    return P


@njit
def revive_from_secondary(P: ndarray, SECONDARY_BANK: ndarray) -> ndarray:

    # Get P idx from secondary bank to revive
    IDX = int(P['n_secondary']) - 1

    # Inherit attrs from the particle
    P['x'] = SECONDARY_BANK[IDX]['x']
    P['y'] = SECONDARY_BANK[IDX]['y']
    P['z'] = SECONDARY_BANK[IDX]['z']
    sample_isotropic(P)
    P['wgt'] = SECONDARY_BANK[IDX]['wgt']
    P['ncoll'] = SECONDARY_BANK[IDX]['ncoll']

    # Reduce the secondary bank counter
    P['n_secondary'] -= 1


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
def collision(P: ndarray, PLANES: list[float], ESTIMATOR: ndarray, FISS_BANK: ndarray, SECONDARY_BANK: ndarray, UFS_MESH: ndarray) -> None:

    # Move particle
    IS_REFLECTED = move(P, PLANES)
    if IS_REFLECTED:
        return

    # Sample fission neutrons implicitly
    sample_fission(P, FISS_BANK, ESTIMATOR, UFS_MESH)

    # Sample absorption
    sample_absorption(P)
    if P['wgt'] == 0.0:
        return

    # Sample scattering
    sample_isotropic(P)

    # Branchless Collision: Update outgoing weight
    if BRANCHLESS_POP_CTRL:
        P['wgt'] = bc_calc_outgoing_wgt(P)

    # Execute roulette if needed
    if RUSSIAN_ROULETTE and (P['wgt'] < ROULETTE_WGT_THRESHOLD or P['wgt'] > ROULETTE_WGT_SURVIVE):
        roulette(P, SECONDARY_BANK, ESTIMATOR)


@njit
def roulette(P: ndarray, SECONDARY_BANK: ndarray, ESTIMATOR: ndarray) -> None:

    # Determine to split or to roulette
    IS_ROULETTE = P['wgt'] < ROULETTE_WGT_THRESHOLD

    # Case 1: Execute roulette
    if IS_ROULETTE:
        IS_KILLED = lcg(P) < (1 - P['wgt'] / ROULETTE_WGT_SURVIVE)
        if IS_KILLED:
            P['wgt'] = 0.0
            return

        # Survive
        P['wgt'] = ROULETTE_WGT_SURVIVE
        return

    # Case 2: Split particle
    N_SPLIT = int(P['wgt'] / ROULETTE_WGT_SURVIVE)
    P['wgt'] /= (N_SPLIT+1)

    # Add remainder to secondary bank
    for idx in range(N_SPLIT):
        idx = int(P['n_secondary']) + idx
        SECONDARY_BANK[idx] = P

    # Update number of particles in secondary bank
    P['n_secondary'] += N_SPLIT
    return

# =============================================================================
# VARIANCE REDUCTION TECHNIQUE
# =============================================================================


@njit
def bc_calc_prob_fission() -> float:
    return NU_XSF / (NU_XSF + XS_S)


@njit
def bc_calc_outgoing_wgt(P: ndarray) -> float:
    return P['wgt'] * (NU_XSF + XS_S) / XS_T


@njit
def ufs_get_wgt(P: ndarray, UFS_MESH: ndarray) -> float:

    # Get source frac
    IDX_X = np.searchsorted(UFS_BIN_X, P['x'])
    IDX_Y = np.searchsorted(UFS_BIN_Y, P['y'])
    IDX_Z = np.searchsorted(UFS_BIN_Z, P['z'])
    IDX = IDX_X-1 + UFS_NX*(IDX_Y-1) + (UFS_NX*UFS_NY) * (IDX_Z-1)
    SRC_FRAC = UFS_MESH[IDX]

    # Early returns
    if SRC_FRAC == 0.0:
        return 1.0

    # Handle population explosion
    # When src_frac is too low (non-zero), pop can explode
    # Mitigate this by introducing a threshold.
    VOL_TO_SRC = UFS_VOL_FRAC / SRC_FRAC
    if VOL_TO_SRC <= 0.10:
        VOL_TO_SRC = 1.0
    return VOL_TO_SRC
