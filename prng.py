import numba as nb
import random
from numba import njit

# =============================================================================
# BASIC LCG
# =============================================================================

RNG_MASTER_SEED = 1
RNG_G = nb.uint64(3512401965023503517)
RNG_C = nb.uint64(0)
RNG_MOD_MASK = nb.uint64(0x7FFFFFFFFFFFFFFF)
RNG_MOD = nb.uint64(0x8000000000000000)
RNG_SEED = nb.uint64(1)
RNG_STRIDE = nb.uint64(152917)


@njit
def lcg(P) -> nb.float64:
    NEW_SEED = RNG_G * P['seed'] + RNG_C & RNG_MOD_MASK
    P['seed'] = NEW_SEED
    return P['seed'] / RNG_MOD


@njit
def skip_ahead(IDX_P: int) -> nb.uint64:
    seed_base = RNG_MASTER_SEED
    g = RNG_G
    c = RNG_C
    g_new = nb.uint64(1)
    c_new = nb.uint64(0)
    mod_mask = RNG_MOD_MASK

    n = IDX_P * RNG_STRIDE
    n = n & mod_mask
    while n > 0:
        if n & 1:
            g_new = g_new * g & mod_mask
            c_new = nb.uint64(c_new * g + c) & mod_mask

        c = nb.uint64((g + 1) * c) & mod_mask
        g = g * g & mod_mask
        n >>= 1

    return (g_new * seed_base + c_new) & mod_mask


def set_seed(SEED: int) -> None:
    RNG_MASTER_SEED = SEED
    random.seed(SEED)
