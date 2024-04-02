from datatype import PARTICLE_TYPE

import numpy as np

# =============================================================================
# MGXS - Hunter Belanger, M&C 2023
# =============================================================================
XS_S = 0.27
XS_C = 0.02
NU_F = 2.5
XS_F = XS_C / (NU_F - 1)
XS_A = XS_F + XS_C
XS_T = XS_S + XS_A
NU_XSF = NU_F * XS_F
K_EFF = NU_XSF / XS_A

# =============================================================================
# INPUT PARAMETERS - Hunter Belanger, M&C 2023
# =============================================================================

# Geom
PX_RHS, PY_RHS, PZ_RHS = 200.0, 200.0, 200.0
PX_LHS, PY_LHS, PZ_LHS = -PX_RHS, -PY_RHS, -PZ_RHS
PLANES = [PX_RHS, PY_RHS, PZ_RHS, PX_LHS, PY_LHS, PZ_LHS]


# Simulation params
SEED = 1
N_PARTICLE = int(1_000)
N_INACTIVE = 50
N_GENERATION = int(10_000)

# =============================================================================
# CONVERGENCE METRICS : Shannon Entropy (SE) and Center of Mass (CoM) available
# SE configs are from Hunter Belanger, M&C 2023
# =============================================================================

CONVERGENCE_METRIC = True
SE_NX, SE_NY, SE_NZ = 8, 8, 8
BIN_X = np.linspace(PX_LHS, PX_RHS, SE_NX+1, endpoint=True)
BIN_Y = np.linspace(PX_LHS, PX_RHS, SE_NX+1, endpoint=True)
BIN_Z = np.linspace(PX_LHS, PX_RHS, SE_NX+1, endpoint=True)
BIN_WIDTH_X = (PX_RHS - PX_LHS) / SE_NX
BIN_WIDTH_Y = (PY_RHS - PY_LHS) / SE_NY
BIN_WIDTH_Z = (PZ_RHS - PZ_LHS) / SE_NZ


# =============================================================================
# Variance Reduction Technique
# =============================================================================

BRANCHLESS_POP_CTRL = True

UFS = False
UFS_NX, UFS_NY, UFS_NZ = 2, 2, 2
UFS_BIN = np.zeros([UFS_NX*UFS_NY*UFS_NZ])

RUSSIAN_ROULETTE = BRANCHLESS_POP_CTRL or UFS
ROULETTE_WGT_THRESHOLD = 0.25
ROULETTE_WGT_SURVIVE = 1.0
