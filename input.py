from datatype import PARTICLE_TYPE

import numpy as np

# =============================================================================
# MGXS
# =============================================================================
K_EFF = 1.0
XS_T = 1.0
XS_S = 0.7
XS_A = XS_T - XS_S
NU_XSF = K_EFF * XS_A

# =============================================================================
# INPUT FILE
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

CONVERGENCE_METRIC = True
SE_NX, SE_NY, SE_NZ = 8, 8, 8
BIN_X = np.linspace(PX_LHS, PX_RHS, SE_NX+1, endpoint=True)
BIN_Y = np.linspace(PX_LHS, PX_RHS, SE_NX+1, endpoint=True)
BIN_Z = np.linspace(PX_LHS, PX_RHS, SE_NX+1, endpoint=True)
BIN_WIDTH_X = (PX_RHS - PX_LHS) / SE_NX
BIN_WIDTH_Y = (PY_RHS - PY_LHS) / SE_NY
BIN_WIDTH_Z = (PZ_RHS - PZ_LHS) / SE_NZ


UFS = True
UFS_NX, UFS_NY, UFS_NZ = 2, 2, 2
UFS_BIN = np.zeros([UFS_NX*UFS_NY*UFS_NZ])
