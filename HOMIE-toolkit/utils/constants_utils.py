"""
Skeleton and joint constants for Xperience-10M.
"""

import numpy as np

# MANO hand joint parent indices (21 joints: wrist + 5 fingers * 4 joints)
MANO_PARENT_INDICES = np.array([
    -1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,
])

# SMPL-H body skeleton parent indices (51 values for joints 1..51)
SMPL_H_BODY_PARENT_INDICES = np.array([
    -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 11, 12, 13, 15, 16, 17, 18,
    19, 21, 22, 19, 24, 25, 19, 27, 28, 19, 30, 31, 19, 33, 34,
    20, 36, 37, 20, 39, 40, 20, 42, 43, 20, 45, 46, 20, 48, 49,
], dtype=np.int32)

__all__ = ["MANO_PARENT_INDICES", "SMPL_H_BODY_PARENT_INDICES"]
