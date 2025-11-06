import numpy as np

from modules.utils import *


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """

    topK_match_results = {
        'indices': [],
        'distances': [],
        'nn_indices': []
    }
    
    mp = np.copy(matrix_profile['mp'])
    mpi = np.copy(matrix_profile['mpi'])
    excl_zone = matrix_profile['excl_zone']

    mp_len = len(mp)

    for _ in range(top_k):
        max_idx = np.argmax(mp)
        max_dist = mp[max_idx]

        if np.isnan(max_dist) or np.isinf(max_dist):
            break

        mp = apply_exclusion_zone(mp, max_idx, excl_zone, np.nan)
        
        topK_match_results['indices'].append(max_idx)
        topK_match_results['distances'].append(max_dist)
        topK_match_results['nn_indices'].append(mpi[max_idx])

    return topK_match_results
