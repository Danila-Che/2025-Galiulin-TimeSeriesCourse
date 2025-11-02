import numpy as np

from modules.utils import *


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    topK_match_results = {
        'indices': [],
        'distances': []
    }

    mp = np.copy(matrix_profile['mp'])
    mpi = np.copy(matrix_profile['mpi'])
    excl_zone = matrix_profile['excl_zone']

    mp_len = len(mp)

    for k in range(top_k):
        min_idx = np.argmin(mp)
        min_dist = mp[min_idx]

        if np.isnan(min_dist) or np.isinf(min_dist):
            break

        paired_idx = mpi[min_idx]

        mp = apply_exclusion_zone(mp, min_idx, excl_zone, np.nan)
        mp = apply_exclusion_zone(mp, paired_idx, excl_zone, np.nan)

        left_motif_idx = min(min_idx, paired_idx)
        right_motif_idx = max(min_idx, paired_idx)

        topK_match_results['indices'].append((left_motif_idx, right_motif_idx))
        topK_match_results['distances'].append(min_dist)

    return topK_match_results
