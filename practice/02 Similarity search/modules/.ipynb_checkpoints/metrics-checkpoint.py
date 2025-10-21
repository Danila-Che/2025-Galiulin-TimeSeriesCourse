import numpy as np
import math


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """
    
    ed_dist = np.sqrt(np.sum(np.square(ts1 - ts2)))

    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    n = ts1.shape[0]

    assert n == ts2.shape[0]

    mu1 = np.mean(ts1)
    mu2 = np.mean(ts2)

    s1 = np.std(ts1)
    s2 = np.std(ts2)

    if s1 == 0 or s2 == 0:
        return 0.

    d = np.dot(ts1, ts2)
    
    norm_ed_dist = np.sqrt(np.abs(2*n*(1 - (d - n*mu1*mu2) / (n*s1*s2))))

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    n, m = len(ts1), len(ts2)
    w = int(max(r * max(n, m), abs(n - m)))

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        start = max(1, i - w)
        end = min(m, i + w) + 1
        
        for j in range(start, end):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2
            dtw[i, j] = cost + min(dtw[i-1, j  ],
                                   dtw[i  , j-1],
                                   dtw[i-1, j-1])

    return dtw[n, m] 
