import numpy as np


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

    n = ts1.shape[0]
    m = ts2.shape[0]

    result = np.full((n, n), np.inf)

    result[0][0] = 0.

    for i in range(n):
        result[i][0] = 0.

    for i in range(m):
        result[0][i] = 0.

    for i in range(n):
        for j in range(m):
            d = np.square(ts1[i] - ts2[j])
            d0 = result[i - 1, j]
            d1 = result[i, j - 1]
            d2 = result[i - 1, j - 1]
            result[i, j] = d + np.min([d0, d1, d2])

    dtw_dist = result[n - 1, m - 1]

    return dtw_dist
