import numpy as np
from numba import njit, float64, int32, boolean, vectorize, guvectorize
from scipy.spatial import KDTree
from functools import partial

__all__ = ["downsample_byN", "add_season", "downsample_byT"]


def downsample_byN(t, nObs):
    """Utility function to randomly choose N observation from a given light curves

    Args:
        t (array_like): Time stamp of observations in the original light curve.
        N (int): The number of observations in the final light curve.

    Returns:
        An mask of the original length to select data point.
    """
    # random choose index
    idx = np.arange(len(t))
    mask = np.zeros_like(idx, dtype=np.bool_)
    true_idx = np.random.choice(idx, nObs, replace=False)

    # assign chosen index to 1/True
    mask[true_idx] = 1

    return mask


def _get_nearest_idx(tIn, x):
    """Internal function to return nearest neighbor in an array."""
    return (np.abs(tIn - x)).argmin()


def downsample_byT(tIn, tOut):
    """Find the indices of a downsampled light curve given output timestamps.

    Args:
        tIn (object): Numpy array containing the timestamps of the original
            light curve.
        tOut (object): Numpy array containing the timestamps of the ouput
            light curve.

    Returns:
        An numpy array of indices, which can be used to select data points from
            the original light curve. Note that there could be duplicates.
    """
    get_nearest = partial(_get_nearest_idx, tIn)
    return np.array(list(map(get_nearest, tOut)))


def add_season(t, lc_start=0, season_start=90, season_end=270):
    """Utility function to impose seasonal gap in mock light curves

    Args:
        t (array_like): Time stamp of observations in a light curve.
        lc_start (float): Light curve starting day within a year (0 -> 365.25). Default to 0.
        season_start (float): Observing season start day within a year. Default to 90.
        season_end (float): Observing season end day within a year. Default to 270.

    Returns:
        An mask of the original length to select data point.
    """
    t = t - t[0]
    t = t + lc_start

    mask = (np.mod(t, 365.25) > season_start) & (np.mod(t, 365.25) < season_end)

    return mask
