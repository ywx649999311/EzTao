"""
A collection of utility functions to assist analysis/simulation of time series data.
"""

import numpy as np
from numba import njit
from functools import partial

__all__ = ["downsample_byN", "add_season", "downsample_byTime", "median_clip"]


def downsample_byN(t, nObs):
    """
    Randomly choose N data points from a given time series.

    Args:
        t (array(float)): Time stamps of the original time series.
        N (int): The number of entries in the final time series.

    Returns:
        A 1d array booleans indicating which data points to keep.
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


def downsample_byTime(tIn, tOut):
    """
    Downsample a time series at desired output time stamps.

    Args:
        tIn (array(float)): Time stamps of the original time series.
        tOut (array(float)): Time stamps of the output time series.

    Returns:
        array(int): Indices for which the data points should be kept from the original 
        time series. Note that there could be duplicates.
    """
    get_nearest = partial(_get_nearest_idx, tIn)
    return np.array(list(map(get_nearest, tOut)))


def add_season(t, lc_start=0, season_start=90, season_end=270):
    """
    Insert seasonal gaps into time series

    Args:
        t (array(float)): Time stamps of the original time series.
        lc_start (float): Starting day for the output time series. (0 -> 365.25). 
            Default to 0.
        season_start (float): Observing season start day within a year. Default to 90.
        season_end (float): Observing season end day within a year. Default to 270.

    Returns:
        A 1d array booleans indicating which data points to keep.
    """
    t = t - t[0]
    t = t + lc_start

    mask = (np.mod(t, 365.25) > season_start) & (np.mod(t, 365.25) < season_end)

    return mask


@njit
def median_clip(y, num_sigma=3):
    """
    Clip time series using a three point median filter.

    The sigma (standard deviation) for the time series is computed from the median  absolute deviation (MAD) as to reduce the effects from extreme outliers, where
    sigma \sim 1.4826*MAD. If more than 10% of the data points are removed, the upper
    bound will be lifted gradually until that fraction drops bellow 10%.

    Args:
        y (array(float)): y values of the original time series.
        num_sigma (int, optional): Data points that are more than this number of sigma
            away from the three point median will be removed. Defaults to 3.

    Returns:
        A 1d array booleans indicating which data points to keep.
    """
    y = np.atleast_1d(y)
    lc_len = y.shape[0]
    sigma = 0.6745 * np.median(np.abs(y - np.median(y)))
    med = y.copy()
    med[1:-2] = [np.median(y[i - 1 : i + 2]) for i in range(1, lc_len - 2)]

    # need to deal with edges of median filter
    med[0] = np.median(np.array([y[-1], y[0], y[1]]))
    med[-1] = np.median(np.array([y[-2], y[-1], y[0]]))

    # compute residual
    res = np.abs(y - med)

    # set clipping thresh hold
    raise_bar = True
    thresh = num_sigma * sigma

    # if remove too much, raise bar until only remove 10%
    while raise_bar:
        ratio = np.sum(res > thresh) / lc_len
        if ratio < 0.1:
            break
        else:
            thresh += 0.1
    return res < thresh
