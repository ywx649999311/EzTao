"""Functions related to CARMA process simulations.
"""

import numpy as np
from math import ceil
import celerite
from celerite import GP
from eztao.ts.utils import add_season, downsample_byN, downsample_byTime
from eztao.carma.CARMATerm import DRW_term

__all__ = ["gpSimFull", "gpSimRand", "gpSimByTime"]


def gpSimFull(carmaTerm, SNR, duration, N, nLC=1, log_flux=True):
    """Simulate full CARMA time series.

    Args:
        carmaTerm (object): celerite GP term.
        SNR (float): Signal to noise ratio defined as ratio between
            CARMA amplitude and the mode of the errors (simulated using
            log normal).
        duration (float): The duration of the simulated time series in days.
        N (int): The number of data points.
        nLC (int, optional): Number of light curves to simulate. Defaults to 1.
        log_flux (bool): Whether the flux/y values should be in log scale, i.e.,
            magnitude. This argument affects how errors are assigned. Defaluts to True.

    Raises:
        RuntimeError: If the input CARMA term/model is not stable, thus cannot be
            solved by celerite.

    Returns:
        Arrays: t, y and yerr of the simulated light curves in numpy arrays.
            Note that errors have been added to y.
    """

    assert isinstance(
        carmaTerm, celerite.celerite.terms.Term
    ), "carmaTerm must a celerite GP term"

    if (not isinstance(carmaTerm, DRW_term)) and (carmaTerm._arroots.real > 0).any():
        raise RuntimeError(
            "The covariance matrix of the provided CARMA term is not positive definite!"
        )

    t = np.linspace(0, duration, N)
    noise = carmaTerm.get_rms_amp() / SNR
    yerr = np.random.lognormal(0, 0.25, N) * noise
    yerr = yerr[np.argsort(np.abs(yerr))]  # small->large

    # init GP and solve matrix
    gp_sim = GP(carmaTerm)
    gp_sim.compute(t)

    # simulate, assign yerr based on y
    t = np.repeat(t[None, :], nLC, axis=0)
    y = gp_sim.sample(size=nLC)

    # format yerr making it heteroscedastic
    yerr = np.repeat(yerr[None, :], nLC, axis=0)

    # if in mag, large value with large error; in flux, the opposite
    if log_flux:
        # ascending sort
        y_rank = y.argsort(axis=1).argsort(axis=1)
        yerr = np.array(list(map(lambda x, y: x[y], yerr, y_rank)))
    else:
        # descending sort
        y_rank = (-y).argsort(axis=1).argsort(axis=1)
        yerr = np.array(list(map(lambda x, y: x[y], yerr, y_rank)))

    yerr_sign = np.random.binomial(1, 0.5, yerr.shape)
    yerr_sign[yerr_sign < 1] = -1
    yerr = yerr * yerr_sign

    if nLC == 1:
        return t[0], y[0] + yerr[0], yerr[0]
    else:
        return t, y + yerr, yerr


def gpSimRand(
    carmaTerm, SNR, duration, N, nLC=1, log_flux=True, season=True, full_N=10_000
):
    """Simulate randomly downsampled CARMA time series.

    Args:
        carmaTerm (object): celerite GP term.
        SNR (float): Signal to noise ratio defined as ratio between
            CARMA amplitude and the mode of the errors (simulated using
            log normal).
        duration (float): The duration of the simulated time series in days.
        N (int): The number of data points in the returned light curves.
        nLC (int, optional): Number of light curves to simulate. Defaults to 1.
        log_flux (bool): Whether the flux/y values should be in log scale, i.e.,
            magnitude. This argument affects how errors are assigned. Defaluts to True.
        season (bool, optional): Whether to simulate seasonal gaps.
            Defaults to True.
        full_N (int, optional): The number of data points the full light curves.
            Defaults to 10_000.

    Returns:
        Arrays: t, y and yerr of the simulated light curves in numpy arrays.
            Note that errors have been added to y.
    """
    t, y, yerr = gpSimFull(carmaTerm, SNR, duration, full_N, nLC=nLC, log_flux=log_flux)
    t = np.atleast_2d(t)
    y = np.atleast_2d(y)
    yerr = np.atleast_2d(yerr)

    # output t & yerr
    tOut = np.empty((nLC, N))
    yOut = np.empty((nLC, N))
    yerrOut = np.empty((nLC, N))

    # downsample
    for i in range(nLC):
        if season:
            mask1 = add_season(t[i])
        else:
            mask1 = np.ones(t[i].shape[0], dtype=np.bool)
        mask2 = downsample_byN(t[i, mask1], N)
        tOut[i, :] = t[i, mask1][mask2]
        yOut[i, :] = y[i, mask1][mask2]
        yerrOut[i, :] = yerr[i, mask1][mask2]

    if nLC == 1:
        return tOut[0], yOut[0], yerrOut[0]
    else:
        return tOut, yOut, yerrOut


def gpSimByTime(carmaTerm, SNR, t, factor=10, nLC=1, log_flux=True):
    """Simulate CARMA time series at the provided timestamps.

    This function uses a 'factor' parameter to determine the sampling rate
    of the initial(full) time series to simulate and downsample from. For
    example, if 'factor' = 10, then the initial time series will be 10 times
    denser than the median sampling rate of the provided timestamps.

    Args:
        carmaTerm (object): celerite GP term.
        SNR (float): Signal to noise ratio defined as ratio between
            CARMA amplitude and the mode of the errors (simulated using
            log normal).
        t (int): Input timestamps.
        factor (int, optional): Paramter to control the ratio in the sampling
            ratebetween the simulated full time series and the desired output.
            Defaults to 10.
        nLC (int, optional): Number of light curves to simulate. Defaults to 1.
        log_flux (bool): Whether the flux/y values should be in log scale, i.e.,
            magnitude. This argument affects how errors are assigned. Defaluts to True.

    Returns:
        Arrays: t, y and yerr of the simulated light curves in numpy arrays.
            Note that errors have been added to y.
    """
    # get number points in full LC based on desired cadence
    duration = ceil(t[-1] - t[0])
    N = factor * ceil(duration / np.median(t[1:] - t[:-1]))

    # simulate full LC
    tFull, yFull, yerrFull = gpSimFull(
        carmaTerm, SNR, duration, N=N, nLC=nLC, log_flux=log_flux
    )
    tFull = np.atleast_2d(tFull)
    yFull = np.atleast_2d(yFull)
    yerrFull = np.atleast_2d(yerrFull)

    # downsample by desired output cadence
    t_expand = np.repeat(t[None, :], nLC, axis=0)
    tOut_idx = np.array(list(map(downsample_byTime, tFull, t_expand)))
    tOut = np.array(list(map(lambda x, y: x[y], tFull, tOut_idx)))
    yOut = np.array(list(map(lambda x, y: x[y], yFull, tOut_idx)))
    yerrOut = np.array(list(map(lambda x, y: x[y], yerrFull, tOut_idx)))

    if nLC == 1:
        return tOut[0], yOut[0], yerrOut[0]
    else:
        return tOut, yOut, yerrOut