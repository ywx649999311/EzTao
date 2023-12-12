"""
A collection of functions for simulating CARMA processes.
"""

import numpy as np
from math import ceil
import celerite
from celerite import GP
from eztao.ts.utils import add_season, downsample_byN, downsample_byTime
from eztao.carma.CARMATerm import DRW_term, CARMA_term

__all__ = ["gpSimFull", "gpSimRand", "gpSimByTime", "addNoise", "pred_lc"]


def gp_sample(gp, size=None, seed=None):
    """Sample Celerite GP with a fixed seed"""

    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()

    gp._recompute()
    if size is None:
        n = rng.standard_normal(len(gp._t))
    else:
        n = rng.standard_normal((len(gp._t), size))

    n = gp.solver.dot_L(n)
    if size is None:
        return gp.mean.get_value(gp._t) + n[:, 0]
    return gp.mean.get_value(gp._t)[None, :] + n.T


def gpSimFull(carmaTerm, SNR, duration, N, nLC=1, log_flux=True, lc_seed=None):
    """
    Simulate CARMA time series using uniform sampling.

    Args:
        carmaTerm (object): An EzTao CARMA kernel.
        SNR (float): Signal-to-noise defined as ratio between CARMA RMS amplitude and
            the median of the measurement errors (simulated using log normal).
        duration (float): The duration of the simulated time series (default in days).
        N (int): The number of data points in the simulated time series.
        nLC (int, optional): Number of time series to simulate. Defaults to 1.
        log_flux (bool): Whether the flux/y values are in astronomical magnitude.
            This argument affects how errors are assigned. Defaults to True.
        lc_seed (int): Random seed for time series simulation. Defaults to None.

    Raises:
        RuntimeError: If the input CARMA term/model is not stable, thus cannot be
            solved by celerite.

    Returns:
        (array(float), array(float), array(float)): Time stamps (default in day), y
        values and measurement errors of the simulated time series.
    """

    assert isinstance(
        carmaTerm, celerite.celerite.terms.Term
    ), "carmaTerm must a celerite GP term"

    if (not isinstance(carmaTerm, DRW_term)) and (carmaTerm._arroots.real > 0).any():
        raise RuntimeError(
            "The covariance matrix of the provided CARMA term is not positive definite!"
        )

    if lc_seed is not None:
        rng = np.random.default_rng(seed=lc_seed)
    else:
        rng = np.random.default_rng()

    t = np.linspace(0, duration, N)
    noise = carmaTerm.get_rms_amp() / SNR
    yerr = rng.lognormal(0, 0.25, N) * noise
    yerr = yerr[np.argsort(np.abs(yerr))]  # small->large

    # init GP and solve matrix
    gp_sim = GP(carmaTerm)
    gp_sim.compute(t)

    # simulate, assign yerr based on y
    t = np.repeat(t[None, :], nLC, axis=0)
    y = gp_sample(gp_sim, size=nLC, seed=lc_seed)

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

    if nLC == 1:
        return t[0], y[0], yerr[0]
    else:
        return t, y, yerr


def gpSimRand(
    carmaTerm,
    SNR,
    duration,
    N,
    nLC=1,
    log_flux=True,
    season=True,
    full_N=10_000,
    lc_seed=None,
    downsample_seed=None,
):
    """
    Simulate CARMA time series randomly downsampled from a much denser full time series.

    Args:
        carmaTerm (object): An EzTao CARMA kernel.
        SNR (float): Signal-to-noise defined as ratio between CARMA RMS amplitude and
            the median of the measurement errors (simulated using log normal).
        duration (float): The duration of the simulated time series (default in days).
        N (int): The number of data points in the simulated time series.
        nLC (int, optional): Number of time series to simulate. Defaults to 1.
        log_flux (bool): Whether the flux/y values are in astronomical magnitude.
            This argument affects how errors are assigned. Defaults to True.
        season (bool, optional): Whether to simulate 6-months seasonal gaps. Defaults
            to True.
        full_N (int, optional): The number of data points in the full time series
            (before downsampling). Defaults to 10_000.
        lc_seed (int): Random seed for full time series simulation. Defaults to None.
        downsample_seed (int): Random seed for downsampling the simulated full time
            series. Defaults to None.

    Returns:
        (array(float), array(float), array(float)): Time stamps (default in day), y
        values and measurement errors of the simulated time series.
    """
    t, y, yerr = gpSimFull(
        carmaTerm, SNR, duration, full_N, nLC=nLC, log_flux=log_flux, lc_seed=lc_seed
    )
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
            mask1 = np.ones(t[i].shape[0], dtype=bool)
        mask2 = downsample_byN(t[i, mask1], N, seed=downsample_seed)
        tOut[i, :] = t[i, mask1][mask2]
        yOut[i, :] = y[i, mask1][mask2]
        yerrOut[i, :] = yerr[i, mask1][mask2]

    if nLC == 1:
        return tOut[0], yOut[0], yerrOut[0]
    else:
        return tOut, yOut, yerrOut


def gpSimByTime(carmaTerm, SNR, t, factor=10, nLC=1, log_flux=True, lc_seed=None):
    """
    Simulate CARMA time series at desired time stamps.

    This function uses a 'factor' parameter to determine the sampling rate of a full
    time series to simulate and downsample from. For example, if 'factor' = 10, then
    the full time series will be 10 times denser than the median sampling rate of the
    provided time stamps.

    Args:
        carmaTerm (object): An EzTao CARMA kernel.
        SNR (float): Signal-to-noise defined as ratio between CARMA RMS amplitude and
            the median of the measurement errors (simulated using log normal).
        t (array(float)): The desired time stamps (starting from zero).
        factor (int, optional): Parameter to control the ratio in the sampling
            rate between the simulated full time series and the desired output one.
            Defaults to 10.
        nLC (int, optional): Number of time series to simulate. Defaults to 1.
        log_flux (bool): Whether the flux/y values are in astronomical magnitude.
            This argument affects how errors are assigned. Defaults to True.
        lc_seed (int): Random seed for time series simulation. Defaults to None.

    Returns:
        (array(float), array(float), array(float)): Time stamps (default in day), y
        values and measurement errors of the simulated time series.
    """
    # get number points in full LC based on desired cadence
    duration = ceil(t[-1] - t[0])
    N = int(factor * ceil(duration / np.median(t[1:] - t[:-1])))

    # simulate full LC
    tFull, yFull, yerrFull = gpSimFull(
        carmaTerm, SNR, duration, N=N, nLC=nLC, log_flux=log_flux, lc_seed=lc_seed
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


def addNoise(y, yerr, seed=None):
    """
    Add (gaussian) noise to the input simulated time series given the measurement uncertainties.

    Args:
        y (array(float)): The 'clean' time series.
        yerr (array(float)): The measurement uncertainties for the input
            time series.
    seed (int): Random seed for simulating noise. Defaults to None.

    Returns:
        array(float): A new time series with simulated (gaussian ) noise added
        on top.
    """

    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()

    vec_norm = np.vectorize(rng.normal, signature="(n),(n)->(n)")
    noise = vec_norm(np.zeros_like(y), yerr)

    return y + noise


def pred_lc(t, y, yerr, params, p, t_pred, return_var=True):
    """
    Generate predicted values at particular time stamps given the initial
    time series and a best-fit model.

    Args:
        t (array(float)): Time stamps of the initial time series.
        y (array(float)): y values (i.e., flux) of the initial time series.
        yerr (array(float)): Measurement errors of the initial time series.
        params (array(float)): Best-fit CARMA parameters
        p (int): The AR order (p) of the given best-fit model.
        t_pred (array(float)): Time stamps to generate predicted time series.
        return_var (bool, optional): Whether to return uncertainties in the mean
            prediction. Defaults to True.

    Returns:
        (array(float), array(float), array(float)): t_pred, mean prediction at t_pred
        and uncertainties (variance) of the mean prediction.
    """

    assert p >= len(params) - p, "The dimension of AR must be greater than that of MA"

    # get ar, ma
    ar = params[:p]
    ma = params[p:]

    # reposition lc
    y_aln = y - np.median(y)

    # init kernel, gp and compute matrix
    kernel = CARMA_term(np.log(ar), np.log(ma))
    gp = celerite.GP(kernel, mean=0)
    gp.compute(t, yerr)

    try:
        mu, var = gp.predict(y_aln, t_pred, return_var=return_var)
    except FloatingPointError as e:
        print(e)
        print("No (super small) variance will be returned")
        return_var = False
        mu, var = gp.predict(y_aln, t_pred, return_var=return_var)

    return t_pred, mu + np.median(y), var
