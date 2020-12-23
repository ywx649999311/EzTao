"""Functions for performing CARMA analysis on LCs.
"""

import numpy as np
from math import ceil
from scipy.optimize import differential_evolution, minimize
import celerite
from celerite import GP
from eztao.carma.CARMATerm import DRW_term, DHO_term, CARMA_term, fcoeffs2coeffs
from eztao.ts.utils import *

__all__ = [
    "gpSimFull",
    "gpSimRand",
    "gpSimByTime",
    "drw_fit",
    "dho_fit",
    "carma_fit",
    "neg_fcoeff_ll",
    "neg_param_ll",
    "drw_log_param_init",
    "carma_log_param_init",
]


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
    tOut_idx = np.array(list(map(downsample_byT, tFull, t_expand)))
    tOut = np.array(list(map(lambda x, y: x[y], tFull, tOut_idx)))
    yOut = np.array(list(map(lambda x, y: x[y], yFull, tOut_idx)))
    yerrOut = np.array(list(map(lambda x, y: x[y], yerrFull, tOut_idx)))

    if nLC == 1:
        return tOut[0], yOut[0], yerrOut[0]
    else:
        return tOut, yOut, yerrOut


## Below is about Fitting
# -------------------------------------------------------------------------------------
def neg_fcoeff_ll(fcoeffs, y, gp):
    """Negative CARMA log likelihood function.

    This method will catch 'overflow/underflow' runtimeWarning and
    return -inf as probablility.

    Args:
        fcoeffs (object): Array-like, CARMA polynomial coeffs in the facotrized form.
        y (object): Array-like, y values of the time series.
        gp (object): celerite GP model with the proper kernel.

    Returns:
        float: neg log likelihood.
    """

    assert gp.kernel.p >= 2, "Use neg_param_ll() instead!"

    # change few runtimewarning action setting
    notify_method = "raise"
    np.seterr(over=notify_method)
    np.seterr(under=notify_method)
    neg_ll = np.inf

    try:
        gp.kernel.set_log_fcoeffs(fcoeffs)
        neg_ll = -gp.log_likelihood(y)
    except celerite.solver.LinAlgError as c:
        # print(c)
        pass
    except Exception as e:
        pass

    return neg_ll


def neg_param_ll(params, y, gp):
    """Negative CARMA log likelihood function.

    This method will catch 'overflow/underflow' runtimeWarning and
    return -inf as probablility.

    Args:
        params (object): Array-like, CARMA parameters.
        y (object): Array-like, y values of the time series.
        gp (object): celerite GP model with the proper kernel.

    Returns:
        float: neg log likelihood.
    """
    assert gp.kernel.p <= 2, "Use neg_fcoeff_ll() instead!"

    # change few runtimewarning action setting
    notify_method = "raise"
    np.seterr(over=notify_method)
    np.seterr(under=notify_method)
    neg_ll = np.inf

    try:
        gp.set_parameter_vector(params)
        neg_ll = -gp.log_likelihood(y)
    except celerite.solver.LinAlgError as c:
        # print(c)
        pass
    except Exception as e:
        pass

    return neg_ll


def drw_log_param_init(std, size=1, max_tau=6.0):
    """Randomly generate DRW parameters.

    Args:
        std (float): The std of the LC to fit.
        size (int, optional): The number of the set of CARMA parameters to generate.
            Defaults to 1.
        max_tau (float): The maximum likely tau in the natual log. Defaults to 6.0.

    Returns:
        Array: The generated DRW parameters in natural log.
    """

    init_tau = np.exp(np.random.rand(size, 1) * max_tau)
    init_amp = np.random.rand(size, 1) * 4 * std
    drw_param = np.hstack((init_amp, init_tau))

    if size == 1:
        return drw_param[0]
    else:
        return drw_param


def carma_log_param_init(p, q, ranges=None, size=1, a=-8.0, b=8.0, shift=0):
    """Randomly generate CARMA parameters from [a, b) in log.

    Args:
        dim (int): For a CARMA(p,q) model, dim=p+q+1.
        ranges (list, optional): A list of tuples of custom ranges to draw parameter
            proposals from. Defaults to None.
        size (int, optional): The number of the set of CARMA parameters to generate.
            Defaults to 1.
        a (float, optional): The lower bound of the ranges, if a range for a specific
            parameter is not specified. Defaults to -8.0.
        b (float, optional): The upper bound of the ranges, if a range for a specific
            parameter is not specified. Defaults to 8.0.

    Returns:
        Array: The generated CAMRA parameters in the natural log.
    """
    dim = int(p + q + 1)
    log_param = np.random.rand(size, int(dim))

    if (ranges is not None) and (len(ranges) == int(dim)):
        for d in range(dim):
            if all(ranges[d]):
                scale = ranges[d][1] - ranges[d][0]
                log_param[:, d] = log_param[:, d] * scale - ranges[d][0]
            else:
                log_param[:, d] = log_param[:, d] * (b - a) + a
    else:
        log_param = log_param * (b - a) + a

    # add shift if amp too large/small
    log_param[:, p:] = log_param[:, p:] + shift

    if size == 1:
        return log_param[0]
    else:
        return log_param


def carma_log_fcoeff_init(p, q, ranges=None, size=1, a=-8.0, b=8.0, shift=0):
    """Randomly generate CARMA poly coefficients from [a, b) in log.

    Args:
        p (int): P order of a CARMA(p, q) model.
        q (int): Q order of a CARMA(p, q) model.
        ranges (list, optional): A list of tuples of custom ranges to draw poly
            coefficient proposals from. Defaults to None.
        size (int, optional): The number of the set of poly coefficients to generate.
            Defaults to 1.
        a (float, optional): The lower bound of the ranges, if a range for a specific
            coefficient is not specified. Defaults to -8.0.
        b (float, optional): The upper bound of the ranges, if a range for a specific
            coefficient is not specified. Defaults to 8.0.
    Returns:
        Array: The generated CAMRA poly coefficients in the natural log.
    """
    dim = int(p + q + 1)
    log_coeff = np.random.rand(size, int(dim))

    if (ranges is not None) and (len(ranges) == int(dim)):
        for d in range(dim):
            if all(ranges[d]):
                scale = ranges[d][1] - ranges[d][0]
                log_coeff[:, d] = log_coeff[:, d] * scale - ranges[d][0]
            else:
                log_coeff[:, d] = log_coeff[:, d] * (b - a) + a
    else:
        log_coeff = log_coeff * (b - a) + a

    # if range for highest order MA not specified
    if (ranges is None) or (not all(ranges[-1])):
        perturb = np.random.rand(size, 1) * 10 - 5
        log_ma_coeff = log_coeff[:, p:]
        low_term = np.zeros((size, 1))

        if q > 0:
            if q % 2 == 0:
                low_term += log_ma_coeff[:, -1][:, np.newaxis]
            for i in range(1, q, 2):
                low_term += log_ma_coeff[:, i][:, np.newaxis]

        # update higher order MA
        log_coeff[:, -1] = -low_term[:, 0] + perturb[:, 0]
        log_coeff[:, -1] += shift

    if size == 1:
        return log_coeff[0]
    else:
        return log_coeff


def sample_carma(p, q, ranges=None, a=-6, b=6, shift=0):
    """Randomly generate a stationary CARMA model given the orders.

    Args:
        p (int): CARMA p order.
        q (int): CARMA q order.
        ranges (list): A list tuple of ranges from which to draw each parameter.
            Defaults to None.

    Returns:
        AR parameters and MA paramters in two seperate arrays.
    """
    init_fcoeffs = np.exp(
        carma_log_fcoeff_init(p, q, ranges=ranges, a=a, b=b, shift=shift)
    )
    ARpars = fcoeffs2coeffs(np.append(init_fcoeffs[:p], [1]))[1:]
    MApars = fcoeffs2coeffs(init_fcoeffs[p:])

    return ARpars, MApars


def _min_opt(
    y, best_fit, gp, init_func, mode, debug, bounds, n_iter, method="L-BFGS-B"
):
    """A wrapper for scipy.optimize.minimize.

    Args:
        y (object): An array of y values.
        best_fit (object): An empty array to store best fit parameters.
        gp (object): celerite GP model object.
        init_func (object): CARMA parameter/coefficient initialization function,
            i.e. drw_log_param_init.
        mode (str): Specify which space to sample, 'param' or 'coeff'.
        debug (bool): Turn on/off debug mode.
        bounds (list): CARMA parameter/coefficient boundaries for the optimizer.
        n_iter (int): Number of iterations to run the optimizer. Defaults to 10.
        method (str, optional): Likelihood optimization method. Defaults to "L-BFGS-B".

    Returns:
        object: An array of best-fit CARMA parameters.
    """

    # set the neg_ll function based on mode
    neg_ll = neg_fcoeff_ll if mode == "coeff" else neg_param_ll

    # placeholder for ll and sols; draw init params
    ll, sols, rs = [], [], []
    initial_params = init_func()

    for i in range(n_iter):
        r = minimize(
            neg_ll,
            initial_params[i],
            method=method,
            bounds=bounds,
            args=(y, gp),
        )

        if r.success and (r.fun != -np.inf):
            if mode == "param":
                gp.kernel.set_parameter_vector(r.x)
            else:
                gp.kernel.set_log_fcoeffs(r.x)

            ll.append(-r.fun)
            sols.append(np.exp(gp.get_parameter_vector()))
        else:
            ll.append(-np.inf)
            sols.append([np.nan] * len(best_fit))

        # save all r for debugging
        rs.append(r)

    best_fit = sols[np.argmax(ll)]

    if debug:
        print(rs)

    return best_fit


def drw_fit(t, y, yerr, debug=False, user_bounds=None, n_iter=10):
    """Fix time series to a DRW model.

    Args:
        t (object): An array of time stamps in days.
        y (object): An array of y values.
        yerr (object): An array of the errors in y values.
        debug (bool, optional): Turn on/off debug mode. Defaults to False.
        user_bounds (list, optional): Parameter boundaries for the optimizer.
            Defaults to None.
        n_iter (int, optional): Number of iterations to run the optimizer if de==False.
            Defaults to 10.

    Raises:
        celerite.solver.LinAlgError: For non-positive definite matrices.

    Returns:
        object: An array of best-fit parameters
    """

    best_fit = np.empty(2)
    std = np.std(y)

    # init bounds for fitting
    if user_bounds is not None and (len(user_bounds) == 2):
        bounds = user_bounds
    else:
        bounds = [(-4, np.log(4 * std)), (-4, 10)]

    # re-position lc
    t = t - t[0]
    y = y - np.median(y)

    # initialize parameter and kernel
    kernel = DRW_term(*drw_log_param_init(std, max_tau=np.log(t[-1] / 8)))
    gp = GP(kernel, mean=0)
    gp.compute(t, yerr)

    best_fit_return = _min_opt(
        y,
        best_fit,
        gp,
        lambda: drw_log_param_init(std, size=n_iter, max_tau=np.log(t[-1] / 8)),
        "param",
        debug,
        bounds,
        n_iter,
    )

    return best_fit_return


def dho_fit(t, y, yerr, debug=False, user_bounds=None, init_ranges=None, n_iter=15):
    """Fix time series to a DHO model.

    Args:
        t (object): An array of time stamps in days.
        y (object): An array of y values.
        yerr (object): An array of the errors in y values.
        debug (bool, optional): Turn on/off debug mode. Defaults to False.
        user_bounds (list, optional): Parameter boundaries for the optimizer.
            Defaults to None.
        init_ranges (list, optional): A list of tuple of custom ranges to draw initial
            parameter proposals from. Defaults to None.
        n_iter (int, optional): Number of iterations to run the optimizer.
            Defaults to 15.

    Raises:
        celerite.solver.LinAlgError: For non-positive definite matrices.

    Returns:
        object: An array of best-fit parameters
    """
    best_fit = np.zeros(4)

    if user_bounds is not None and (len(user_bounds) == 4):
        bounds = user_bounds
    else:
        bounds = [(-15, 15)] * 4

    # re-position lc
    t = t - t[0]
    y = y - np.median(y)

    # determine shift due amp too large/small
    shift = 0
    if np.std(y) < 1e-4 or np.std(y) > 1e-4:
        shift = np.log(np.std(y))
        bounds[2:] += shift

    # initialize parameter, kernel and GP
    kernel = DHO_term(*carma_log_param_init(2, 1, shift=shift))
    gp = GP(kernel, mean=0)
    gp.compute(t, yerr)

    best_fit_return = _min_opt(
        y,
        best_fit,
        gp,
        lambda: carma_log_param_init(
            2, 1, ranges=init_ranges, size=n_iter, shift=shift
        ),
        "param",
        debug,
        bounds,
        n_iter,
    )

    return best_fit_return


def carma_fit(
    t,
    y,
    yerr,
    p,
    q,
    debug=False,
    user_bounds=None,
    init_ranges=None,
    n_iter=10,
):
    """Fit time series to any CARMA model.

    Args:
        t (object): An array of time stamps in days.
        y (object): An array of y values.
        yerr (object): An array of the errors in y values.
        p (int): P order of a CARMA(p, q) model.
        q (int): Q order of a CARMA(p, q) model.
        debug (bool, optional): Turn on/off debug mode. Defaults to False.
        user_bounds (list, optional): Parameter boundaries for the optimizer. If p > 0,
            those are boundaries for the coefficients of the factored polynomial.
            Defaults to None.
        init_ranges (list, optional): A list of tuple of custom ranges to draw initial
            parameter proposals from. If p > 0, same as the user_bounds. Defaults to
            None.
        n_iter (int, optional): Number of iterations to run the optimizer if de==False.
            Defaults to 10.

    Raises:
        celerite.solver.LinAlgError: For non-positive definite matrices.

    Returns:
        object: An array of best-fit CARMA parameters.
    """
    dim = int(p + q + 1)
    best_fit = np.empty(dim)

    # init bounds for fitting
    if user_bounds is not None and (len(user_bounds) == dim):
        bounds = user_bounds
    else:
        bounds = [(-15, 15)] * dim

    # re-position lc
    t = t - t[0]
    y = y - np.median(y)

    # determine shift due amp too large/small
    shift = 0
    if np.std(y) < 1e-4 or np.std(y) > 1e-4:
        shift = np.log(np.std(y))

    # initialize parameter and kernel
    ARpars, MApars = sample_carma(p, q, shift=shift)
    kernel = CARMA_term(np.log(ARpars), np.log(MApars))
    gp = GP(kernel, mean=0)
    gp.compute(t, yerr)

    if p > 2:
        mode = "coeff"
        init_func = lambda: carma_log_fcoeff_init(
            p, q, ranges=init_ranges, size=n_iter, shift=shift
        )
        bounds[-1] += shift
    else:
        mode = "param"
        init_func = lambda: carma_log_param_init(
            p, q, ranges=init_ranges, size=n_iter, shift=shift
        )
        bounds[p:] += shift

    best_fit_return = _min_opt(
        y,
        best_fit,
        gp,
        init_func,
        mode,
        debug,
        bounds,
        n_iter,
    )

    return best_fit_return
