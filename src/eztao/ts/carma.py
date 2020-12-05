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
    "gpSimByT",
    "drw_fit",
    "dho_fit",
    "carma_fit",
    "neg_fcoeff_ll",
    "neg_param_ll",
    "drw_log_param_init",
    "dho_log_param_init",
    "carma_log_param_init",
]


def gpSimFull(carmaTerm, SNR, duration, N, nLC=1):
    """Simulate full CARMA time series.

    Args:
        carmaTerm (object): celerite GP term.
        SNR (float): Signal to noise ratio defined as ratio between
            CARMA amplitude and the mode of the errors (simulated using
            log normal).
        duration (float): The duration of the simulated time series in days.
        N (int): The number of data points.
        nLC (int, optional): Number of light curves to simulate. Defaults to 1.

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
    yerr = yerr[np.argsort(np.abs(yerr))]

    # init GP and solve matrix
    gp_sim = GP(carmaTerm)
    gp_sim.compute(t)

    # simulate, assign yerr based on y
    t = np.repeat(t[None, :], nLC, axis=0)
    y = gp_sim.sample(size=nLC)

    # format yerr making it heteroscedastic
    y_rank = y.argsort(axis=1).argsort(axis=1)
    yerr = np.repeat(yerr[None, :], nLC, axis=0)
    yerr = np.array(list(map(lambda x, y: x[y], yerr, y_rank)))
    yerr_sign = np.random.binomial(1, 0.5, yerr.shape)
    yerr_sign[yerr_sign < 1] = -1
    yerr = yerr * yerr_sign

    if nLC == 1:
        return t[0], y[0] + yerr[0], yerr[0]
    else:
        return t, y + yerr, yerr


def gpSimRand(carmaTerm, SNR, duration, N, nLC=1, season=True, full_N=10_000):
    """Simulate randomly downsampled CARMA time series.

    Args:
        carmaTerm (object): celerite GP term.
        SNR (float): Signal to noise ratio defined as ratio between
            CARMA amplitude and the mode of the errors (simulated using
            log normal).
        duration (float): The duration of the simulated time series in days.
        N (int): The number of data points in the returned light curves.
        nLC (int, optional): Number of light curves to simulate. Defaults to 1.
        season (bool, optional): Whether to simulate seasonal gaps.
            Defaults to True.
        full_N (int, optional): The number of data points the full light curves.
            Defaults to 10_000.

    Returns:
        Arrays: t, y and yerr of the simulated light curves in numpy arrays.
            Note that errors have been added to y.
    """
    t, y, yerr = gpSimFull(carmaTerm, SNR, duration, full_N, nLC=nLC)
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


def gpSimByT(carmaTerm, SNR, t, factor=10, nLC=1):
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

    Returns:
        Arrays: t, y and yerr of the simulated light curves in numpy arrays.
            Note that errors have been added to y.
    """
    # get number points in full LC based on desired cadence
    duration = ceil(t[-1] - t[0])
    N = factor * ceil(duration / np.median(t[1:] - t[:-1]))

    # simulate full LC
    tFull, yFull, yerrFull = gpSimFull(carmaTerm, SNR, duration, N=N, nLC=nLC)
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
    """CARMA neg log likelihood function.

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
    neg_ll = -np.inf

    try:
        gp.kernel.set_log_fcoeffs(fcoeffs)
        neg_ll = -gp.log_likelihood(y)
    except celerite.solver.LinAlgError as c:
        print(c)
    except Exception as e:
        pass

    return neg_ll


def neg_param_ll(params, y, gp):
    """CARMA neg log likelihood function.

    This method will catch 'overflow/underflow' runtimeWarning and
    return -inf as probablility.

    Args:
        params (object): Array-like, CARMA parameters.
        y (object): Array-like, y values of the time series.
        gp (object): celerite GP model with the proper kernel.

    Returns:
        float: neg log likelihood.
    """

    # change few runtimewarning action setting
    notify_method = "raise"
    np.seterr(over=notify_method)
    np.seterr(under=notify_method)
    neg_ll = -np.inf

    try:
        gp.set_parameter_vector(params)
        neg_ll = -gp.log_likelihood(y)
        # break
    except celerite.solver.LinAlgError as c:
        print(c)
    except Exception as e:
        pass

    return neg_ll


def drw_log_param_init(std):
    """Randomly generate DRW parameters.

    Args:
        std (float): The std of the LC to fit.
    Returns:
        list: The generated DRW parameters in natural log.
    """

    init_tau = np.exp(np.random.uniform(0, 6, 1)[0])
    init_amp = np.random.uniform(0, 4 * std, 1)[0]

    return np.log([init_amp, init_tau])


def dho_log_param_init():
    """Randomly generate DHO parameters.

    Returns:
        list: The generated DHO parameters in natural log.
    """

    log_a1 = np.random.uniform(-10, 5, 1)[0]
    log_a2 = np.random.uniform(-10, 0, 1)[0]
    log_b0 = np.random.uniform(-10, 0, 1)[0]
    log_b1 = np.random.uniform(-10, 0, 1)[0]

    return np.array([log_a1, log_a2, log_b0, log_b1])


def carma_log_param_init(dim):
    """Randomly generate DHO parameters from [-8, 1] in log.

    Args:
        dim (int): For a CARMA(p,q) model, dim=p+q+1.
    Returns:
        list: The generated CAMRA parameters in natural log.
    """

    log_param = np.random.uniform(-5, 5, int(dim))

    return log_param


def carma_log_fcoeff_init(dim):
    """Randomly generate DHO parameters from [-8, 1] in log.

    Args:
        dim (int): For a CARMA(p,q) model, dim=p+q+1.
    Returns:
        list: The generated CAMRA parameters in natural log.
    """

    log_param = np.random.uniform(-6, 3, int(dim))

    return log_param


def sample_carma(p, q):
    """Randomly drawing a valid CARMA process given the order.

    Args:
        p (int): CARMA p order.
        q (int): CARMA q order.

    Returns:
        AR parameters and MA paramters in seperate arrays.
    """
    init_fcoeffs = np.exp(carma_log_fcoeff_init(p + q + 1))
    ARpars = fcoeffs2coeffs(np.append(init_fcoeffs[:p], [1]))[:-1][::-1]
    MApars = fcoeffs2coeffs(init_fcoeffs[p:])

    return ARpars, MApars


def _de_opt(y, best_fit, gp, init_func, mode, debug, bounds):
    """Differential Evolution optimizer wrapper.

    Args:
        y (object): An array of y values.
        best_fit (object): An empty array to store best fit parameters.
        gp (object): celerite GP model object.
        init_func (object): CARMA parameter initialization function,
            i.e. drw_log_param_init.
        mode (str): Specify which space to sample, 'param' or 'coeff'.
        debug (bool): Turn on/off debug mode.
        bounds (list): Initial parameter boundaries for the optimizer.

    Returns:
        object: An array of best-fit parameters
    """

    # dynamic control of fitting flow
    succeded = False  # ever succeded
    run_ct = 0
    jac_log_rec = 10

    # set the neg_ll function based on mode
    neg_ll = neg_fcoeff_ll if mode == "coeff" else neg_param_ll

    # set bound based on LC std for amp
    while run_ct < 5:
        run_ct += 1
        r = differential_evolution(neg_ll, bounds=bounds, args=(y, gp), maxiter=200)

        if r.success:
            succeded = True
            if mode == "param":
                best_fit[:] = np.exp(r.x)
            else:
                gp.kernel.set_log_fcoeffs(r.x)
                best_fit[:] = np.exp(gp.get_parameter_vector())

            if "jac" not in r.keys():
                run_ct += 5
            else:
                jac_log = np.log10(np.dot(r.jac, r.jac) + 1e-8)

                # if positive jac, then increase bounds
                if jac_log > 0:
                    bounds = [(x[0] - 1, x[1] + 1) for x in bounds]
                else:
                    run_ct += 5

                # update best-fit if smaller jac found
                if jac_log < jac_log_rec:
                    jac_log_rec = jac_log

        else:
            bounds = [(x[0] - 1, x[1] + 1) for x in bounds]
            gp.set_parameter_vector(init_func())

    # If opitimizer never reached minima, assign nan
    if not succeded:
        best_fit[:] = np.nan

    if debug:
        print(r)

    return best_fit


def _min_opt(
    y, best_fit, gp, init_func, mode, debug, bounds, n_iter, method="L-BFGS-B"
):
    """A wrapper for scipy.optimize.minimize.

    Args:
        y (object): An array of y values.
        best_fit (object): An empty array to store best fit parameters.
        gp (object): celerite GP model object.
        init_func ([type]): CARMA parameter initialization function,
            i.e. drw_log_param_init.
        mode (str): Specify which space to sample, 'param' or 'coeff'.
        debug (bool, optional): Turn on/off debug mode.
        bounds (list): Initial parameter boundaries for the optimizer.
        n_iter (int, optional): Number of iterations to run the optimizer.
            Defaults to 10.
        method (str, optional): Likelihood optimization method.

    Returns:
        object: An array of best-fit parameters
    """

    # set the neg_ll function based on mode
    neg_ll = neg_fcoeff_ll if mode == "coeff" else neg_param_ll

    # placeholder for ll and sols
    ll, sols, rs = [], [], []

    for i in range(n_iter):
        initial_params = init_func()
        r = minimize(
            neg_ll,
            initial_params,
            method=method,
            bounds=bounds,
            args=(y, gp),
        )

        if r.success:
            if mode == "param":
                gp.kernel.set_parameter_vector(r.x)
            else:
                gp.kernel.set_log_fcoeffs(r.x)

            ll.append(gp.log_likelihood(y))
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


def drw_fit(t, y, yerr, de=True, debug=False, user_bounds=None, n_iter=10):
    """Fix time series to a DRW model.

    Args:
        t (object): An array of time stamps in days.
        y (object): An array of y values.
        yerr (object): An array of the errors in y values.
        de (bool, optional): Whether to use differential_evolution as the
            optimizer. Defaults to True.
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
    kernel = DRW_term(*drw_log_param_init(std))
    gp = GP(kernel, mean=np.median(y))
    gp.compute(t, yerr)

    if de:
        best_fit_return = _de_opt(
            y,
            best_fit,
            gp,
            lambda: drw_log_param_init(std),
            "param",
            debug,
            bounds,
        )
    else:
        best_fit_return = _min_opt(
            y,
            best_fit,
            gp,
            lambda: drw_log_param_init(std),
            "param",
            debug,
            bounds,
            n_iter,
        )

    return best_fit_return


def dho_fit(t, y, yerr, de=True, debug=False, user_bounds=None, n_iter=10):
    """Fix time series to a DHO model.

    Args:
        t (object): An array of time stamps in days.
        y (object): An array of y values.
        yerr (object): An array of the errors in y values.
        de (bool, optional): Whether to use differential_evolution as the
            optimizer. Defaults to True.
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
    best_fit = np.zeros(4)

    if user_bounds is not None and (len(user_bounds) == 4):
        bounds = user_bounds
    else:
        bounds = [(-10, 13), (-14, 7), (-10, 6), (-12, 3)]

    # re-position lc
    t = t - t[0]
    y = y - np.median(y)

    # initialize parameter, kernel and GP
    kernel = DHO_term(*dho_log_param_init())
    gp = GP(kernel, mean=np.mean(y))
    gp.compute(t, yerr)

    if de:
        best_fit_return = _de_opt(
            y,
            best_fit,
            gp,
            lambda: dho_log_param_init(),
            "param",
            debug,
            bounds,
        )
    else:
        best_fit_return = _min_opt(
            y,
            best_fit,
            gp,
            lambda: dho_log_param_init(),
            "param",
            debug,
            bounds,
            n_iter,
        )

    return best_fit_return


def carma_fit(
    t, y, yerr, p, q, de=True, debug=False, mode="coeff", user_bounds=None, n_iter=10
):
    """Fit time series to any CARMA model.

    Args:
        t (object): An array of time stamps in days.
        y (object): An array of y values.
        yerr (object): An array of the errors in y values.
        p (int): P order of a CARMA(p, q) model.
        q (int): Q order of a CARMA(p, q) model.
        de (bool, optional): Whether to use differential_evolution as the
            optimizer. Defaults to True.
        debug (bool, optional): Turn on/off debug mode. Defaults to False.
        mode (str, optional): Specify which space to sample, 'param' or 'coeff'.
            Defaults to 'coeff'.
        user_bounds (list, optional): Factorized polynomial coefficient boundaries
            for the optimizer. Defaults to None.
        n_iter (int, optional): Number of iterations to run the optimizer if de==False.
            Defaults to 10.

    Raises:
        celerite.solver.LinAlgError: For non-positive definite matrices.

    Returns:
        object: An array of best-fit CARMA parameters
    """
    dim = int(p + q + 1)
    best_fit = np.empty(dim)

    # init bounds for fitting
    if user_bounds is not None and (len(user_bounds) == dim):
        bounds = user_bounds
    elif p == 2 and q == 1:
        bounds = [(-10, 13), (-14, 7), (-10, 6), (-12, 3)]
    elif p == 2 and q == 0:
        bounds = [(-10, 16), (-14, 16), (-13, 15)]
    else:
        ARbounds = [(-6, 1)] * p
        MAbounds = [(-6, -1)] * (q + 1)
        bounds = ARbounds + MAbounds

    # re-position lc
    t = t - t[0]
    y = y - np.median(y)

    # initialize parameter and kernel
    ARpars, MApars = sample_carma(p, q)
    kernel = CARMA_term(np.log(ARpars), np.log(MApars))
    gp = GP(kernel, mean=np.median(y))
    gp.compute(t, yerr)

    if mode == "coeff":
        init_func = lambda: carma_log_fcoeff_init(dim)
    else:
        init_func = lambda: carma_log_param_init(dim)

    if de:
        best_fit_return = _de_opt(
            y,
            best_fit,
            gp,
            init_func,
            mode,
            debug,
            bounds,
        )
    else:
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
