"""Functions for performing CARMA analysis on LCs.
"""

import numpy as np
from math import ceil
from scipy.optimize import differential_evolution, minimize
import celerite
from celerite import GP
from eztao.carma.CARMATerm import DRW_term, DHO_term, CARMA_term
from eztao.ts.utils import *

__all__ = [
    "gpSimFull",
    "gpSimRand",
    "gpSimByT",
    "drw_fit",
    "dho_fit",
    "carma_fit",
    "neg_ll",
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
        Exception: If celerite cannot factorize after 10 trials.

    Returns:
        Arrays: t, y and yerr of the simulated light curves in numpy arrays.
            Note that errors are added to y.
    """

    assert isinstance(
        carmaTerm, celerite.celerite.terms.Term
    ), "carmaTerm must a celerite GP term"

    gp_sim = GP(carmaTerm)

    t = np.linspace(0, duration, N)
    noise = carmaTerm.get_rms_amp() / SNR
    yerr = np.random.lognormal(0, 0.25, N) * noise
    yerr = yerr[np.argsort(np.abs(yerr))]

    # factor and factor_num to track factorization error
    factor = True
    fact_num = 0
    yerr_reg = 1.123e-12

    while factor:
        try:
            gp_sim.compute(t, yerr_reg)
            factor = False
        except Exception:
            # if error, try to re-init t and yerr_reg
            # t = np.linspace(0, duration, N)
            yerr_reg += 1.123e-12

            fact_num += 1
            if fact_num > 10:
                raise Exception(
                    "Celerite cannot factorize the GP"
                    + " covairance matrix, try again!"
                )

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
            Note that errors are added to y.
    """
    t, y, yerr = gpSimFull(carmaTerm, SNR, duration, full_N, nLC=nLC)

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
            Note that errors are added to y.
    """
    # get number points in full LC based on desired cadence
    duration = ceil(t[-1] - t[0])
    N = factor * ceil(duration / np.median(t[1:] - t[:-1]))

    # simulate full LC
    tFull, yFull, yerrFull = gpSimFull(carmaTerm, SNR, duration, N=N, nLC=nLC)

    # downsample by desired output cadence
    t_expand = np.repeat(t[None, :], nLC, axis=0)
    tOut_idx = np.array(list(map(downsample_byT, tFull, t_expand)))
    tOut = np.array(list(map(lambda x, y: x[y], tFull, tOut_idx)))
    yOut = np.array(list(map(lambda x, y: x[y], yFull, tOut_idx)))
    yerrOut = np.array(list(map(lambda x, y: x[y], yerrFull, tOut_idx)))

    return tOut, yOut, yerrOut


## Below is about Fitting
def neg_ll(params, y, yerr, gp):
    """CARMA neg log likelihood function.

    This method will catch 'overflow/underflow' runtimeWarning and
    return -inf as probablility.

    Args:
        params (object): Array-like, CARMA parameters.
        y (object): Array-like, y values of the time series.
        yerr (object): Array-like, error in y values of the time series.
        gp (object): celerite GP model with the proper kernel.

    Returns:
        float: neg log likelihood.
    """

    # change few runtimewarning action setting
    notify_method = "raise"
    np.seterr(over=notify_method)
    np.seterr(under=notify_method)

    # params = np.array(params)
    dim = params.shape[0]
    run = True
    lap = 0

    while run:
        if lap > 10:
            return -np.inf

        lap += 1
        try:
            gp.set_parameter_vector(params)
            neg_ll = -gp.log_likelihood(y)
            run = False
        except celerite.solver.LinAlgError:
            params += 1e-6 * np.random.randn(dim)
            continue
        except np.linalg.LinAlgError:
            params += 1e-6 * np.random.randn(dim)
            continue
        except FloatingPointError:
            return -np.inf

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

    log_a1 = np.random.uniform(-10, 1, 1)[0]
    log_a2 = np.random.uniform(-14, -3, 1)[0]
    log_b0 = np.random.uniform(-10, -5, 1)[0]
    log_b1 = np.random.uniform(-10, -5, 1)[0]

    return np.array([log_a1, log_a2, log_b0, log_b1])


def carma_log_param_init(dim):
    """Randomly generate DHO parameters from [-8, 1] in log.

    Args:
        dim (int): For a CARMA(p,q) model, dim=p+q+1.
    Returns:
        list: The generated CAMRA parameters in natural log.
    """

    log_param = np.random.uniform(-8, 2, int(dim))

    return log_param


def _de_opt(y, yerr, best_fit, gp, init_func, debug, bounds):
    """Defferential Evolution optimizer wrapper.

    Args:
        t (object): An array of time stamps in days.
        y (object): An array of y values.
        best_fit (object): An empty array to store best fit parameters.
        gp (object): celerite GP model object.
        init_func ([type]): CARMA parameter initialization function,
            i.e. drw_log_param_init.
        debug (bool, optional): Turn on/off debug mode.
        bounds (list): Initial parameter boundaries for the optimizer.

    Returns:
        object: An array of best-fit parameters
    """

    # dynamic control of fitting flow
    rerun = True
    succeded = False  # ever succeded
    run_ct = 0
    jac_log_rec = 10

    # set bound based on LC std for amp
    while rerun and (run_ct < 5):
        run_ct += 1
        r = differential_evolution(
            neg_ll, bounds=bounds, args=(y, yerr, gp), maxiter=200
        )

        if r.success:
            succeded = True
            best_fit[:] = np.exp(r.x)

            if "jac" not in r.keys():
                rerun = False
            else:
                jac_log = np.log10(np.dot(r.jac, r.jac) + 1e-8)

                # if positive jac, then increase bounds
                if jac_log > 0:
                    bounds = [(x[0] - 1, x[1] + 1) for x in bounds]
                else:
                    rerun = False

                # update best-fit if smaller jac found
                if jac_log < jac_log_rec:
                    jac_log_rec = jac_log
                    best_fit[:] = np.exp(r.x)
        else:
            bounds = [(x[0] - 1, x[1] + 1) for x in bounds]
            gp.set_parameter_vector(init_func())

    # If opitimizer never reached minima, assign nan
    if not succeded:
        best_fit[:] = np.nan

    # Below code is used to visualize if stuck in local minima
    if debug:
        print(r)

    return best_fit


def _min_opt(y, yerr, best_fit, gp, init_func, debug, bounds, method="L-BFGS-B"):
    """A wrapper for scipy.optimize.minimize.

    Args:
        t (object): An array of time stamps in days.
        y (object): An array of y values.
        best_fit (object): An empty array to store best fit parameters.
        gp (object): celerite GP model object.
        init_func ([type]): CARMA parameter initialization function,
            i.e. drw_log_param_init.
        debug (bool, optional): Turn on/off debug mode.
        bounds (list): Initial parameter boundaries for the optimizer.
        method (str): Likelihood optimization method.

    Returns:
        object: An array of best-fit parameters
    """

    # dynamic control of fitting flow
    rerun = True
    succeded = False  # ever succeded
    run_ct = 0
    jac_log_rec = 10

    # set bound based on LC std for amp
    while rerun and (run_ct < 5):
        initial_params = gp.get_parameter_vector()
        run_ct += 1
        r = minimize(
            neg_ll,
            initial_params,
            method=method,
            bounds=bounds,
            args=(y, yerr, gp),
        )
        if r.success:
            succeded = True
            best_fit[:] = np.exp(r.x)

            if "jac" not in r.keys():
                rerun = False
            else:
                jac_log = np.log10(np.dot(r.jac, r.jac) + 1e-8)

                # if positive jac, then increase bounds
                if jac_log > 0:
                    bounds = [(x[0] - 1, x[1] + 1) for x in bounds]
                else:
                    rerun = False

                # update best-fit if smaller jac found
                if jac_log < jac_log_rec:
                    jac_log_rec = jac_log
                    best_fit[:] = np.exp(r.x)
        else:
            bounds = [(x[0] - 1, x[1] + 1) for x in bounds]
            gp.set_parameter_vector(init_func())

    # If opitimizer never reached minima, assign nan
    if not succeded:
        best_fit[:] = np.nan

    # Below code is used to visualize if stuck in local minima
    if debug:
        print(r)

    return best_fit


def drw_fit(t, y, yerr, debug=False, user_bounds=None):
    """Fix time series to DRW model

    Args:
        t (object): An array of time stamps in days.
        y (object): An array of y values.
        yerr (object): An array of the errors in y values.
        debug (bool, optional): Turn on/off debug mode. Defaults to False.
        user_bounds (list, optional): Parameter boundaries for the optimizer.
            Defaults to None.

    Raises:
        Exception: If celerite cannot factorize after 5 trials.

    Returns:
        object: An array of best-fit parameters
    """

    best_fit = np.empty(2)
    std = np.sqrt(np.var(y) - np.var(yerr))

    # init bounds for fitting
    if user_bounds is not None and (len(user_bounds) == 2):
        bounds = user_bounds
    else:
        bounds = [(-4, np.log(4 * std)), (-4, 10)]

    # re-position lc
    t = t - t[0]
    y = y - np.median(y)

    # dynamic control of fitting flow
    compute = True  # handle can't factorize in gp.compute()
    compute_ct = 0

    # initialize parameter and kernel
    kernel = DRW_term(*drw_log_param_init(std))
    gp = GP(kernel, mean=np.median(y))

    # compute can't factorize, try 4 more times
    while compute & (compute_ct < 5):
        compute_ct += 1
        try:
            gp.compute(t, yerr)
            compute = False
        except celerite.solver.LinAlgError:
            if compute_ct > 4:
                raise Exception("celerite can't factorize matrix!")
            gp.set_parameter_vector(drw_log_param_init(std))

    best_fit_return = _de_opt(
        y,
        yerr,
        best_fit,
        gp,
        lambda: drw_log_param_init(std),
        debug,
        bounds,
    )

    return best_fit_return


def dho_fit(t, y, yerr, debug=False, user_bounds=None):
    """Fix time series to DHO model

    Args:
        t (object): An array of time stamps in days.
        y (object): An array of y values.
        yerr (object): An array of the errors in y values.
        debug (bool, optional): Turn on/off debug mode. Defaults to False.
        user_bounds (list, optional): Parameter boundaries for the optimizer.
            Defaults to None.

    Raises:
        Exception: If celerite cannot factorize after 5 trials.

    Returns:
        object: An array of best-fit parameters
    """
    best_fit = np.zeros(4)

    if user_bounds is not None and (len(user_bounds) == 4):
        bounds = user_bounds
    else:
        bounds = [(-10, 7), (-14, 7), (-12, -2), (-11, -2)]

    # re-position lc
    t = t - t[0]
    y = y - np.median(y)

    # dynamic control of fitting flow
    compute = True  # handle can't factorize in gp.compute()
    compute_ct = 0

    # initialize parameter, kernel and GP
    kernel = DHO_term(*dho_log_param_init())
    gp = GP(kernel, mean=np.mean(y))

    # compute can't factorize, try 4 more times
    while compute & (compute_ct < 5):
        compute_ct += 1
        try:
            gp.compute(t, yerr)
            compute = False
        except celerite.solver.LinAlgError:
            if compute_ct > 4:
                raise Exception("celerite can't factorize matrix!")
            gp.set_parameter_vector(dho_log_param_init())

    best_fit_return = _de_opt(
        y,
        yerr,
        best_fit,
        gp,
        lambda: dho_log_param_init(),
        debug,
        bounds,
    )

    return best_fit_return


def carma_fit(t, y, yerr, p, q, de=True, debug=False, user_bounds=None):
    """Fit time series to all CARMA model

    Args:
        t (object): An array of time stamps in days.
        y (object): An array of y values.
        yerr (object): An array of the errors in y values.
        p (int): P order of a CARMA(p, q) model.
        q (int): Q order of a CARMA(p, q) model.
        de (bool, optional): Whether to use differential_evolution as the
            optimizer. Defaults to True.
        debug (bool, optional): Turn on/off debug mode. Defaults to False.
        user_bounds (list, optional): Parameter boundaries for the optimizer.
            Defaults to None.

    Raises:
        Exception: If celerite cannot factorize after 5 trials.

    Returns:
        object: An array of best-fit parameters
    """
    dim = int(p + q + 1)
    best_fit = np.empty(dim)

    # init bounds for fitting
    if user_bounds is not None and (len(user_bounds) == dim):
        bounds = user_bounds
    else:
        bounds = [(-10, 5)] * dim

    # re-position lc
    t = t - t[0]
    y = y - np.median(y)

    # dynamic control of fitting flow
    compute = True  # handle can't factorize in gp.compute()
    compute_ct = 0

    # initialize parameter and kernel
    carma_log_params = carma_log_param_init(dim)
    kernel = CARMA_term(carma_log_params[:p], carma_log_params[p:])
    gp = GP(kernel, mean=np.median(y))

    # compute can't factorize, try 4 more times
    while compute & (compute_ct < 5):
        compute_ct += 1
        try:
            gp.compute(t, yerr)
            compute = False
        except celerite.solver.LinAlgError:
            if compute_ct > 4:
                raise Exception("celerite can't factorize matrix!")
            gp.set_parameter_vector(carma_log_param_init(dim))

    if de:
        best_fit_return = _de_opt(
            y,
            yerr,
            best_fit,
            gp,
            lambda: carma_log_param_init(dim),
            debug,
            bounds,
        )
    else:
        best_fit_return = _min_opt(
            y,
            yerr,
            best_fit,
            gp,
            lambda: carma_log_param_init(dim),
            debug,
            bounds,
        )

    return best_fit_return
