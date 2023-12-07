"""
A collection of functions to fit/analyze time series using CARMA models.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation as mad
import celerite
from celerite import GP
from eztao.carma.CARMATerm import DRW_term, DHO_term, CARMA_term
from functools import partial

# change numpy warn setting
old_settings = np.seterr(all="warn")
np.seterr(invalid="ignore")

__all__ = [
    "drw_fit",
    "dho_fit",
    "carma_fit",
    "flat_prior",
    "neg_fcoeff_ll",
    "neg_param_ll",
    "neg_lp_flat",
    "drw_log_param_init",
    "dho_log_param_init",
    "carma_log_fcoeff_init",
    "sample_carma",
    "scipy_opt",
]


def neg_fcoeff_ll(log_fcoeffs, y, gp):
    """
    Negative log likelihood function for CARMA specified in the factored poly space.

    This method will catch 'overflow/underflow' runtimeWarning and
    return inf as probability.

    Args:
        log_fcoeffs (array(float)): Coefficients (in natural log) of a CARMA model in
            the factored polynomial space.
        y (array(float)): y values of the input time series.
        gp (object): celerite GP object with a proper CARMA kernel.

    Returns:
        float: Negative log likelihood.
    """

    assert gp.kernel.p >= 2, "Use neg_param_ll() instead!"
    neg_ll = np.inf

    try:
        gp.kernel.set_log_fcoeffs(log_fcoeffs)
        neg_ll = -gp.log_likelihood(y)
    except celerite.solver.LinAlgError:
        # print(c)
        pass
    except Exception:
        pass

    return neg_ll


def neg_param_ll(log_params, y, gp):
    """
    Negative log likelihood function for CARMA specified in the nominal space.

    This method will catch 'overflow/underflow' runtimeWarning and
    return inf as probability.

    Args:
        log_params (array(float)): Natural log of CARMA parameters.
        y (array(float)): y values of the input time series.
        gp (object): celerite GP object with a proper CARMA kernel.

    Returns:
        float: Negative log likelihood.
    """
    assert gp.kernel.p <= 2, "Use neg_fcoeff_ll() instead!"
    neg_ll = np.inf

    try:
        gp.set_parameter_vector(log_params)
        neg_ll = -gp.log_likelihood(y)
    except celerite.solver.LinAlgError:
        # print(c)
        pass
    except Exception:
        pass

    return neg_ll


def flat_prior(log_params, bounds):
    """
    A flat prior function. Returns 0 if "log_params" are within the given "bounds",
    negative infinity otherwise.

    Args:
        log_params (array(float)): CARMA parameters in natural log.
        bounds (array((float, float)): An array of boundaries.

    Returns:
        float: 0 or negative infinity.
    """

    dim = len(log_params)
    assert bounds.shape == (dim, 2), "Dimension mismatch for the boundaries!"

    bounds_idx = [i for i in range(dim) if all(bounds[i])]
    in_bounds = np.array(
        [(bounds[i, 0] <= log_params[i] <= bounds[i, 1]) for i in bounds_idx]
    )

    if all(in_bounds):
        return 0
    else:
        return -np.inf


def neg_lp_flat(log_params, y, gp, bounds=None, mode="fcoeff"):
    """
    Negative log probability function using a flat prior.

    Args:
        log_params (array(float)): CARMA parameters (or coefficients of the factored
            characteristic polynomial) in natural log.
        y (array(float)): y values of the input time series.
        gp (object): celerite GP object with a proper CARMA kernel.
        bounds (array((float, float)): An array of boundaries. Defaults to None.
        mode (str, optional): The parameter space in which proposals are made. The mode
            determines which loglikehood function to use. Defaults to "fcoeff".

    Returns:
        float: Log probability of the proposed parameters.
    """

    if mode == "param":
        neg_ll = neg_param_ll
    else:
        neg_ll = neg_fcoeff_ll

    return -flat_prior(log_params, bounds) + neg_ll(log_params, y, gp)


def drw_log_param_init(amp_range, log_tau_range, size=1):
    """
    Randomly generate DRW parameters.

    Args:
        amp_range (object): An array containing the range of DRW amplitude to simulate.
        log_tau_range (object): An array containing the range of DRW timescale
            (in natural log) to simulate.
        size (int, optional): The number of the set of DRW parameters to generate.
            Defaults to 1.

    Returns:
        array(float): A ndarray of DRW parameters in natural log.
    """

    # validate inputs
    assert (
        len(amp_range) == len(log_tau_range) == 2
    ), "Dimension mismatch: Check the input ranges!"
    assert (np.array(amp_range) > 0).all(), "Amplitudes must be positive!"

    amp_scale = amp_range[1] - amp_range[0]
    log_tau_scale = log_tau_range[1] - log_tau_range[0]
    assert amp_scale > 0, "Lower bound must be smaller than upper bound!"
    assert log_tau_scale > 0, "Lower bound must be smaller than upper bound!"

    # draw samples
    init_tau = np.exp(np.random.rand(size, 1) * log_tau_scale + log_tau_range[0])
    init_amp = np.random.rand(size, 1) * amp_scale + amp_range[0]
    log_drw_param = np.log(np.hstack((init_amp, init_tau)))

    if size == 1:
        return log_drw_param[0]
    else:
        return log_drw_param


def dho_log_param_init(ar_range=[-6, 10], ma_range=[-10, 2], size=1):
    """
    Randomly generate DHO coefficients in the space of the factored polynomials

    The default ranges are optimized for normalized light curves (with a standard
    deviation of unity).

    Args:
        ar_range(object, optional): The range (in natural log) for DHO AR parameters.
            Defaults to [-6, 10].
        ma_range(object, optional): The range (in natural log) for DHO MA parameters.
            Defaults to [-10, 2].
        size (int, optional): The number of the set of DHO parameters to generate.
            Defaults to 1.

    Returns:
        array(float): A ndarray of DHO parameters in natural log.
    """

    ## validate input
    assert (
        len(ar_range) == len(ma_range) == 2
    ), "Dimension mismatch: Check the input ranges!"

    ar_scale = ar_range[1] - ar_range[0]
    ma_scale = ma_range[1] - ma_range[0]
    assert ar_scale > 0, "Lower bound must be smaller than upper bound!"
    assert ma_scale > 0, "Lower bound must be smaller than upper bound!"

    ## draw samples
    p, q = (2, 1)
    log_params = np.random.rand(size, 4)

    # adjust ranges
    log_params[:, :p] = log_params[:, :p] * ar_scale + ar_range[0]
    log_params[:, p:-1] = log_params[:, p:-1] * ma_scale + ma_range[0]

    if size == 1:
        return log_params[0]
    else:
        return log_params


def carma_log_fcoeff_init(
    p, q, ar_range=[-8, 8], ma_range=[-10, 6], ma_mult_range=[-10, 0], size=1
):
    """
    Randomly generate CARMA coefficients in the space of the factored polynomials

    The default ranges are optimized for normalized light curves (with a standard
    deviation of unity).

    Args:
        p (int): The p order of a CARMA(p, q) model.
        q (int): The q order of a CARMA(p, q) model.
        ar_range(object, optional): The range (in natural log) for AR polynomial
            coefficients. Defaults to [-8, 8].
        ma_range(object, optional): The range (in natural log) for MA polynomial
            coefficients. Defaults to [-10, 6].
        ma_mult_range(object, optional): The range for the MA multiplier (the coefficient
            of the highest-order term in the MA characteristic polynomial). Defaults to
            [-10, 0].
        size (int, optional): The number of the set of coefficients to generate.
            Defaults to 1.

    Returns:
        array(float): A ndarray of coeffs for the factored polynomials in natural log.

    .. note:: The notation (index) in the returned coefficients follows that in
        Jones et al. (1981). The last coefficient in the returned array is not part of
        the coefficients, rather a simple multiplying factor of the entire polynomial,
        which is needed to obtain the nominal CARMA representation.
    """
    ## validate inputs
    assert (
        len(ar_range) == len(ma_range) == len(ma_mult_range) == 2
    ), "Dimension mismatch: Check the input ranges!"

    ar_scale = ar_range[1] - ar_range[0]
    ma_scale = ma_range[1] - ma_range[0]
    ma_mult_scale = ma_mult_range[1] - ma_mult_range[0]
    assert ar_scale > 0, "Lower bound must be smaller than upper bound!"
    assert ma_scale > 0, "Lower bound must be smaller than upper bound!"
    assert ma_mult_scale > 0, "Lower bound must be smaller than upper bound!"

    ## draw samples
    dim = int(p + q + 1)
    log_fcoeffs = np.random.rand(size, int(dim))

    # set AR, MA range
    log_fcoeffs[:, :p] = log_fcoeffs[:, :p] * ar_scale + ar_range[0]
    log_fcoeffs[:, p:-1] = log_fcoeffs[:, p:-1] * ma_scale + ma_range[0]

    # set ma_mult range
    log_fcoeffs[:, -1] = log_fcoeffs[:, -1] * ma_mult_scale + ma_mult_range[0]

    if size == 1:
        return log_fcoeffs[0]
    else:
        return log_fcoeffs


def sample_carma(p, q):
    """
    Randomly generate a stationary CARMA model given the orders (p and q).

    Args:
        p (int): The p order of a CARMA(p, q) model.
        q (int): The q order of a CARMA(p, q) model.

    Returns:
        AR and MA coefficients in two separate arrays.
    """
    init_log_fcoeffs = carma_log_fcoeff_init(p, q)
    logAR, logMA = CARMA_term.fcoeffs2carma_log(init_log_fcoeffs, p)
    return np.exp(logAR), np.exp(logMA)


def scipy_opt(
    y,
    gp,
    init_func,
    neg_lp_func,
    n_opt,
    mode="fcoeff",
    debug=False,
    opt_kwargs={},
    opt_options={},
):
    """
    A wrapper for scipy.optimize.minimize method.

    Args:
        y (array(float)): y values of the input time series.
        gp (object): celerite GP object with a proper CARMA kernel.
        init_func (object): A user-provided function to generate initial
            guesses for the optimizer. Defaults to None.
        neg_lp_func (object): A user-provided function to compute negative
            probability given an array of parameters, an array of time series values and
            a celerite GP instance. Defaults to None.
        n_opt (int): Number of iterations to run the optimizer.
        mode (str, optional): The parameter space in which to make proposals, this
            should be determined in the "_fit" functions based on the value of the p
            order. Defaults to "fcoeff".
        debug (bool, optional): Turn on/off debug mode. Defaults to False.
        opt_kwargs (dict, optional): Keyword arguments for scipy.optimize.minimize.
            Defaults to {}.
        opt_options (dict, optional): "options" argument for scipy.optimize.minimize.
            Defaults to {}.

    Returns:
        Best-fit parameters if "debug" is False, an array of scipy.optimize.OptimizeResult objects otherwise.
    """

    initial_params = init_func(size=n_opt)
    dim = gp.kernel.p + gp.kernel.q + 1

    rs = []
    for i in range(n_opt):
        r = minimize(
            neg_lp_func,
            initial_params[i],
            args=(y, gp),
            **opt_kwargs,
            options=opt_options,
        )

        rs.append(r)

    if debug:
        return rs
    else:
        good_rs = [r for r in rs if r.success and r.fun != -np.inf]

        if len(good_rs) == 0:
            return [np.nan] * dim
        else:
            lls = [-r.fun for r in good_rs]
            log_sols = [r.x for r in good_rs]
            best_sol = log_sols[np.argmax(lls)]

            if mode == "fcoeff":
                return np.concatenate(CARMA_term.fcoeffs2carma(best_sol, gp.kernel.p))
            else:
                return np.exp(best_sol)


def drw_fit(
    t,
    y,
    yerr,
    init_func=None,
    neg_lp_func=None,
    optimizer_func=None,
    n_opt=10,
    user_bounds=None,
    scipy_opt_kwargs={},
    scipy_opt_options={},
    debug=False,
):
    """
    Fit DRW.

    Args:
        t (array(float)): Time stamps of the input time series (the default unit is day).
        y (array(float)): y values of the input time series.
        yerr (array(float)): Measurement errors for y values.
        init_func (object, optional): A user-provided function to generate initial
            guesses for the optimizer. Defaults to None.
        neg_lp_func (object, optional): A user-provided function to compute negative
            probability given an array of parameters, an array of time series values and
            a celerite GP instance. Defaults to None.
        optimizer_func (object, optional): A user-provided optimizer function.
            Defaults to None.
        n_opt (int, optional): Number of optimizers to run. Defaults to 10.
        user_bounds (list, optional): Parameter boundaries for the default optimizer.
            Defaults to None.
        scipy_opt_kwargs (dict, optional): Keyword arguments for scipy.optimize.minimize.
            Defaults to {}.
        scipy_opt_options (dict, optional): "options" argument for scipy.optimize.minimize.
            Defaults to {}.
        debug (bool, optional): Turn on/off debug mode. Defaults to False.

    Returns:
        array(float): Best-fit DRW parameters
    """
    # re-position lc; compute some stat
    t = t - t[0]
    y = y - np.median(y)
    std = np.std(y)
    min_dt = np.min(t[1:] - t[:-1])

    # init bounds for fitting
    if user_bounds is not None and (len(user_bounds) == 2):
        bounds = user_bounds
    else:
        bounds = [
            (np.log(std / 50), np.log(3 * std)),
            (np.log(min_dt / 5), np.log(t[-1])),
        ]

    # determine negative log probability function
    if neg_lp_func is None:
        neg_lp = partial(neg_lp_flat, bounds=np.array(bounds), mode="param")
    else:
        neg_lp = neg_lp_func

    # define ranges to generate initial guesses
    amp_range = [std / 50, 3 * std]
    log_tau_range = [np.log(min_dt / 5), np.log(t[-1] / 10)]

    # initialize parameter and kernel
    kernel = DRW_term(*drw_log_param_init(amp_range, log_tau_range))
    gp = GP(kernel, mean=0)
    gp.compute(t, yerr)

    # determine initialize function
    if init_func is None:
        init = partial(drw_log_param_init, amp_range, log_tau_range)
    else:
        init = init_func

    # determine optimizer function
    if optimizer_func is None:
        scipy_opt_kwargs.update({"method": "L-BFGS-B", "bounds": bounds})
        opt = partial(
            scipy_opt,
            mode="param",
            opt_kwargs=scipy_opt_kwargs,
            opt_options=scipy_opt_options,
            debug=debug,
        )
    else:
        opt = optimizer_func

    best_fit_return = opt(y, gp, init, neg_lp, n_opt)

    return best_fit_return


def dho_fit(
    t,
    y,
    yerr,
    init_func=None,
    neg_lp_func=None,
    optimizer_func=None,
    n_opt=20,
    user_bounds=None,
    scipy_opt_kwargs={},
    scipy_opt_options={},
    debug=False,
):
    """
    Fit DHO to time series

    The default settings are optimized for normalized LCs.

    Args:
        t (array(float)): Time stamps of the input time series (the default unit is day).
        y (array(float)): y values of the input time series.
        yerr (array(float)): Measurement errors for y values.
        init_func (object, optional): A user-provided function to generate initial
            guesses for the optimizer. Defaults to None.
        neg_lp_func (object, optional): A user-provided function to compute negative
            probability given an array of parameters, an array of time series values and
            a celerite GP instance. Defaults to None.
        optimizer_func (object, optional): A user-provided optimizer function.
            Defaults to None.
        n_opt (int, optional): Number of optimizers to run.. Defaults to 20.
        user_bounds (list, optional): Parameter boundaries for the default optimizer and
            the default flat prior. Defaults to None.
        scipy_opt_kwargs (dict, optional): Keyword arguments for scipy.optimize.minimize.
            Defaults to {}.
        scipy_opt_options (dict, optional): "options" argument for scipy.optimize.minimize.
            Defaults to {}.
        debug (bool, optional): Turn on/off debug mode. Defaults to False.

    Raises:
        celerite.solver.LinAlgError: For non-positive definite autocovariance matrices.

    Returns:
        array(float): Best-fit DHO parameters
    """

    # determine user defined boundaries if any
    if user_bounds is not None and (len(user_bounds) == 4):
        bounds = user_bounds
    else:
        bounds = [(-15, 15)] * 4
        bounds[2:] = [(a[0] - 8, a[1] - 8) for a in bounds[2:]]

    # re-position/normalize lc
    t = t - t[0]
    y = y - np.median(y)
    y_std = mad(y) * 1.4826
    y = y / y_std
    yerr = yerr / y_std

    # determine negative log probability function
    if neg_lp_func is None:
        neg_lp = partial(neg_lp_flat, bounds=np.array(bounds), mode="param")
    else:
        neg_lp = neg_lp_func

    # initialize parameter, kernel and GP
    kernel = DHO_term(*dho_log_param_init())
    gp = GP(kernel, mean=0)
    gp.compute(t, yerr)

    # determine initialize function
    if init_func is None:
        init = partial(dho_log_param_init)
    else:
        init = init_func

    # determine the optimizer function
    if optimizer_func is None:
        scipy_opt_kwargs.update({"method": "L-BFGS-B", "bounds": bounds})
        opt = partial(
            scipy_opt,
            mode="param",
            opt_kwargs=scipy_opt_kwargs,
            opt_options=scipy_opt_options,
            debug=debug,
        )
    else:
        opt = optimizer_func

    # get best-fit solution & adjust MA params (multiply by y_std)
    # if not fit found, return all nan
    best_fit_return = opt(y, gp, init, neg_lp, n_opt)
    try:
        best_fit_return[2:] = best_fit_return[2:] * y_std
    except:
        pass

    return best_fit_return


def carma_fit(
    t,
    y,
    yerr,
    p,
    q,
    init_func=None,
    neg_lp_func=None,
    optimizer_func=None,
    n_opt=20,
    user_bounds=None,
    scipy_opt_kwargs={},
    scipy_opt_options={},
    debug=False,
):
    """
    Fit an arbitrary CARMA model

    The default settings are optimized for normalized LCs.

    Args:
        t (array(float)): Time stamps of the input time series (the default unit is day).
        y (array(float)): y values of the input time series.
        yerr (array(float)): Measurement errors for y values.
        p (int): The p order of a CARMA(p, q) model.
        q (int): The q order of a CARMA(p, q) model.
        init_func (object, optional): A user-provided function to generate initial
            guesses for the optimizer. Defaults to None.
        neg_lp_func (object, optional): A user-provided function to compute negative
            probability given an array of parameters, an array of time series values and
            a celerite GP instance. Defaults to None.
        optimizer_func (object, optional): A user-provided optimizer function.
            Defaults to None.
        n_opt (int, optional): Number of optimizers to run.
            Defaults to 20.
        user_bounds (array(float), optional): Parameter boundaries for the default
            optimizer. If p > 2, these are boundaries for the coefficients of the
            factored polynomial. Defaults to None.
        scipy_opt_kwargs (dict, optional): Keyword arguments for scipy.optimize.minimize.
            Defaults to {}.
        scipy_opt_options (dict, optional): "options" argument for scipy.optimize.minimize.
            Defaults to {}.
        debug (bool, optional): Turn on/off debug mode. Defaults to False.

    Raises:
        celerite.solver.LinAlgError: For non-positive definite autocovariance matrices.

    Returns:
        array(float): Best-fit parameters
    """
    # set core config
    dim = int(p + q + 1)
    mode = "fcoeff" if p > 2 else "param"

    # init bounds for fitting
    if user_bounds is not None and (len(user_bounds) == dim):
        bounds = user_bounds
    else:
        bounds = [(-15, 15)] * dim
        bounds[p:-1] = [(a[0] - 5, a[1] - 5) for a in bounds[p:-1]]
        bounds[-1] = (-15, 5)

    # re-position lc
    t = t - t[0]
    y = y - np.median(y)
    y_std = mad(y) * 1.4826
    y = y / y_std
    yerr = yerr / y_std

    # initialize parameter and kernel
    ARpars, MApars = sample_carma(p, q)
    kernel = CARMA_term(np.log(ARpars), np.log(MApars))
    gp = GP(kernel, mean=0)
    gp.compute(t, yerr)

    # determine/set init func
    if init_func is not None:
        init = init_func
    else:
        init = partial(carma_log_fcoeff_init, p, q)

    # determine/set negative log probability function
    if neg_lp_func is None:
        neg_lp = partial(neg_lp_flat, bounds=np.array(bounds), mode=mode)
    else:
        neg_lp = neg_lp_func

    # determine/set optimizer function
    if optimizer_func is None:
        scipy_opt_kwargs.update({"method": "L-BFGS-B", "bounds": bounds})
        opt = partial(
            scipy_opt,
            mode=mode,
            opt_kwargs=scipy_opt_kwargs,
            opt_options=scipy_opt_options,
            debug=debug,
        )
    else:
        opt = optimizer_func

    # get best-fit solution & adjust MA params (multiply by y_std)
    # if not fit found, return all nan
    best_fit_return = opt(y, gp, init, neg_lp, n_opt)
    try:
        best_fit_return[p:] = best_fit_return[p:] * y_std
    except:
        pass

    return best_fit_return
