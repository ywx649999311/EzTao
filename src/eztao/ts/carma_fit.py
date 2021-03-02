"""
A collection of functions to fit/analyze time series using CARMA models.
"""

import numpy as np
from math import ceil
from scipy.optimize import minimize
import celerite
from celerite import GP
from eztao.carma.CARMATerm import DRW_term, DHO_term, CARMA_term, fcoeffs2coeffs
from functools import partial

__all__ = [
    "drw_fit",
    "dho_fit",
    "carma_fit",
    "flat_prior",
    "neg_fcoeff_ll",
    "neg_param_ll",
    "neg_lp_flat",
    "drw_log_param_init",
    "carma_log_param_init",
    "carma_log_fcoeff_init",
    "sample_carma",
    "scipy_opt",
]


def neg_fcoeff_ll(fcoeffs, y, gp):
    """
    Negative log likelihood function for CARMA specified in the factored poly space.

    This method will catch 'overflow/underflow' runtimeWarning and
    return inf as probability.

    Args:
        fcoeffs (array(float)): Coefficients (in natural log) of a CARMA model in
            the factored polynomial space.
        y (array(float)): y values of the input time series.
        gp (object): celerite GP object with a proper CARMA kernel.

    Returns:
        float: negative log likelihood.
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
    """
    Negative log likelihood function for CARMA specified in the nominal space.

    This method will catch 'overflow/underflow' runtimeWarning and
    return inf as probability.

    Args:
        params (array(float)): CARMA parameters.
        y (array(float)): y values of the input time series.
        gp (object): celerite GP object with a proper CARMA kernel.

    Returns:
        float: negative log likelihood.
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


def flat_prior(params, bounds):
    """
    A flat prior function. Returns 0 if "params" are within the given "bounds",
    negative infiinity otherwise.

    Args:
        params (array(float)): CARMA parameters in natural log.
        bounds (array((float, float)): An array of boundaries.

    Returns:
        float: 0 or negative infinity.
    """

    dim = len(params)
    assert bounds.shape == (dim, 2), "Dimension mismatch for the boundaries!"

    bounds_idx = [i for i in range(dim) if all(bounds[i])]
    in_bounds = np.array(
        [(bounds[i, 0] <= params[i] <= bounds[i, 1]) for i in bounds_idx]
    )

    if all(in_bounds):
        return 0
    else:
        return -np.inf


def neg_lp_flat(params, y, gp, bounds=None, mode="fcoeff"):
    """
    Negative log probability function using a flat prior.

    Args:
        params (array(float)): CARMA parameters in natural log.
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

    return -flat_prior(params, bounds) + neg_ll(params, y, gp)


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

    init_tau = np.exp(
        np.random.rand(size, 1) * (log_tau_range[1] - log_tau_range[0])
        + log_tau_range[0]
    )
    init_amp = np.random.rand(size, 1) * (amp_range[1] - amp_range[0]) + amp_range[0]
    drw_param = np.hstack((init_amp, init_tau))

    if size == 1:
        return drw_param[0]
    else:
        return drw_param


def carma_log_param_init(p, q, ranges=None, size=1, a=-8.0, b=8.0, shift=0):
    """
    Randomly generate CARMA parameters from [a, b) in natural log.

    Args:
        dim (int): For a CARMA(p,q) model, dim=p+q+1.
        ranges (list, optional): Tuples of custom ranges to draw parameter proposals
            from. Defaults to None.
        size (int, optional): The number of the set of CARMA parameters to generate.
            Defaults to 1.
        a (float, optional): The lower bound of the ranges, if the range for a specific
            parameter is not specified. Defaults to -8.0.
        b (float, optional): The upper bound of the ranges, if the range for a specific
            parameter is not specified. Defaults to 8.0.

    Returns:
        array(float): A ndarray of CAMRA parameters in natural log.
    """
    dim = int(p + q + 1)
    log_param = np.random.rand(size, int(dim))

    if (ranges is not None) and (len(ranges) == int(dim)):
        for d in range(dim):
            if all(ranges[d]):
                scale = ranges[d][1] - ranges[d][0]
                log_param[:, d] = log_param[:, d] * scale + ranges[d][0]
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
    """
    Randomly generate CARMA coefficients in the factored polynomial space from [a, b)
    in natural log.

    Args:
        p (int): The p order of a CARMA(p, q) model.
        q (int): The q order of a CARMA(p, q) model.
        ranges (list, optional): Tuples of custom ranges to draw polynomial coefficient
            proposals from. Defaults to None.
        size (int, optional): The number of the set of coefficients to generate.
            Defaults to 1.
        a (float, optional): The lower bound of the ranges, if the range for a specific
            coefficient is not specified. Defaults to -8.0.
        b (float, optional): The upper bound of the ranges, if the range for a specific
            coefficient is not specified. Defaults to 8.0.

    Returns:
        array(float): A ndarray of CAMRA parameters in natural log.
    """
    dim = int(p + q + 1)
    log_coeff = np.random.rand(size, int(dim))

    if (ranges is not None) and (len(ranges) == int(dim)):
        for d in range(dim):
            if all(ranges[d]):
                scale = ranges[d][1] - ranges[d][0]
                log_coeff[:, d] = log_coeff[:, d] * scale + ranges[d][0]
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
    """
    Randomly generate a stationary CARMA model given the orders (p and q).

    Args:
        p (int): The p order of a CARMA(p, q) model.
        q (int): The q order of a CARMA(p, q) model.
        ranges (list): Tuple of custom ranges to draw polynomial coefficients
            from. Defaults to None.

    Returns:
        AR and MA coefficients in two separate arrays.
    """
    init_fcoeffs = np.exp(
        carma_log_fcoeff_init(p, q, ranges=ranges, a=a, b=b, shift=shift)
    )
    ARpars = fcoeffs2coeffs(np.append(init_fcoeffs[:p], [1]))[1:]
    MApars = fcoeffs2coeffs(init_fcoeffs[p:])

    return ARpars, MApars


def scipy_opt(
    y,
    gp,
    init_func,
    neg_lp_func,
    n_iter,
    mode="fcoeff",
    debug=False,
    opt_kwargs={},
    opt_options={},
):
    """
    A wrapper for scipy.optimize.minimze method.

    Args:
        y (array(float)): y values of the input time series.
        gp (object): celerite GP object with a proper CARMA kernel.
        init_func (object): A user-provided function to generate initial
            guesses for the optimizer. Defaults to None.
        neg_lp_func (object): A user-provided function to compute negative
            probability given an array of parameters, an array of time series values and
            a celerite GP instance. Defaults to None.
        n_iter (int): Number of iterations to run the optimizer.
        mode (str, optional): The parameter space in which to make proposals, this
            should be determined in the "_fit" functions based on the value of the p
            order. Defaults to "fcoeff".
        debug (bool, optional): Turn on/off debug mode. Defaults to False.
        opt_kwargs (dict, optional): Keyword arguments for scipy.optimize.minimze.
            Defaults to {}.
        opt_options (dict, optional): "options" argument for scipy.optimize.minimze.
            Defaults to {}.

    Returns:
        Best-fit parameters if "debug" is False, an array of scipy.optimize.OptimizeResult objects otherwise.
    """

    initial_params = init_func(size=n_iter)
    dim = gp.kernel.p + gp.kernel.q + 1

    rs = []
    for i in range(n_iter):
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
    n_iter=10,
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
        n_iter (int, optional): Number of iterations to run the optimizer. Defaults to 10.
        user_bounds (list, optional): Parameter boundaries for the default optimizer.
            Defaults to None.
        scipy_opt_kwargs (dict, optional): Keyword arguments for scipy.optimize.minimze.
            Defaults to {}.
        scipy_opt_options (dict, optional): "options" argument for scipy.optimize.minimze.
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

    best_fit_return = opt(
        y,
        gp,
        init,
        neg_lp,
        n_iter,
    )
    return best_fit_return


def dho_fit(
    t,
    y,
    yerr,
    init_func=None,
    neg_lp_func=None,
    optimizer_func=None,
    n_iter=20,
    user_bounds=None,
    scipy_opt_kwargs={},
    scipy_opt_options={},
    debug=False,
):
    """
    Fit DHO to time series.

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
        n_iter (int, optional): The number of optimizers to initialize. Defaults to 20.
        user_bounds (list, optional): Parameter boundaries for the default optimizer.
            Defaults to None.
        scipy_opt_kwargs (dict, optional): Keyword arguments for scipy.optimize.minimze.
            Defaults to {}.
        scipy_opt_options (dict, optional): "options" argument for scipy.optimize.minimze.
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

    # re-position lc
    t = t - t[0]
    y = y - np.median(y)

    # determine shift due to amplitude being too large/small
    shift = np.array(0)
    if np.std(y) < 1e-3 or np.std(y) > 1e3:
        shift = np.log(np.std(y))
        bounds[2:] += shift

    # determine negative log probability function
    if neg_lp_func is None:
        neg_lp = partial(neg_lp_flat, bounds=np.array(bounds), mode="param")
    else:
        neg_lp = neg_lp_func

    # initialize parameter, kernel and GP
    kernel = DHO_term(*carma_log_param_init(2, 1, shift=float(shift)))
    gp = GP(kernel, mean=0)
    gp.compute(t, yerr)

    # determine initialize function
    if init_func is None:
        init = partial(carma_log_param_init, 2, 1, shift=float(shift))
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

    best_fit_return = opt(
        y,
        gp,
        init,
        neg_lp,
        n_iter,
    )

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
    n_iter=20,
    user_bounds=None,
    scipy_opt_kwargs={},
    scipy_opt_options={},
    debug=False,
):
    """
    Fit an arbitrary CARMA model.

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
        n_iter (int, optional): Number of iterations to run the optimizer if de==False.
            Defaults to 20.
        user_bounds (array(float), optional): Parameter boundaries for the default
            optimizer. If p > 2, these are boundaries for the coefficients of the
            factored polynomial. Defaults to None.
        scipy_opt_kwargs (dict, optional): Keyword arguments for scipy.optimize.minimze.
            Defaults to {}.
        scipy_opt_options (dict, optional): "options" argument for scipy.optimize.minimze.
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

    # re-position lc
    t = t - t[0]
    y = y - np.median(y)

    # determine/set shift due amp too large/small
    shift = np.array(0)
    if np.std(y) < 1e-4 or np.std(y) > 1e4:
        shift = np.log(np.std(y))

    # initialize parameter and kernel
    ARpars, MApars = sample_carma(p, q, shift=float(shift))
    kernel = CARMA_term(np.log(ARpars), np.log(MApars))
    gp = GP(kernel, mean=0)
    gp.compute(t, yerr)

    # determine/set init func
    if init_func is not None:
        init = init_func
    elif mode == "fcoeff":
        init = partial(carma_log_fcoeff_init, p, q, shift=float(shift))
        bounds[-1] += shift
    else:
        init = partial(carma_log_param_init, p, q, shift=float(shift))
        bounds[p:] += shift

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

    best_fit_return = opt(
        y,
        gp,
        init,
        neg_lp,
        n_iter,
    )

    return best_fit_return