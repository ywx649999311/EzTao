"""
A collection of functions to fit/analyze time series using CARMA models.
"""

import numpy as np
from math import ceil
from scipy.optimize import minimize
import celerite
from celerite import GP
from eztao.carma.CARMATerm import DRW_term, DHO_term, CARMA_term, fcoeffs2coeffs

__all__ = [
    "drw_fit",
    "dho_fit",
    "carma_fit",
    "neg_fcoeff_ll",
    "neg_param_ll",
    "drw_log_param_init",
    "carma_log_param_init",
    "carma_log_fcoeff_init",
    "sample_carma",
]


def neg_fcoeff_ll(fcoeffs, y, gp):
    """
    Negative log likelihood function for CARMA specified in the factored poly space.

    This method will catch 'overflow/underflow' runtimeWarning and
    return inf as probability.

    Args:
        fcoeffs (array(float)): Coefficients of a CARMA model in the factored polynomial 
            space.
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


def drw_log_param_init(std, size=1, max_tau=6.0):
    """
    Randomly generate DRW parameters.

    Args:
        std (float): The standard deviation of the input time series.
        size (int, optional): The number of the set of DRW parameters to generate. 
            Defaults to 1.
        max_tau (float): The maximum likely tau (in natural log). Defaults to 6.0.

    Returns:
        array(float): A ndarray of DRW parameters in natural log.
    """

    init_tau = np.exp(np.random.rand(size, 1) * max_tau)
    init_amp = np.random.rand(size, 1) * 4 * std
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
    """
    Randomly generate a stationary CARMA model given the orders (p and q).

    Args:
        p (int): The p order of a CARMA(p, q) model.
        q (int): The q order of a CARMA(p, q) model.
        ranges (list): Tuple of custom ranges to draw polynomial coefficients from. 
            Defaults to None.

    Returns:
        AR and MA coefficients in two separate arrays.
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
        y (array(float)): An array of y values.
        best_fit (array(float)): An empty array to store best fit parameters.
        gp (object): celerite GP model object.
        init_func (object): CARMA parameter/coefficient initialization function,
            i.e. drw_log_param_init.
        mode (str): Specify which space to sample, 'param' or 'coeff'.
        debug (bool): Turn on/off debug mode.
        bounds (list): CARMA parameter/coefficient boundaries for the optimizer.
        n_iter (int): Number of iterations to run the optimizer. Defaults to 10.
        method (str, optional): scipy.optimize.minimize method. Defaults to "L-BFGS-B".

    Returns:
        array(float): Best-fit CARMA parameters.
    """

    # set the neg_ll function based on mode
    neg_ll = neg_fcoeff_ll if mode == "coeff" else neg_param_ll

    # placeholder for ll and sols; draw init params
    ll, sols, rs = [], [], []
    initial_params = init_func()

    for i in range(n_iter):
        r = minimize(
            neg_ll, initial_params[i], method=method, bounds=bounds, args=(y, gp),
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
    """
    Fit time series to DRW.

    Args:
        t (array(float)): Time stamps of the input time series (the default unit is day).
        y (array(float)): y values of the input time series.
        yerr (array(float)): Measurement errors for y values.
        debug (bool, optional): Turn on/off debug mode. Defaults to False.
        user_bounds (list, optional): Parameter boundaries for the optimizer.
            Defaults to None.
        n_iter (int, optional): Number of iterations to run the optimizer. Defaults to 10.

    Raises:
        celerite.solver.LinAlgError: For non-positive definite autocovariance matrices.

    Returns:
        array(float): Best-fit parameters
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
    """
    Fit time series to DHO/CARMA(2,1).

    Args:
        t (array(float)): Time stamps of the input time series (the default unit is day).
        y (array(float)): y values of the input time series.
        yerr (array(float)): Measurement errors for y values.
        debug (bool, optional): Turn on/off debug mode. Defaults to False.
        user_bounds (list, optional): Parameter boundaries for the optimizer.
            Defaults to None.
        init_ranges (list, optional): Tuples of custom ranges to draw polynomial 
            coefficient proposals from. Defaults to None.
        n_iter (int, optional): Number of iterations to run the optimizer.
            Defaults to 15.

    Raises:
        celerite.solver.LinAlgError: For non-positive definite autocovariance matrices.

    Returns:
        array(float): Best-fit parameters
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
    shift = np.array(0)
    if np.std(y) < 1e-4 or np.std(y) > 1e4:
        shift = np.log(np.std(y))
        bounds[2:] += shift

    # initialize parameter, kernel and GP
    kernel = DHO_term(*carma_log_param_init(2, 1, shift=float(shift)))
    gp = GP(kernel, mean=0)
    gp.compute(t, yerr)

    best_fit_return = _min_opt(
        y,
        best_fit,
        gp,
        lambda: carma_log_param_init(
            2, 1, ranges=init_ranges, size=n_iter, shift=float(shift)
        ),
        "param",
        debug,
        bounds,
        n_iter,
    )

    return best_fit_return


def carma_fit(
    t, y, yerr, p, q, debug=False, user_bounds=None, init_ranges=None, n_iter=15,
):
    """
    Fit time series to an arbitrary CARMA model.

    Args:
        t (array(float)): Time stamps of the input time series (the default unit is day).
        y (array(float)): y values of the input time series.
        yerr (array(float)): Measurement errors for y values.
        p (int): The p order of a CARMA(p, q) model.
        q (int): The q order of a CARMA(p, q) model.
        debug (bool, optional): Turn on/off debug mode. Defaults to False.
        user_bounds (list, optional): Parameter boundaries for the optimizer. If p > 2,
            these are boundaries for the coefficients of the factored polynomial.
            Defaults to None.
        init_ranges (list, optional): Tuples of custom ranges to draw initial
            parameter proposals from. If p > 2, same as for the user_bounds. Defaults to
            None.
        n_iter (int, optional): Number of iterations to run the optimizer if de==False.
            Defaults to 15.

    Raises:
        celerite.solver.LinAlgError: For non-positive definite autocovariance matrices.

    Returns:
        array(float): Best-fit parameters
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
    shift = np.array(0)
    if np.std(y) < 1e-4 or np.std(y) > 1e4:
        shift = np.log(np.std(y))

    # initialize parameter and kernel
    ARpars, MApars = sample_carma(p, q, shift=float(shift))
    kernel = CARMA_term(np.log(ARpars), np.log(MApars))
    gp = GP(kernel, mean=0)
    gp.compute(t, yerr)

    if p > 2:
        mode = "coeff"
        init_func = lambda: carma_log_fcoeff_init(
            p, q, ranges=init_ranges, size=n_iter, shift=float(shift)
        )
        bounds[-1] += shift
    else:
        mode = "param"
        init_func = lambda: carma_log_param_init(
            p, q, ranges=init_ranges, size=n_iter, shift=float(shift)
        )
        bounds[p:] += shift

    best_fit_return = _min_opt(y, best_fit, gp, init_func, mode, debug, bounds, n_iter,)

    return best_fit_return
