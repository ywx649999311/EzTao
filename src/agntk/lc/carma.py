"""Functions for performing CARMA analysis on LCs.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from celerite import GP
import celerite
from agntk.carma.CARMATerm import *
from agntk.lc.utils import *


def gpSimFull(carmaTerm, SNR, duration, N, nLC=1):
    """Simulate full CARMA time series.

    Args:
        carmaTerm (object): celerite GP term.
        SNR (float): Signal to noise ratio defined as ratio between 
            CARMA amplitude and the standard deviation of the errors.
        duration (float): The duration of the simulated time series in days.
        N (int): The number of data points.
        nLC (int, optional): Number of light curves to simulate. Defaults to 1.

    Raises:
        Exception: If celerite cannot factorize after 10 trials.

    Returns:
        Arrays: t, y and yerr of the simulated light curves in numpy arrays.
    """

    assert isinstance(
        carmaTerm, celerite.celerite.terms.Term
    ), "carmaTerm must a celerite GP term"

    gp_sim = GP(carmaTerm)

    t = np.linspace(0, duration, N)
    yerr = np.random.normal(0, carmaTerm.get_rms_amp() / SNR, N)

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
            t = np.linspace(0, duration, N)
            yerr_reg += 1.123e-12

            fact_num += 1
            if fact_num > 10:
                raise Exception(
                    "Celerite cannot factorize the GP"
                    + " covairance matrix, try again!"
                )

    t = np.repeat(t[None, :], nLC, axis=0)
    yerr = np.repeat(yerr[None, :], nLC, axis=0)
    y = gp_sim.sample(size=nLC) + yerr

    return t, y, yerr


def gpSimRand(carmaTerm, SNR, duration, N, nLC=1, season=True, full_N=10_000):
    """Simulate downsampled CARMA time series.

    Args:
        carmaTerm (object): celerite GP term.
        SNR (float): Signal to noise ratio defined as ratio between 
            CARMA amplitude and the standard deviation of the errors.
        duration (float): The duration of the simulated time series in days.
        N (int): The number of data points in the returned light curves.
        nLC (int, optional): Number of light curves to simulate. Defaults to 1.
        season (bool, optional): Whether to simulate seasonal gaps. 
            Defaults to True.
        full_N (int, optional): The number of data points the full light curves. 
            Defaults to 10_000.

    Returns:
        Arrays: t, y and yerr of the simulated light curves in numpy arrays.
    """

    assert isinstance(
        carmaTerm, celerite.celerite.terms.Term
    ), "carmaTerm must a celerite GP term"

    t, y, yerr = gpSimFull(carmaTerm, SNR, duration, full_N, nLC=nLC)

    # output t & yerr
    t_out = np.empty((nLC, N))
    y_out = np.empty((nLC, N))
    yerr_out = np.empty((nLC, N))

    # downsample
    for i in range(nLC):
        mask1 = add_season(t[i])
        mask2 = downsample_byN(t[i, mask1], N)
        t_out[i, :] = t[i, mask1][mask2]
        y_out[i, :] = y[i, mask1][mask2]
        yerr_out[i, :] = yerr[i, mask1][mask2]

    return t_out, y_out, yerr_out


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

    params = np.array(params)
    dim = len(params)
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

    return [log_a1, log_a2, log_b0, log_b1]


def carma_log_param_init(dim):
    """Randomly generate DHO parameters from [-8, 1] in log.
    
    Args:
        dim (int): For a CARMA(p,q) model, dim=p+q+1.
    Returns:
        list: The generated CAMRA parameters in natural log.
    """

    log_param = np.random.uniform(-8, 1, dim)

    return log_param


def drw_fit(t, y, yerr, de=True, debug=False, plot=False, user_bounds=None):
    """Fix time series to DRW model

    Args:
        t (object): An array of time stamps in days.
        y (object): An array of y values.
        yerr (object): An array of the errors in y values.
        de (bool, optional): Whether to use differential_evolution as the 
            optimizer. Defaults to True. 
        debug (bool, optional): Turn on/off debug mode. Defaults to False.
        plot (bool, optional): Whether plot likelihood surface. 
            Defaults to False.
        user_bounds (list, optional): Parameter boundaries for the optimizer. 
            Defaults to None.

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
    rerun = True
    compute = True  # handle can't factorize in gp.compute()
    succeded = False  # ever succeded
    compute_ct = 0
    counter = 0
    jac_log_rec = 10

    # initialize parameter and kernel
    kernel = DRW_term(*drw_log_param_init(std))
    gp = GP(kernel, mean=np.median(y))
    gp.compute(t, yerr)

    # compute can't factorize, try 4 more times
    while compute & (compute_ct < 5):
        compute_ct += 1
        try:
            gp.compute(t, yerr)
            compute = False
        except celerite.solver.LinAlgError:
            gp.set_parameter_vector(drw_log_param_init(std))

    if de:
        # set bound based on LC std for amp
        while rerun and (counter < 5):
            counter += 1
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
                gp.set_parameter_vector(drw_log_param_init(std))

    else:
        initial_params = gp.get_parameter_vector()

        while rerun and (counter < 5):
            counter += 1
            r = minimize(
                neg_ll,
                initial_params,
                method="L-BFGS-B",
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
                gp.set_parameter_vector(drw_log_param_init(std))

    # If opitimizer never reached minima, assign nan
    if not succeded:
        best_fit[:] = np.nan

    # Below code is used to visualize if stuck in local minima
    if debug:
        print(r)

    return best_fit


def dho_fit(t, y, yerr, debug=False, plot=False, user_bounds=None):
    """Fix time series to DHO model

    Args:
        t (object): An array of time stamps in days.
        y (object): An array of y values.
        yerr (object): An array of the errors in y values.
        de (bool, optional): Whether to use differential_evolution as the 
            optimizer. Defaults to True. 
        debug (bool, optional): Turn on/off debug mode. Defaults to False.
        plot (bool, optional): Whether plot likelihood surface. 
            Defaults to False.
        user_bounds (list, optional): Parameter boundaries for the optimizer. 
            Defaults to None.

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
    rerun = True
    compute = True  # handle can't factorize in gp.compute()
    succeded = False  # ever succeded
    compute_ct = 0
    counter = 0
    jac_log_rec = 10

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
            gp.set_parameter_vector(dho_log_param_init())

    # set bound based on LC std for amp
    while rerun and (counter < 5):
        counter += 1
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
            gp.set_parameter_vector(dho_log_param_init())

    # If opitimizer never reached minima, assign nan
    if not succeded:
        best_fit[:] = np.nan

    # Below code is used to visualize if stuck in local minima
    if debug:
        print(r)

    return best_fit


def carma_fit(t, y, yerr, p, q, de=True, debug=False, plot=False, user_bounds=None):
    """Fix time series to all CARMA model

    Args:
        t (object): An array of time stamps in days.
        y (object): An array of y values.
        yerr (object): An array of the errors in y values.
        p (int): P order of a CARMA(p, q) model.
        q (int): Q order of a CARMA(p, q) model.
        de (bool, optional): Whether to use differential_evolution as the 
            optimizer. Defaults to True. 
        debug (bool, optional): Turn on/off debug mode. Defaults to False.
        plot (bool, optional): Whether plot likelihood surface. 
            Defaults to False.
        user_bounds (list, optional): Parameter boundaries for the optimizer. 
            Defaults to None.

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
    rerun = True
    compute = True  # handle can't factorize in gp.compute()
    succeded = False  # ever succeded
    compute_ct = 0
    counter = 0
    jac_log_rec = 10

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
            gp.set_parameter_vector(carma_log_param_init(dim))

    if de:
        # set bound based on LC std for amp
        while rerun and (counter < 5):
            counter += 1
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
                gp.set_parameter_vector(carma_log_param_init(dim))

    else:
        initial_params = gp.get_parameter_vector()

        while rerun and (counter < 5):
            counter += 1
            r = minimize(
                neg_ll,
                initial_params,
                method="L-BFGS-B",
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
                gp.set_parameter_vector(carma_log_param_init(dim))

    # If opitimizer never reached minima, assign nan
    if not succeded:
        best_fit[:] = np.nan

    # Below code is used to visualize if stuck in local minima
    if debug:
        print(r)

    return best_fit
