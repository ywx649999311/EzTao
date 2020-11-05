import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import celerite
from ..carma.CARMATerm import *

mpl.rc_file("eztao.rc")


def plot_drw_ll(
    t,
    y,
    yerr,
    best_params,
    gp,
    prob_func,
    amp_range=None,
    tau_range=None,
    nLevels=20,
    **kwargs,
):
    """
    Plot DRW likelihood surface

    Args:
        t (array-like): An array of time points
        y (array-like): An array of fluxes at the abvoe time points
        yerr (array-like): An array of photometric errors
        best_params (array-like): Best-fit parameters in [amp, tau]
        gp (object): A DRW celerite GP object
        prob_func (func): Posterior/Likelihood function with 
            args=(params, y, yerr, gp)
        amp_range (array-like, optional): The range of parameters to eval 
            probability. Defaults to None.
        tau_range (array-like, optional): The range of parameters to eval 
            probability. Defaults to None.
        nLevels (int, optional): Contour plot number of levels. Defaults to 20.
    
    Kwargs:
        grid_size (int): The number of points to eval likelihood along a 
            given axis.
        true_params (array-like): The true parameters if the given light curve 
            is simulated.
    """
    best_amp, best_tau = best_params
    grid_size = 50

    if amp_range is None:
        amp_range = [0.5 * best_amp, 1.5 * best_amp]
    if tau_range is None:
        tau_range = [0.5 * best_tau, 1.5 * best_tau]

    # check if custom resoultion
    if "grid_size" in kwargs:
        grid_size = int(kwargs.get("grid_size"))

    taus = np.log(
        np.logspace(np.log10(tau_range[0]), np.log10(tau_range[1]), grid_size)
    )
    amps = np.log(np.linspace(amp_range[0], amp_range[1], grid_size))

    param_grid = np.meshgrid(amps, taus)  # log scale
    reg_param_grid = np.exp(param_grid)  # normal scale

    # flatten to list
    ls_params_combo = list(zip(param_grid[0].flatten(), param_grid[1].flatten()))

    # run in bulk
    gp.compute(t, yerr)
    vec_neg_ll = np.vectorize(prob_func, excluded=[1, 2, 3], signature="(n)->()")
    bulk_lp = np.negative(vec_neg_ll(ls_params_combo, y, yerr, gp))
    lp_reshape = bulk_lp.reshape((grid_size, grid_size))

    # normalize, better contours
    max_ll, min_ll = np.max(bulk_lp), np.min(bulk_lp)
    log_levels = (
        1
        + max_ll
        - np.sort(np.logspace(0, np.log(max_ll - min_ll), nLevels, base=np.e))[::-1]
    )
    norm = mpl.colors.SymLogNorm(
        linthresh=np.abs(max_ll), linscale=1, vmin=min_ll, vmax=max_ll + 1, base=np.e
    )
    fig = plt.figure(figsize=(8, 5), dpi=100)
    _ = plt.contourf(
        reg_param_grid[0],
        reg_param_grid[1],
        lp_reshape,
        log_levels,
        norm=norm,
        extend="both",
    )
    plt.title("Likelihood Surface")
    plt.xlabel("Amp")
    plt.ylabel("Tau")
    plt.scatter(best_params[0], best_params[1], marker="*", s=200, label="Best-fit")

    if "true_params" in kwargs:
        true_params = kwargs.get("true_params")
        plt.scatter(true_params[0], true_params[1], marker="*", s=200, label="True")

    plt.colorbar(_, format="%.0f")
    plt.legend()


def plot_dho_ll(
    t,
    y,
    yerr,
    best_params,
    gp,
    prob_func,
    inner_dim=10,
    outer_dim=4,
    ranges=[(None, None), (None, None), (None, None), (None, None)],
    nLevels=20,
    **kwargs,
):
    """
    Plot DHO likelihood surface

    Args:
        t (array-like): An array of time points
        y (array-like): An array of fluxes at the abvoe time points
        yerr (array-like): An array of photometric errors
        best_params (array-like): Best-fit parameters in [a1, a2, b0, b1]
        gp (object): A DRW celerite GP object
        prob_func (func): Posterior/Likelihood function with 
            args=(params, y, yerr, gp)
        inner_dim (int, optional): The number of points to eval likelihood along 
            a1 and a2. Defaults to 10.
        outer_dim (int, optional): The number of points to eval likelihood along 
            b0 and b1. Defaults to 4.
        ranges (list, optional): Parameters range (log) within which to plot 
            the surface. Defaults to [(None, None), (None, None), (None, None), 
            (None, None)].
        nLevels (int, optional): Contour plot number of levels. Defaults to 20.
   
    Kwargs:
        true_params (array-like): The true parameters in log scale if the 
            given light curve is simulated.
    """
    best_log = np.log(best_params)
    num_param = len(best_log)
    param_range = np.zeros((num_param, 2))

    # check if any ranges provided
    for i in range(num_param):
        if any(map(lambda x: x is None, ranges[i])):
            param_range[i, :] = np.array([best_log[i] - 1, best_log[i] + 1])
        else:
            param_range[i, :] = np.array(ranges[i])

    # create meshgrid to compute ll
    inner_grid_params = [0, 1]
    outer_grid_params = [2, 3]
    grid_ls = []

    for i in range(num_param):
        pm_range = param_range[i]

        # check if param is in inner grid
        if i in inner_grid_params:
            grid_ls.append(np.linspace(pm_range[0], pm_range[1], inner_dim))
        else:
            wide_grid = np.linspace(pm_range[0], pm_range[1], outer_dim + 1)
            delta = wide_grid[1] - wide_grid[0]
            grid_ls.append(wide_grid[1:] - delta / 2)

    param_grid = np.meshgrid(*grid_ls)  # log scale

    # prepare for np.vectorize
    vec_neg_ll = np.vectorize(prob_func, excluded=[1, 2, 3], signature="(n)->()")
    param_combos = zip(
        param_grid[0].flatten(),
        param_grid[1].flatten(),
        param_grid[2].flatten(),
        param_grid[3].flatten(),
    )
    ls_param_combos = list(param_combos)

    # run in bulk
    gp.compute(t, yerr)
    bulk_ll = np.negative(vec_neg_ll(ls_param_combos, y, yerr, gp))

    # back to shape (dim, dim, dim, dim)
    dims = np.empty(num_param, dtype=np.int)
    dims[inner_grid_params] = inner_dim
    dims[outer_grid_params] = outer_dim

    ll_reshape = bulk_ll.reshape(tuple(dims))
    max_ll, min_ll = np.max(bulk_ll), np.min(bulk_ll)
    idx_max = np.unravel_index(
        np.argmax(np.median(ll_reshape, axis=(0, 1)), axis=None), (outer_dim, outer_dim)
    )
    log_levels = (
        1 + max_ll - np.sort(np.logspace(0, np.log10(max_ll - min_ll), nLevels))[::-1]
    )

    # plot
    figsize = (outer_dim * 5, outer_dim * 5)
    fig, axs = plt.subplots(
        outer_dim, outer_dim, figsize=figsize, sharey=True, sharex=True
    )
    images = {}

    b0s = grid_ls[2]
    b1s = grid_ls[3]

    # loop over each grid point on outler layer
    for i in range(outer_dim):
        for j in range(outer_dim):
            _ = axs[i, j].contourf(
                param_grid[0][:, :, i, j],
                param_grid[1][:, :, i, j],
                ll_reshape[:, :, i, j],
                log_levels,
                extend="both",
            )
            images[f"{i}_{j}"] = _
            axs[i, j].scatter(
                best_log[0], best_log[1], marker="*", s=200, label="LL best-fit"
            )
            if "true_params" in kwargs:
                true_params = kwargs.get("true_params")
                axs[i, j].scatter(
                    true_params[0], true_params[1], marker="*", s=200, label="True"
                )

            axs[i, j].set_title(
                f"log(b0):{b0s[i]:.2f};log(b1):{b1s[j]:.2f}", fontsize=18
            )

    # reset the norm
    norm = mpl.colors.SymLogNorm(
        linthresh=np.abs(max_ll), linscale=1, vmin=min_ll, vmax=max_ll + 1, base=10
    )
    for key in images:
        images[key].set_norm(norm)

    fig.colorbar(
        images[f"{idx_max[0]}_{idx_max[1]}"], ax=axs, fraction=0.05, format="%.0f"
    )
    fig.text(0.5, 0.06, "log(a1)", ha="center", fontsize=25)
    fig.text(0.06, 0.5, "log(a2)", va="center", rotation="vertical", fontsize=25)
    axs[idx_max[0], idx_max[1]].legend()


def plot_pred_drw_lc(
    lc_df, best_amp, best_tau, num_data=500, time_col="t", y_col="y", yerr_col="yerr"
):
    """
    Plot GP predicted light curve given best-fit parameters. 

    Args:
        lc_df (dataframe): The dataframe containing the light curve.
        best_amp (float): Best-fit DRW amplitude.
        best_tau (float): Best-fit DRW characteristic timescale.
        num_data (int): The number of points in the predicated LC.
        time_col (str, optional): Time columne name. Defaults to 't'.
        y_col (str, optional): Y axis column name. Defaults to 'y'.
        yerr_col (str, optional): Error column name. Defaults to 'yerr'.
    """

    # get LC data
    lc_df = lc_df.sort_values(by=time_col).reset_index(drop=True)
    t = lc_df[time_col].values - lc_df[time_col].min()
    y = lc_df[y_col].values
    yerr = lc_df[yerr_col].values

    # create GP model using params
    kernel = DRW_term(np.log(best_amp), np.log(best_tau))
    gp = celerite.GP(kernel, mean=np.mean(y))
    gp.compute(t, yerr)

    # generate pred LC
    t_pred = np.linspace(0, np.max(t), num_data)
    return_var = True
    try:
        mu, var = gp.predict(y, t_pred, return_var=return_var)
        std = np.sqrt(var)
    except FloatingPointError as e:
        print(e)
        print("No variance will be returned")
        return_var = False
        mu, var = gp.predict(y, t_pred, return_var=return_var)

    # Plot the data
    fig = plt.figure(figsize=(10, 5))
    color = "#ff7f0e"
    plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0, label="Input Data")
    plt.plot(t_pred, mu, color=color, label="Mean Prection")
    if return_var:
        plt.fill_between(
            t_pred, mu + std, mu - std, color=color, alpha=0.3, edgecolor="none"
        )
    plt.ylabel(r"Flux (arb. unit)")
    plt.xlabel(r"Time (day)")
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
    plt.title("Maximum Likelihood Prediction")
    plt.tight_layout()


def plot_pred_dho_lc(
    lc_df, best_params, num_data=500, time_col="t", y_col="y", yerr_col="yerr"
):
    """
    Plot GP predicted light curve given best-fit parameters. 

    Args:
        lc_df (dataframe): The dataframe containing the light curve.
        best_params (array-like): Best-fit DHO parameters.
        num_data (int): The number of points in the predicated LC.
        time_col (str, optional): Time columne name. Defaults to 't'.
        y_col (str, optional): Y axis column name. Defaults to 'y'.
        yerr_col (str, optional): Error column name. Defaults to 'yerr'.
    """

    # get LC data
    lc_df = lc_df.sort_values(by=time_col).reset_index(drop=True)
    t = lc_df[time_col].values - lc_df[time_col].min()
    y = lc_df[y_col].values
    yerr = lc_df[yerr_col].values

    # create GP model using params
    kernel = DHO_term(*np.log(best_params))
    gp = celerite.GP(kernel, mean=np.mean(y))
    gp.compute(t, yerr)

    # generate pred LC
    t_pred = np.linspace(0, np.max(t), num_data)
    return_var = True
    try:
        mu, var = gp.predict(y, t_pred, return_var=return_var)
        std = np.sqrt(var)
    except FloatingPointError as e:
        print(e)
        print("No variance will be returned")
        return_var = False
        mu, var = gp.predict(y, t_pred, return_var=return_var)

    # Plot the data
    fig = plt.figure(figsize=(10, 5))
    color = "#ff7f0e"
    plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0, label="Input Data")
    plt.plot(t_pred, mu, color=color, label="Mean Prection")
    if return_var:
        plt.fill_between(
            t_pred, mu + std, mu - std, color=color, alpha=0.3, edgecolor="none"
        )
    plt.ylabel(r"Flux (arb. unit)")
    plt.xlabel(r"Time (day)")
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
    plt.title("Maximum Likelihood Prediction")
    plt.tight_layout()


def plot_pred_lc(
    lc_df, best_params, p, q, num_data=500, time_col="t", y_col="y", yerr_col="yerr"
):
    """
    Plot GP predicted light curve given best-fit parameters. 

    Args:
        lc_df (dataframe): The dataframe containing the light curve.
        best_params (array-like): Best-fit CARMA parameters.
        p (int): CARMA p order.
        q (int): CARMA q order.
        num_data (int): The number of points in the predicated LC.
        time_col (str, optional): Time columne name. Defaults to 't'.
        y_col (str, optional): Y axis column name. Defaults to 'y'.
        yerr_col (str, optional): Error column name. Defaults to 'yerr'.
    """

    assert len(best_params) == int(p + q + 1)

    # get LC data
    lc_df = lc_df.sort_values(by=time_col).reset_index(drop=True)
    t = lc_df[time_col].values - lc_df[time_col].min()
    y = lc_df[y_col].values
    yerr = lc_df[yerr_col].values

    # create GP model using params
    kernel = CARMA_term(np.log(best_params[:p]), np.log(best_params[p:]))
    gp = celerite.GP(kernel, mean=np.mean(y))
    compute = True
    compute_ct = 0

    # compute can't factorize, try 4 more times
    while compute & (compute_ct < 5):
        compute_ct += 1
        try:
            gp.compute(t, yerr)
            compute = False
        except celerite.solver.LinAlgError:
            new_params = np.log(best_params) + 1e-6 * np.random.randn(p + q + 1)
            gp.set_parameter_vector(new_params)

    # generate pred LC
    t_pred = np.linspace(0, np.max(t), num_data)
    return_var = True
    try:
        mu, var = gp.predict(y, t_pred, return_var=return_var)
        std = np.sqrt(var)
    except FloatingPointError as e:
        print(e)
        print("No variance will be returned")
        return_var = False
        mu, var = gp.predict(y, t_pred, return_var=return_var)

    # Plot the data
    fig = plt.figure(figsize=(10, 5))
    color = "#ff7f0e"
    plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0, label="Input Data")
    plt.plot(t_pred, mu, color=color, label="Mean Prection")
    if return_var:
        plt.fill_between(
            t_pred, mu + std, mu - std, color=color, alpha=0.3, edgecolor="none"
        )
    plt.ylabel(r"Flux (arb. unit)")
    plt.xlabel(r"Time (day)")
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
    plt.title("Maximum Likelihood Prediction")
    plt.tight_layout()
