import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import celerite
from ..carma.CARMATerm import *


def plot_drw_ll(
    t,
    y,
    yerr,
    best_params,
    gp,
    prob_func,
    amp_range=None,
    tau_range=None,
    nLevels=10,
    **kwargs,
):
    """Plot DRW likelihood surface.

    Args:
        t (object): An array of time points.
        y (object): An array of fluxes at the abvoe time points.
        yerr (object): An array of photometric errors.
        best_params (object): Best-fit parameters in [amp, tau].
        gp (object): A DRW celerite GP object.
        prob_func (func): Posterior/Likelihood function with args=(params, y, yerr, gp)
        amp_range (object, optional): The range of parameters to evaluate likelihod.
            Defaults to None.
        tau_range (object, optional): The range of parameters to evaluate likelihod.
            Defaults to None.
        nLevels (int, optional): The number of levels in the coutour plot.
            Defaults to 10.

    Kwargs:
        grid_size (int): The number of points to evaluate likelihood along a given axis.
        true_params (object): The true DRW parameters of the provided light curve.
    """
    best_amp, best_tau = best_params
    grid_size = 40

    if amp_range is None:
        amp_range = [best_amp / np.e, np.e * best_amp]
    if tau_range is None:
        tau_range = [best_tau / np.e, np.e * best_tau]

    # check if custom resoultion
    if "grid_size" in kwargs:
        grid_size = int(kwargs.get("grid_size"))

    log_taus = np.linspace(np.log(tau_range[0]), np.log(tau_range[1]), grid_size)
    log_amps = np.linspace(np.log(amp_range[0]), np.log(amp_range[1]), grid_size)

    param_grid = np.meshgrid(log_amps, log_taus)  # log scale
    reg_param_grid = np.exp(param_grid)  # normal scale

    # flatten to list
    ls_params_combo = list(zip(param_grid[0].flatten(), param_grid[1].flatten()))

    # run in bulk
    gp.compute(t, yerr)
    vec_neg_ll = np.vectorize(prob_func, excluded=[1, 2, 3], signature="(n)->()")
    bulk_ll = np.negative(vec_neg_ll(ls_params_combo, y, yerr, gp))
    ll_reshape = bulk_ll.reshape((grid_size, grid_size))

    # normalize, better contours
    max_ll, min_ll = np.max(bulk_ll), np.min(bulk_ll)
    ll_range = max_ll - min_ll
    divnorm = mpl.colors.DivergingNorm(
        vmin=min_ll, vmax=max_ll + 1, vcenter=(5 + max_ll - 0.02 * ll_range)
    )

    # compute contour levels
    delta_levels = np.exp(np.linspace(0, np.log(ll_range), 10, endpoint=False))
    levels = max_ll - delta_levels
    levels = levels[::-1]

    # plot the main axes
    fig = plt.figure(dpi=150)
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])
    img = plt.contourf(
        param_grid[0],
        param_grid[1],
        ll_reshape,
        levels,
        norm=divnorm,
        extend="both",
        alpha=0.9,
    )
    plt.contour(
        param_grid[0], param_grid[1], ll_reshape, levels, colors=("k",), linewidths=0.5
    )
    ax.yaxis.set_major_locator(plt.MaxNLocator(7, prune="both"))
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    # colorbar
    cbar_ax = fig.add_axes([0.74, 0.1, 0.03, 0.8])
    cbar = plt.colorbar(img, cax=cbar_ax, format="%.f")
    cbar.minorticks_off()

    ax.scatter(
        np.log(best_params[0]),
        np.log(best_params[1]),
        marker="*",
        s=150,
        label="Best-fit",
        zorder=10,
    )

    if "true_params" in kwargs:
        true_params = kwargs.get("true_params")
        ax.scatter(
            np.log(true_params[0]),
            np.log(true_params[1]),
            marker="*",
            s=150,
            label="True",
            zorder=10,
        )

    ax.set_title("Loglikelihood Surface")
    ax.set_xlabel(r"$Log$(Amplitude)")
    ax.set_ylabel(r"$Log\/(\tau)$")
    ax.legend(markerscale=1)


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
    nLevels=10,
    **kwargs,
):
    """Plot DHO likelihood surface.

    Args:
        t (object): An array of time points.
        y (object): An array of fluxes at the abvoe time points.
        yerr (object): An array of photometric errors.
        best_params (object): Best-fit parameters in [a1, a2, b0, b1].
        gp (object): A DRW celerite GP object.
        prob_func (func): Posterior/Likelihood function with args=(params, y, yerr, gp).
        inner_dim (int, optional): The number of points to eval likelihood along
            a1 and a2. Defaults to 10.
        outer_dim (int, optional): The number of points to eval likelihood along
            b0 and b1. Defaults to 4.
        ranges (list, optional): Parameters range (log) within which to plot
            the surface. Defaults to [(None, None), (None, None), (None, None),
            (None, None)].
        nLevels (int, optional): Contour plot number of levels. Defaults to 10.

    Kwargs:
        true_params (object): The true parameters of the underlying DHO process if known.
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
    gp.set_parameter_vector(best_log)
    gp.compute(t, yerr)
    bulk_ll = np.negative(vec_neg_ll(ls_param_combos, y, yerr, gp))

    # back to shape (dim, dim, dim, dim)
    dims = np.empty(num_param, dtype=np.int)
    dims[inner_grid_params] = inner_dim
    dims[outer_grid_params] = outer_dim
    ll_reshape = bulk_ll.reshape(tuple(dims))

    # normalize & compute contour levels
    max_ll, min_ll = np.max(bulk_ll), np.min(bulk_ll)
    ll_range = max_ll - min_ll
    delta_levels = np.exp(np.linspace(0, np.log(ll_range), 10, endpoint=False))
    levels = max_ll - delta_levels
    levels = levels[::-1]
    divnorm = mpl.colors.DivergingNorm(
        vmin=min_ll, vmax=max_ll + 1, vcenter=(5 + max_ll - 0.1 * ll_range)
    )

    # determine the frame containing the best ll
    idx_max = np.unravel_index(
        np.argmax(np.median(ll_reshape, axis=(0, 1)), axis=None), (outer_dim, outer_dim)
    )

    # plot
    fig, axs = plt.subplots(
        outer_dim, outer_dim, figsize=(9, 8), sharey=True, sharex=True, dpi=200
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
                levels,
                extend="both",
            )
            images[f"{i}_{j}"] = _
            axs[i, j].scatter(
                best_log[0], best_log[1], marker="*", s=100, c="b", alpha=0.8
            )
            if "true_params" in kwargs:
                true_params = kwargs.get("true_params")
                axs[i, j].scatter(
                    true_params[0],
                    true_params[1],
                    marker="*",
                    s=100,
                    label="True",
                    c="orange",
                )
            axs[i, j].text(
                0.1,
                0.8,
                f"log(b0):{b0s[i]:.2f}\nlog(b1):{b1s[j]:.2f}",
                fontdict={"size": 6, "color": "w", "weight": 550},
                transform=axs[i, j].transAxes,
            )

    # reset the norm
    plt.subplots_adjust(hspace=0, wspace=0)
    for key in images:
        images[key].set_norm(divnorm)

    fig.colorbar(
        images[f"{idx_max[0]}_{idx_max[1]}"], ax=axs, fraction=0.05, format="%.0f"
    )

    # title and axis label
    fig.text(0.46, 0.04, "log(a1)", ha="center", fontsize=15)
    fig.text(0.04, 0.5, "log(a2)", va="center", rotation="vertical", fontsize=15)
    axs[idx_max[0], idx_max[1]].scatter(
        best_log[0], best_log[1], marker="*", s=150, c="r"
    )
    fig.suptitle("DHO Loglikelihood Surface", x=0.5, y=0.92)


def plot_pred_lc(t, y, yerr, best_ar, best_ma, t_pred):
    """
    Plot GP predicted light curve given best-fit parameters.

    Args:
        t (object): An array of time points
        y (object): An array of fluxes at the abvoe time points
        yerr (object): An array of photometric errors
        best_ar (object): Best-fit CARMA AR parameters in an array.
        best_ma (object): Best-fit CARMA MA parameters in an array.
        t_pred (object): Time points at which to generate predictions.
    """

    assert len(best_ar) >= len(
        best_ma
    ), "The dimension of AR must be greater than that of MA"

    # create GP model using params
    kernel = CARMA_term(np.log(best_ar), np.log(best_ma))
    gp = celerite.GP(kernel, mean=np.mean(y))
    compute_ct = 0

    # if compute can't factorize, try 4 more times
    while compute_ct < 5:
        try:
            gp.compute(t, yerr)
            compute_ct += 5
        except celerite.solver.LinAlgError:
            compute_ct += 1
            new_params = np.log(
                np.concatenate([best_ar, best_ma])
            ) + 1e-6 * np.random.randn(len(best_ar) + len(best_ma))
            gp.set_parameter_vector(new_params)

    # generate pred LC
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
    t_range = t_pred[-1] - t_pred[0]
    fig, ax = plt.subplots(1, 1, dpi=150, figsize=(8, 4))
    color = "#ff7f0e"
    ax.errorbar(
        t, y, yerr=yerr, fmt=".k", capsize=2, label="Original Data", markersize=3
    )
    ax.plot(t_pred, mu, color=color, label="Mean Prediction", alpha=0.8)
    if return_var:
        ax.fill_between(
            t_pred, mu + std, mu - std, color=color, alpha=0.3, edgecolor="none"
        )
    ax.set_xlim(t_pred[0] - t_range / 50, t_pred[-1] + t_range / 50)
    ax.set_ylabel(r"Flux (arb. unit)")
    ax.set_xlabel(r"Time (day)")
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_title("Maximum Likelihood Prediction")
    ax.legend()
    fig.tight_layout()
