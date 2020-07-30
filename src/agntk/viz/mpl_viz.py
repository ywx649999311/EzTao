import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rc_file(
    "https://raw.githubusercontent.com/ywx649999311/project_template"
    "/master/%7B%7Bcookiecutter.project_name%7D%7D/src/vis/mpl/yu_basic.rc"
)


def plot_drw_ll(
    t, y, yerr, best_params, gp, prob_func, amp_range=None, tau_range=None, **kwargs
):
    """
    Function to plot DRW likelihood surface

    Args:
        t (array-like): An array of time points
        y (array-like): An array of fluxes at the abvoe time points
        yerr (array-like): An array of photometric errors
        best_params (array-like): Best-fit parameters in [amp, tau]
        gp (object): A DRW celerite GP object
        prob_func (func): Vectorized posterior/probability function
        amp_range (array-like, optional): The range of parameters to eval 
            probability. Defaults to None.
        tau_range (array-like, optional): The range of parameters to eval 
            probability. Defaults to None.
    Kwargs:
        grid_size (int): The number of points to eval likelihood along a 
            given axis.
        true_params (array-like): The true parameters if the given light curve 
            is simulated.
    """
    best_amp, best_tau = best_params
    grid_size = 50
    nLevels = 100

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
    bulk_lp = prob_func(ls_params_combo, y, yerr, gp)
    lp_reshape = bulk_lp.reshape((grid_size, grid_size))

    fig = plt.figure(figsize=(10, 6))
    plt.contourf(reg_param_grid[0], reg_param_grid[1], -lp_reshape, nLevels)
    plt.title("Likelihood Surface")
    plt.xlabel("Amp")
    plt.ylabel("Tau")
    plt.scatter(best_params[0], best_params[1], marker="*", s=200, label="Best-fit")

    if "true_params" in kwargs:
        true_params = kwargs.get("true_params")
        plt.scatter(true_params[0], true_params[1], marker="*", s=200, label="True")

    plt.colorbar()
    plt.legend()
