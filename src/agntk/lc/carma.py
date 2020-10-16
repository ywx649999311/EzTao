"""Functions for performing CARMA analysis on LCs.
"""

import numpy as np
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

    while factor:
        try:
            gp_sim.compute(t, yerr)
            factor = False
        except Exception:
            # if error, try to re-init t and yerr
            t = np.linspace(0, duration, N)
            yerr = np.random.normal(0, carmaTerm.get_rms_amp() / SNR, N)

            fact_num += 1
            if fact_num > 10:
                raise Exception(
                    "Celerite cannot factorize the GP"
                    + " covairance matrix, try again!"
                )

    t = np.repeat(t[None, :], nLC, axis=0)
    yerr = np.repeat(yerr[None, :], nLC, axis=0)
    y = gp_sim.sample(size=nLC)

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
