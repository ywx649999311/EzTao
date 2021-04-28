"""Testing CARMA fitting.
"""

import numpy as np
from eztao.carma import DRW_term, DHO_term, CARMA_term
from eztao.ts import *
from celerite import GP
from joblib import Parallel, delayed
import pytest

# init kernels
drw1 = DRW_term(np.log(0.35), np.log(100))
drw2 = DRW_term(np.log(0.15), np.log(300))
drw3 = DRW_term(np.log(0.25), np.log(800))
dho1 = DHO_term(np.log(0.04), np.log(0.0027941), np.log(0.004672), np.log(0.0257))
dho2 = DHO_term(np.log(0.06), np.log(0.0001), np.log(0.0047), np.log(0.0157))
carma31 = CARMA_term(np.log([3, 2.8, 0.8]), np.log([1, 5]))
carma30 = CARMA_term(np.log([3, 3.189, 0.05]), np.log([0.5]))
test_kernels = [drw1, drw2, drw3, dho1, dho2, carma30, carma31]


def test_drwFit():

    for kernel in [drw1, drw2, drw3]:
        t, y, yerr = gpSimRand(kernel, 50, 365 * 10.0, 200, nLC=200, season=False)
        best_fit_drw = np.array(
            Parallel(n_jobs=-1)(
                delayed(drw_fit)(t[i], y[i], yerr[i]) for i in range(len(t))
            )
        )

        best_perturb = np.sqrt(2 * best_fit_drw[:, 0] ** 2 / best_fit_drw[:, 1])
        perturb_diff = np.log(best_perturb) - np.log(kernel.get_perturb_amp())

        # make sure half of the best-fits within +/- 20% of the truth
        assert np.percentile(perturb_diff, 25) > np.log(0.8)
        assert np.percentile(perturb_diff, 75) < np.log(1.2)


def test_dhoFit():

    t1, y1, yerr1 = gpSimRand(dho1, 100, 365 * 10.0, 200, nLC=100, season=False)
    best_fit_dho1 = np.array(
        Parallel(n_jobs=-1)(
            delayed(dho_fit)(t1[i], y1[i], yerr1[i]) for i in range(len(t1))
        )
    )

    diff1 = np.log(best_fit_dho1[:, -1]) - dho1.parameter_vector[-1]

    # make sure half of the best-fits is reasonable based-on
    # previous simulations. (see LC_fit_fuctions.ipynb)
    assert np.percentile(diff1, 25) > -0.3
    assert np.percentile(diff1, 75) < 0.2

    # the second test will down scale lc by 1e6
    t2, y2, yerr2 = gpSimRand(dho2, 100, 365 * 10.0, 200, nLC=100, season=False)
    best_fit_dho2 = np.array(
        Parallel(n_jobs=-1)(
            delayed(dho_fit)(t2[i], y2[i] / 1e6, yerr2[i] / 1e6) for i in range(len(t2))
        )
    )

    diff2 = np.log(best_fit_dho2[:, -2]) - (dho2.parameter_vector[-2] - np.log(1e6))

    # make sure half of the best-fits is reasonable based-on
    # previous simulations. (see LC_fit_fuctions.ipynb)
    assert np.percentile(diff2, 25) > -0.3
    assert np.percentile(diff2, 75) < 0.2


def test_carmaFit():

    t1, y1, yerr1 = gpSimRand(carma30, 200, 365 * 10.0, 1500, nLC=150, season=False)
    best_fit_carma1 = np.array(
        Parallel(n_jobs=-1)(
            delayed(carma_fit)(t1[i], y1[i] / 1e-6, yerr1[i] / 1e-6, 3, 0)
            for i in range(len(t1))
        )
    )

    diff1 = np.log(best_fit_carma1[:, -1]) - (
        carma30.parameter_vector[-1] - np.log(1e-6)
    )

    # make sure half of the best-fits is within +/- 50% of the true
    assert np.percentile(diff1, 25) > -0.4
    assert np.percentile(diff1, 75) < 0.4

    # the second test will down scale lc by 1e6
    t2, y2, yerr2 = gpSimRand(carma31, 300, 365 * 10.0, 2000, nLC=200, season=False)
    best_fit_carma2 = np.array(
        Parallel(n_jobs=-1)(
            delayed(carma_fit)(t2[i], y2[i], yerr2[i], 3, 1) for i in range(len(t2))
        )
    )

    diff2 = np.log(best_fit_carma2[:, 0]) - (carma31.parameter_vector[0])

    # make sure half of the best-fits is within +/- 50% of the true
    assert np.percentile(diff2, 25) > -0.4
    assert np.percentile(diff2, 75) < 0.4


def test_pred_lc():
    """Test the carma_sim.pred_lc function."""

    nLC = 5
    for kernel in [drw2, dho1]:
        t0, y0, yerr0 = gpSimRand(kernel, 10, 365 * 10.0, 100, nLC=nLC)

        for i in range(nLC):
            best = carma_fit(t0[i], y0[i], yerr0[i], kernel.p, kernel.q)

            # check if residual < error
            t1, mu1, var1 = pred_lc(t0[i], y0[i], yerr0[i], best, kernel.p, t0[i])
            assert mu1.shape == t1.shape
            assert np.std(mu1 - y0[i]) < np.median(np.abs(yerr0[i]))

            # check if any NaN in pred lc
            t_pred = np.linspace(t1[0], t1[-1], 1000)
            t2, mu2, var2 = pred_lc(t0[i], y0[i], yerr0[i], best, kernel.p, t_pred)
            assert mu2.shape == t2.shape
            assert not np.isnan(mu2).any()
