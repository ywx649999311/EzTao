"""Testing CARMA Simulation.
"""

import numpy as np
from eztao.carma import DRW_term, DHO_term, CARMA_term
from eztao.ts.carma import *
from celerite import GP
from joblib import Parallel, delayed
import pytest

# init kernels
drw1 = DRW_term(np.log(0.35), np.log(100))
drw2 = DRW_term(np.log(0.15), np.log(300))
drw3 = DRW_term(np.log(0.25), np.log(800))
dho1 = DHO_term(np.log(0.04), np.log(0.0027941), np.log(0.004672), np.log(0.0257))
dho2 = DHO_term(np.log(0.06), np.log(0.0001), np.log(0.0047), np.log(0.0157))
carma30a = CARMA_term(np.log([3, 2.8, 0.8]), np.log([1]))
carma30b = CARMA_term(np.log([3, 3.189, 1.2]), np.log([1]))
carma_invalid = CARMA_term(
    [1.95797093, -3.84868981, 0.71100209], [0.36438868, -2.96417798, 0.77545961]
)
test_kernels = [drw1, drw2, drw3, dho1, dho2, carma30a, carma30b]


def test_invalidSim():
    """Test if sim function throws an exception when providing unstable term."""
    with pytest.raises(RuntimeError):
        t, y, yerr = gpSimFull(carma_invalid, 20, 365 * 10.0, 10000)


def test_simRand():
    """Test function gpSimRand."""
    # SNR = 10
    for kernel in test_kernels:
        t, y, yerr = gpSimRand(kernel, 20, 365 * 10.0, 150, nLC=100, season=False)
        log_amp = np.log(kernel.get_rms_amp())

        # compute error in log space
        err = np.log(np.sqrt(np.var(y, axis=1) - np.var(yerr, axis=1))) - log_amp

        assert np.percentile(err, 25) > np.log(0.5)
        assert np.percentile(err, 75) < np.log(1.2)
        assert np.abs(np.median(err)) < 0.4

        # check returned dimension
        assert t.shape[1] == y.shape[1] == yerr.shape[1] == 150
        assert t.shape[0] == y.shape[0] == yerr.shape[0] == 100

    # test single LC simulation
    t, y, yerr = gpSimRand(dho2, 20, 365 * 10.0, 150, nLC=1, season=False)
    assert t.shape[0] == y.shape[0] == yerr.shape[0] == 150


def test_simByTime():
    """Test function gpSimByTime."""
    t = np.sort(np.random.uniform(0, 3650, 5000))
    kernels = [drw1, dho1, carma30b]
    nLC = 2
    SNR = 20

    for k in kernels:
        amp = k.get_rms_amp()
        tOut, yOut, yerrOut = gpSimByTime(k, SNR, t, nLC=nLC)

        assert tOut.shape == (nLC, len(t))
        assert np.sum(yOut[0] < 0) > 0
        assert (np.argsort(yOut - yerrOut) == np.argsort(np.abs(yerrOut))).all()
        assert np.allclose(np.median(np.abs(yerrOut)), amp / SNR, rtol=0.2)

    tOut, yOut, yerrOut = gpSimByTime(dho2, SNR, t, nLC=1)
    assert tOut.shape[0] == yOut.shape[0] == yerrOut.shape[0] == t.shape[0]


def test_drwFit():

    for kernel in [drw1, drw2]:
        t, y, yerr = gpSimRand(kernel, 50, 365 * 10.0, 500, nLC=100, season=False)
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

    # use de
    t1, y1, yerr1 = gpSimRand(dho1, 200, 365 * 10.0, 1000, nLC=100, season=False)
    best_fit_dho1 = np.array(
        Parallel(n_jobs=-1)(
            delayed(dho_fit)(t1[i], y1[i], yerr1[i]) for i in range(len(t1))
        )
    )

    diff1 = np.log(best_fit_dho1[:, -1]) - dho1.parameter_vector[-1]

    # make sure half of the best-fits is reasonal based-on
    # previous simulations. (see LC_fit_fuctions.ipynb)
    assert np.percentile(diff1, 25) > -0.35
    assert np.percentile(diff1, 75) < 0.1

    # use min
    t2, y2, yerr2 = gpSimRand(dho2, 200, 365 * 10.0, 1000, nLC=100, season=False)
    best_fit_dho2 = np.array(
        Parallel(n_jobs=-1)(
            delayed(dho_fit)(t2[i], y2[i], yerr2[i], diffEv=False)
            for i in range(len(t2))
        )
    )

    diff2 = np.log(best_fit_dho2[:, -1]) - dho2.parameter_vector[-1]

    # make sure half of the best-fits is reasonal based-on
    # previous simulations. (see LC_fit_fuctions.ipynb)
    assert np.percentile(diff2, 25) > -0.35
    assert np.percentile(diff2, 75) < 0.1


def test_carmaFit():

    carma20a = CARMA_term(np.log([0.03939692, 0.00027941]), np.log([0.0046724]))
    carma20b = CARMA_term(np.log([0.08, 0.00027941]), np.log([0.046724]))

    t1, y1, yerr1 = gpSimRand(carma20a, 100, 365 * 10.0, 500, nLC=100, season=False)
    best_fit_carma1 = np.array(
        Parallel(n_jobs=-1)(
            delayed(carma_fit)(t1[i], y1[i], yerr1[i], 2, 0, mode="coeff")
            for i in range(len(t1))
        )
    )

    diff1 = np.log(best_fit_carma1[:, -1]) - carma20a.parameter_vector[-1]

    # make sure half of the best-fits is reasonal based-on
    # previous simulations. (see LC_fit_fuctions.ipynb)
    assert np.percentile(diff1, 25) > -0.35
    assert np.percentile(diff1, 75) < 0.1

    t2, y2, yerr2 = gpSimRand(carma20b, 100, 365 * 10.0, 500, nLC=100, season=False)
    best_fit_carma2 = np.array(
        Parallel(n_jobs=-1)(
            delayed(carma_fit)(t2[i], y2[i], yerr2[i], 2, 0, diffEv=False, mode="coeff")
            for i in range(len(t2))
        )
    )

    diff2 = np.log(best_fit_carma2[:, -1]) - carma20b.parameter_vector[-1]

    # make sure half of the best-fits is reasonal based-on
    # previous simulations. (see LC_fit_fuctions.ipynb)
    assert np.percentile(diff2, 25) > -0.35
    assert np.percentile(diff2, 75) < 0.1

    # use min opt, pass if no error thrown
    for i in range(5):
        try:
            carma_fit(t1[i * 5], y1[i * 5], yerr1[i * 5], 3, 2, diffEv=False)
            carma_fit(t2[i * 5], y2[i * 5], yerr2[i * 5], 3, 0, diffEv=False)
        except ValueError as ve:
            if "violates bound" in ve.message:
                print(ve.message)
            else:
                raise ValueError("Unrecognized ValueEroor!")
