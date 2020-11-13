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


def test_simByT():
    """Test function gpSimByT."""
    t = np.sort(np.random.uniform(0, 3650, 5000))
    kernels = [drw1, dho1, carma30b]
    nLC = 2
    SNR = 20

    for k in kernels:
        amp = k.get_rms_amp()
        tOut, yOut, yerrOut = gpSimByT(k, SNR, t, nLC=nLC)

        assert tOut.shape == (nLC, len(t))
        assert np.sum(yOut[0] < 0) > 0
        assert (np.argsort(yOut - yerrOut) == np.argsort(np.abs(yerrOut))).all()
        assert np.allclose(np.median(np.abs(yerrOut)), amp / SNR, rtol=0.2)


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

    for kernel in [dho1, dho2]:
        t, y, yerr = gpSimRand(kernel, 200, 365 * 10.0, 1000, nLC=100, season=False)
        best_fit_dho = np.array(
            Parallel(n_jobs=-1)(
                delayed(dho_fit)(t[i], y[i], yerr[i]) for i in range(len(t))
            )
        )

        diff = np.log(best_fit_dho[:, -1]) - kernel.parameter_vector[-1]

        # make sure half of the best-fits is reasonal based-on
        # previous simulations. (see LC_fit_fuctions.ipynb)
        assert np.percentile(diff, 25) > -0.35
        assert np.percentile(diff, 75) < 0.1


def test_carmaFit():

    carma20a = CARMA_term(np.log([0.03939692, 0.00027941]), np.log([0.0046724]))
    carma20b = CARMA_term(np.log([0.08, 0.00027941]), np.log([0.046724]))

    for kernel in [carma20a, carma20b]:
        t, y, yerr = gpSimRand(kernel, 100, 365 * 10.0, 500, nLC=100, season=False)
        best_fit_carma = np.array(
            Parallel(n_jobs=-1)(
                delayed(carma_fit)(t[i], y[i], yerr[i], 2, 0) for i in range(len(t))
            )
        )

        diff = np.log(best_fit_carma[:, -1]) - kernel.parameter_vector[-1]

        # make sure half of the best-fits is reasonal based-on
        # previous simulations. (see LC_fit_fuctions.ipynb)
        assert np.percentile(diff, 25) > -0.35
        assert np.percentile(diff, 75) < 0.1
