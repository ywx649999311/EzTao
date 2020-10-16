"""Testing CARMA Simulation.
"""

import numpy as np
from agntk.carma.CARMATerm import *
from agntk.lc.carma import *
from celerite import GP

# init kernels
drw1 = DRW_term(np.log(0.35), np.log(100))
drw2 = DRW_term(np.log(0.15), np.log(300))
drw3 = DRW_term(np.log(0.25), np.log(500))
dho1 = DHO_term(np.log(2), np.log(1.2), np.log(1), np.log(3))
dho2 = DHO_term(np.log(2), np.log(0.8), np.log(1), np.log(0.5))
carma30a = CARMA_term(np.log([3, 2.8, 0.8]), np.log([1]))
carma30b = CARMA_term(np.log([3, 3.2, 1.2]), np.log([1]))
test_kernels = [drw1, drw2, drw3, dho1, dho2, carma30a, carma30b]


def test_simRand():

    # SNR = 10
    for kernel in test_kernels:
        t, y, yerr = gpSimRand(kernel, 10, 365 * 10.0, 150, nLC=100)
        carma_amp = kernel.get_rms_amp()

        # compute percentage error
        err = (
            np.sqrt(np.var(y, axis=1) - np.var(yerr, axis=1)) - carma_amp
        ) / carma_amp

        # pass if std of percentage error < 0.25
        assert np.std(err) < 0.25
