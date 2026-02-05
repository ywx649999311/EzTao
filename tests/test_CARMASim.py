"""Testing CARMA simulation."""

import numpy as np
from eztao.carma import DRW_term, DHO_term, CARMA_term
from eztao.ts import *
import pytest

# init kernels
drw1 = DRW_term(np.log(0.35), np.log(100))
drw2 = DRW_term(np.log(0.15), np.log(300))
drw3 = DRW_term(np.log(0.25), np.log(800))
dho1 = DHO_term(np.log(0.04), np.log(0.0027941), np.log(0.004672), np.log(0.0257))
dho2 = DHO_term(np.log(0.06), np.log(0.0001), np.log(0.0047), np.log(0.0157))
carma31 = CARMA_term(np.log([3, 2.8, 0.8]), np.log([1, 5]))
carma30 = CARMA_term(np.log([3, 3.189, 0.05]), np.log([0.5]))
carma_invalid = CARMA_term(
    [1.95797093, -3.84868981, 0.71100209], [0.36438868, -2.96417798, 0.77545961]
)
test_kernels = [drw1, drw2, drw3, dho1, dho2, carma30, carma31]
rng = np.random.default_rng()


def test_invalidSim():
    """Test if sim function throws an exception when providing unstable term."""
    with pytest.raises(RuntimeError):
        t, y, yerr = gpSimFull(carma_invalid, 20, 365 * 10.0, 10000)


def test_simFullSeed():
    """Test the lc_seed option of gpSimFull."""

    # SNR = 20
    nLC = 5
    for kernel in test_kernels:
        t, y1, yerr1 = gpSimFull(kernel, 20, 365 * 10.0, 1000, nLC=nLC, lc_seed=10)
        t, y2, yerr2 = gpSimFull(kernel, 20, 365 * 10.0, 1000, nLC=nLC, lc_seed=10)

        # agree between two separate calls
        for i in range(nLC):
            assert np.allclose(y1[i], y2[i])
            assert np.allclose(yerr1[i], yerr2[i])

        # disagree between different simulations from one call
        assert not np.allclose(y1[0], y1[nLC - 1])
        assert not np.allclose(y1[1], y1[nLC - 2])
        assert not np.allclose(yerr1[0], yerr1[nLC - 1])
        assert not np.allclose(yerr1[1], yerr1[nLC - 2])


def test_simRand():
    """Test function gpSimRand."""
    # SNR = 20
    for kernel in test_kernels:
        t, y, yerr = gpSimRand(kernel, 20, 365 * 10.0, 150, nLC=100, season=False)
        log_amp = np.log(kernel.get_rms_amp())

        # compute offset in log space
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

    # test regular flux (not in mag)
    tF, yF, yerrF = gpSimRand(carma31, 20, 365 * 10.0, 150, nLC=1, log_flux=False)
    assert (np.argsort(yF) == np.argsort(-yerrF)).all()


def test_simRandSeed():
    """Test the downsample_seed option of gpSimRand."""

    # SNR = 20
    nLC = 1
    for kernel in test_kernels:
        t1, y1, yerr1 = gpSimRand(
            kernel, 20, 365 * 10.0, 150, nLC=nLC, lc_seed=10, downsample_seed=10
        )
        t2, y2, yerr2 = gpSimRand(
            kernel, 20, 365 * 10.0, 150, nLC=nLC, lc_seed=10, downsample_seed=10
        )
        t3, y3, yerr3 = gpSimRand(
            kernel, 20, 365 * 10.0, 150, nLC=nLC, lc_seed=10, downsample_seed=20
        )

        assert np.allclose(t1, t2)
        assert not np.allclose(t1, t3)


def test_simByTime():
    """Test function gpSimByTime."""
    t = np.sort(rng.uniform(0, 3650, 5000))
    kernels = [drw1, dho1, carma30]
    nLC = 2
    SNR = 20

    for k in kernels:
        amp = k.get_rms_amp()
        tOut, yOut, yerrOut = gpSimByTime(k, SNR, t, nLC=nLC)

        assert tOut.shape == (nLC, len(t))
        assert np.sum(yOut[0] < 0) > 0
        assert (np.argsort(yOut) == np.argsort(yerrOut)).all()
        assert np.allclose(np.median(np.abs(yerrOut)), amp / SNR, rtol=0.2)

    # test single LC simulation
    tOut, yOut, yerrOut = gpSimByTime(dho2, SNR, t, nLC=1)
    assert tOut.shape[0] == yOut.shape[0] == yerrOut.shape[0] == t.shape[0]


def test_addNoiseSeed():
    """Test the seed option of addNoise."""

    nLC = 1
    t1, y1, yerr1 = gpSimRand(
        test_kernels[-1], 20, 365 * 10.0, 150, nLC=nLC, lc_seed=10, downsample_seed=10
    )
    t2, y2, yerr2 = gpSimRand(
        test_kernels[-1], 20, 365 * 10.0, 150, nLC=nLC, lc_seed=10, downsample_seed=10
    )

    noise_y1 = addNoise(y1, yerr1, seed=10)
    noise_y2 = addNoise(y2, yerr2, seed=10)
    noise_y3 = addNoise(y1, yerr1, seed=20)

    assert np.allclose(noise_y1, noise_y2)
    assert not np.allclose(noise_y1, noise_y3)
