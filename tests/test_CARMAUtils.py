import numpy as np
from eztao.carma import DRW_term, DHO_term, CARMA_term
from eztao.carma.model_utils import *

# init kernels
drw1 = DRW_term(np.log(0.35), np.log(100))
drw2 = DRW_term(np.log(0.15), np.log(300))
drw3 = DRW_term(np.log(0.25), np.log(800))
dho1 = DHO_term(np.log(0.04), np.log(0.0027941), np.log(0.004672), np.log(0.0257))
dho2 = DHO_term(np.log(0.06), np.log(0.0001), np.log(0.0047), np.log(0.0157))
carma30a = CARMA_term(np.log([3, 2.8, 0.8]), np.log([1]))
carma30b = CARMA_term(np.log([3, 3.2, 1.2]), np.log([1]))
# carma_kernels = [dho1, dho2, carma30a, carma30b]


def test_psd():
    # test PSD
    f = np.fft.rfftfreq(4000, 0.25)

    for kernel in [dho1, dho2, carma30a, carma30b]:
        arparams = np.exp(kernel.parameter_vector[: kernel.p])
        maparams = np.exp(kernel.parameter_vector[kernel.p :])
        assert np.allclose(
            np.log10(carma_psd(arparams, maparams)(f)[1:]),
            np.log10(gp_psd(kernel)(f)[1:]),
            atol=1e-2,
        )

    for kernel in [drw1, drw2]:
        amp = np.exp(kernel.parameter_vector[0])
        tau = np.exp(kernel.parameter_vector[1])
        assert np.allclose(
            np.log10(drw_psd(amp, tau)(f)[1:]),
            np.log10(gp_psd(kernel)(f)[1:]),
            atol=1e-2,
        )

