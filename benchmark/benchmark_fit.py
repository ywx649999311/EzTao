"""Fitting benchmark module"""
import numpy as np
import pandas as pd
from eztao.carma import DRW_term, DHO_term, CARMA_term
from eztao.ts.carma import *
from celerite import GP
import sys, time

# init kernels
drw1 = DRW_term(np.log(0.35), np.log(100))
drw2 = DRW_term(np.log(0.15), np.log(300))
drw3 = DRW_term(np.log(0.25), np.log(800))
dho1 = DHO_term(np.log(0.04), np.log(0.0027941), np.log(0.004672), np.log(0.0257))
dho2 = DHO_term(np.log(0.06), np.log(0.0001), np.log(0.0047), np.log(0.0157))
carma30a = CARMA_term(np.log([3, 2.8, 0.8]), np.log([1]))
carma30b = CARMA_term(np.log([3, 3.189, 1.2]), np.log([1]))


def main(output_path):

    # DRW
    ndpts = [100, 1_000, 10_000]
    max_dpt = ndpts[-1]
    drw_dts = []
    nLC = 20
    drw_init_params = [
        drw1.get_parameter_vector(),
        drw2.get_parameter_vector(),
        drw3.get_parameter_vector(),
    ]

    for dpts in ndpts:
        start = time.time()
        for i, kernel in enumerate([drw1, drw2, drw3]):
            kernel.set_parameter_vector(drw_init_params[i])
            t, y, yerr = gpSimRand(kernel, 100, 3650, dpts, full_N=max_dpt * 3, nLC=nLC)
            for j in range(nLC):
                drw_fit(t[j], y[j], yerr[j])
        drw_dts.append(time.time() - start)

    ## DHO
    dho_dts = []
    dho_init_params = [dho1.get_parameter_vector(), dho2.get_parameter_vector()]

    for dpts in ndpts:
        start = time.time()
        for i, kernel in enumerate([dho1, dho2]):
            kernel.set_parameter_vector(dho_init_params[i])
            t, y, yerr = gpSimRand(kernel, 100, 3650, dpts, full_N=max_dpt * 3, nLC=nLC)
            for j in range(nLC):
                dho_fit(t[j], y[j], yerr[j])
        dho_dts.append(time.time() - start)

    # save to file
    df = pd.DataFrame(
        {
            "data points": ndpts,
            "drw": np.array(drw_dts) / 60,
            "dho": np.array(dho_dts) / 40,
            # "carma_30": np.array(carma_30_dts) / 100,
        }
    )

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    output_path = sys.argv[1]
    main(output_path)