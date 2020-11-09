"""Benchmark module"""
import time
import numpy as np
import pandas as pd
from eztao.carma import DRW_term, DHO_term, CARMA_term
from eztao.ts.carma import (
    gpSimRand,
    neg_ll,
    drw_log_param_init,
    dho_log_param_init,
    carma_log_param_init,
)
from celerite import GP
import celerite
import sys


def main(output_path):

    ## test DRW
    drw_kernel = DRW_term(np.log(0.3), np.log(250))
    ndpts = [100, 1_000, 10_000, 100_000]
    max_dpt = ndpts[-1]
    drw_dts = []
    gp_fit_drw = GP(drw_kernel)
    init_params = drw_kernel.get_parameter_vector()

    for dpts in ndpts:
        # must reset to avoid unstable parameters
        gp_fit_drw.set_parameter_vector(init_params)
        t, y, yerr = gpSimRand(drw_kernel, 100, 10 * 3650, dpts, full_N=max_dpt * 3)
        drw_std = np.sqrt(np.var(y) - np.var(yerr))
        gp_fit_drw.compute(t[0], yerr[0])

        start = time.time()
        for i in range(100):
            neg_ll(init_params, y[0], yerr[0], gp_fit_drw)
        drw_dts.append(time.time() - start)

    ## test DHO
    dho_kernel = DHO_term(*dho_log_param_init())
    dho_dts = []
    gp_fit_dho = GP(dho_kernel)
    init_params = dho_kernel.get_parameter_vector()

    for dpts in ndpts:
        gp_fit_dho.set_parameter_vector(
            init_params
        )  # must reset to avoid unstable parameters
        t, y, yerr = gpSimRand(dho_kernel, 100, 10 * 3650, dpts, full_N=max_dpt * 3)
        gp_fit_dho.compute(t[0], yerr[0])

        start = time.time()
        for i in range(100):
            neg_ll(init_params, y[0], yerr[0], gp_fit_dho)
        dho_dts.append(time.time() - start)

    ## CARMA(3,0)
    p = 3
    q = 0
    carma30a = CARMA_term(np.log([3, 3.25, 1.2]), np.log([1]))
    params = carma30a.get_parameter_vector()
    carma_30_dts = []
    gp_fit_carma_30 = GP(carma30a)

    for dpts in ndpts:
        gp_fit_carma_30.set_parameter_vector(
            params
        )  # must reset to avoid unstable parameters
        t, y, yerr = gpSimRand(carma30a, 100, 10 * 3650, dpts, full_N=max_dpt * 3)
        gp_fit_carma_30.compute(t[0], y[0])

        start = time.time()
        for i in range(100):
            neg_ll(params, y[0], yerr[0], gp_fit_carma_30)
        carma_30_dts.append(time.time() - start)

    # save to file
    df = pd.DataFrame(
        {
            "data points": ndpts,
            "drw": np.array(drw_dts) / 100,
            "dho": np.array(dho_dts) / 100,
            "carma_30": np.array(carma_30_dts) / 100,
        }
    )

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    path = sys.argv[1]
    main(path)