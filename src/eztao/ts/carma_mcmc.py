"""A module containing functions to run MCMC using emcee."""

import numpy as np
import emcee
from eztao.ts import carma_fit, neg_param_ll, neg_fcoeff_ll
from eztao.carma import CARMA_term
from celerite import GP


def mcmc(t, y, yerr, p, q, n_walkers=32, burn_in=500, n_samples=2000, init_param=None):
    """A simple wrapper to run quick MCMC using emcee.

    Args:
        t (object): An array of time stamps (default in days).
        y (object): An array of y values.
        yerr (object): An array of the errors in y values.
        p (int): P order of a CARMA(p, q) model.
        q (int): Q order of a CARMA(p, q) model.
        n_walkers (int, optional): Number of MCMC walkers. Defaults to 32.
        burn_in (int, optional): Number of burn in steps. Defaults to 500.
        n_samples (int, optional): Number of MCMC steps to run. Defaults to 2000.
        init_param ([type], optional): The initial position for the MCMC walker.
            Defaults to None.

    Returns:
        object: The emcee sampler object.
        ndarray: The MCMC flatchain (n_walkers*n_samplers, dim) in CARMA space if
            p > 2, otherwise empty.
        ndarray: The MCMC chain (n_walkers, n_samplers, dim) in CARMA space if
            p > 2, otherwise empty.
    """
    assert p > q, "p order must be greater than q order."

    if init_param is not None and p <= 2:
        assert len(init_param) == int(
            p + q + 1
        ), "The initial parameters doesn't match the dimension of the CARMA model!"
    else:
        print("Searching for best-fit CARMA parameters...")
        init_param = carma_fit(t, y, yerr, p, q, n_iter=200)

    # set on param or fcoeff
    if p > 2:
        ll = lambda *args: -neg_fcoeff_ll(*args)
        init_sample = CARMA_term.carma2fcoeffs(
            np.log(init_param[:p]), np.log(init_param[p:])
        )
    else:
        ll = lambda *args: -neg_param_ll(*args)
        init_sample = init_param

    # create vectorized functions
    vec_fcoeff2carma = np.vectorize(
        CARMA_term.fcoeffs2carma,
        excluded=[
            1,
        ],
        signature="(n)->(m),(k)",
    )

    # reposition ts
    t = t - t[0]
    y = y - np.median(y)

    # init celerite kernel/GP
    kernel = CARMA_term(np.log(init_param[:p]), np.log(init_param[p:]))
    gp = GP(kernel, mean=0)
    gp.compute(t, yerr)

    # init sampler
    ndim = len(init_sample)
    sampler = emcee.EnsembleSampler(n_walkers, ndim, ll, args=[y, gp])

    print("Running burn-in...")
    p0 = np.log(init_sample) + 1e-8 * np.random.randn(n_walkers, ndim)
    p0, lp, _ = sampler.run_mcmc(p0, burn_in)

    print("Running production...")
    sampler.reset()
    sampler.run_mcmc(p0, n_samples)

    carma_flatchain = np.array([])
    carma_chain = np.array([])

    if p > 2:
        ar, ma = vec_fcoeff2carma(sampler.flatchain, p)
        carma = np.log(np.hstack((ar, ma)))
        carma_flatchain = carma
        carma_chain = carma.reshape((n_walkers, n_samples, ndim), order="F")

    return sampler, carma_flatchain, carma_chain