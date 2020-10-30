"""A set of functions to assist CARMA analysis.

Methods:
    carma_psd: Return a function that computes the PSD of the specified CARMA    
        process at given frequencies. 
    drw_psd: Return a function that computes the PSD of the specified DRW    
        process at given frequencies. 
    gp_psd: Return a function that computes the PSD of the native celerite GP 
        at given frequencies.
    carma_sf: Return a function that computes the structure function of the 
        specified CARMA process at given time lages.
    carma_acf: Return a function that computes the auto-correlation function of the 
        specified CARMA process at given time lages.
"""
import numpy as np
from .CARMATerm import acf


def drw_psd(amp, tau):
    """Return a function that computes DRW Power Spectral Density.

    Args:
        amp (float): DRW amplitude
        tau (float): DRW decorrelation timescale

    Returns:
        A function that takes in frequencies and returns PSD at the 
            given frequencies.
    """

    # convert amp, tau to CARMA parameters
    a0 = 1 / tau
    sigma2 = 2 * amp ** 2 * a0

    def psd(f):
        return sigma2 / (a0 ** 2 + (2 * np.pi * f) ** 2)

    return psd


def carma_psd(arparams, maparams):
    """Return a function that computes Power Spectral Density.

    Args:
        arparams (object): Numpy array containing AR parameters.
        maparams (object): Numpy array containing MA parameters

    Returns:
        A function that takes in frequencies and returns PSD at the 
            given frequencies.
    """
    arparams = np.insert(arparams, 0, 1)
    maparams = np.array(maparams)
    arparams_rv = arparams[::-1]

    def psd(f):
        # init terms
        num_terms = np.complex(0)
        denom_terms = np.complex(0)

        for i, param in enumerate(maparams):
            num_terms += param * np.power(2 * np.pi * f * (1j), i)

        for k, param in enumerate(arparams_rv):
            denom_terms += param * np.power(2 * np.pi * f * (1j), k)

        num = np.abs(np.power(num_terms, 2))
        denom = np.abs(np.power(denom_terms, 2))

        return num / denom

    return psd


def gp_psd(carmaTerm):
    """Return a function that computes native GP PSD.

    Args:
        carmaTerm (object): A celerite CARMA term.
    """

    def psd(f):
        return 2.5 * carmaTerm.get_psd(2 * np.pi * f)

    return psd
