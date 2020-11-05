"""A set of functions to assist CARMA analysis.

Methods:
    carma_psd: Return a function that computes the PSD of the specified CARMA    
        process at given frequencies. 
    drw_psd: Return a function that computes the PSD of the specified DRW    
        process at given frequencies. 
    gp_psd: Return a function that computes the PSD of the native celerite GP 
        at given frequencies.
    carma_sf: Return a function that computes the structure function of the 
        specified CARMA process at given time lags.
    drw_sf: Return a function that computes the structure function of the 
        specified DRW process at given time lags.
    carma_acf: Return a function that computes the auto-correlation function of the 
        specified CARMA process at given time lags.
    drw_acf: Return a function that computes the auto-correlation function of the 
        specified DRW process at given time lags.
"""
import numpy as np
from .CARMATerm import acf, CARMA_term


def drw_psd(amp, tau):
    """Return a function that computes DRW Power Spectral Density (PSD).

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
    """Return a function that computes Power Spectral Density (PSD).

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
    """Return a function that computes native GP Power Spectral Density (PSD).

    Args:
        carmaTerm (object): A celerite CARMA term.
    Returns:
        A function that takes in frequencies and returns PSD at the 
            given frequencies.
    """

    def psd(f):
        return 2.5 * carmaTerm.get_psd(2 * np.pi * f)

    return psd


def drw_acf(tau):
    """Return a function that computes the DRW autocorrelation function (ACF).

    Args:
        tau (float): DRW decorrelation timescale.

    Returns:
        A function that takes in time lags and returns ACF at the given lags.
    """
    # convert to CARMA parameter
    a0 = 1 / tau

    def acf(lag):
        return np.exp(-a0 * lag)

    return acf


def carma_acf(arparams, maparams):
    """Return a function that computes the model autocorrelation function (ACF).

    Args:
        arparams (list): AR parameters in a list (or array-like)
        maparams (list): MA parameters in a list (or array-like)

    Returns:
        A function that takes in time lags and returns ACF at the given lags.
    """

    autocorr = acf(arparams, maparams)
    roots = np.roots(np.append([1], arparams))
    gpTerm = CARMA_term(np.log(arparams), np.log(maparams))

    def autocorr_func(lag):
        R = 0
        for i, r in enumerate(roots):
            R += autocorr[i] * np.exp(r * lag)

        return R / gpTerm.get_rms_amp() ** 2

    return autocorr_func


def drw_sf(amp, tau):
    """Return a function that computes the structure function (SF) of DRW.

    Args:
        amp (float): DRW variability amplitude
        tau (float): DRW decorrelation timescale.

    Returns:
        A function that takes in time lags and returns SF at the given lags.
    """

    def sf(lag):
        return np.sqrt(amp ** 2 * (1 - drw_acf(tau)(lag)))

    return sf


def carma_sf(arparams, maparams):
    """Return a function that computes the CARMA structure function (SF).

    Args:
        arparams (list): AR parameters in a list (or array-like)
        maparams (list): MA parameters in a list (or array-like)

    Returns:
        A function that takes in time lags and returns SF at the given lags.
    """
    gpTerm = CARMA_term(np.log(arparams), np.log(maparams))
    amp2 = gpTerm.get_rms_amp() ** 2

    def sf(lag):
        return np.sqrt(amp2 * (1 - carma_acf(arparams, maparams)(lag)))

    return sf
