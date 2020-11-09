import numpy as np
from celerite import terms
from numba import njit, float64, complex128, int32, vectorize

__all__ = ["acf", "DRW_term", "DHO_term", "CARMA_term"]


@njit(complex128[:](complex128[:]))
def _compute_roots(coeffs):
    """Internal jitted function to compute roots"""

    # find roots using np and make roots that are almost real real
    roots = np.roots(coeffs)
    roots[np.abs(roots.imag) < 1e-10] = roots[np.abs(roots.imag) < 1e-10].real

    return roots


@njit(float64[:](float64[:]))
def _compute_exp(params):
    return np.exp(params)


@njit(complex128[:](complex128[:], float64[:], float64[:]))
def acf(arroots, arparam, maparam):
    """Return CARMA ACF coefficients given model parameter in Brockwell et al.
    2001 notation.

    Args:
        arroots (object): AR roots in a numpy array
        arparam (object): AR parameters in a numpy array
        maparam (object): MA parameters in a numpy array

    Returns:
        ACF coefficients in an array, each element correspond to a root
    """
    p = arparam.shape[0]
    q = maparam.shape[0] - 1
    sigma = maparam[0]

    # MA param into Kell's notation
    # arparam = np.array(arparam)
    maparam = np.array([x / sigma for x in maparam])

    # init acf product terms
    num_left = np.zeros(p, dtype=np.complex128)
    num_right = np.zeros(p, dtype=np.complex128)
    denom = -2 * arroots.real + np.zeros_like(arroots) * 1j

    for k in range(q + 1):
        num_left += maparam[k] * np.power(arroots, k)
        num_right += maparam[k] * np.power(np.negative(arroots), k)

    for j in range(1, p):
        root_idx = np.arange(p)
        root_k = arroots[np.roll(root_idx, j)]
        denom *= (root_k - arroots) * (np.conj(root_k) + arroots)

    return sigma ** 2 * num_left * num_right / denom


class DRW_term(terms.Term):
    """Damped Random Walk term.

    Args:
        log_sigma(float): Sigma is the standard deviation of the DRW process.
        log_tau(float): Tau is the characteristic timescale of the DRW process.
    """

    parameter_names = ("log_sigma", "log_tau")

    def get_real_coefficients(self, params):
        log_sigma, log_tau = params

        return (np.exp(2 * log_sigma), 1 / np.exp(log_tau))

    def get_perturb_amp(self):
        """Return the perturbing noise amplitude (b0)."""
        log_sigma, log_tau = self.get_parameter_vector()

        return np.exp((2 * log_sigma - np.log(1 / 2) - log_tau) / 2)

    def get_rms_amp(self):
        """Return the amplitude of CARMA process."""
        log_sigma, log_tau = self.get_parameter_vector()

        return np.exp(log_sigma)


class CARMA_term(terms.Term):
    """General CARMA term with arbitray parameters.

    Args:
        log_arpars (list): The logarithm of AR coefficients.
        log_mapars (list): The logarithm of MA coefficients.
    """

    def __init__(self, log_arpars, log_mapars, *args, **kwargs):

        arpar_temp = "log_a{}"
        mapar_temp = "log_b{}"
        arpar_names = ("log_a1",)
        mapar_names = ("log_b0",)

        self.arpars = _compute_exp(np.array(log_arpars))
        self.mapars = _compute_exp(np.array(log_mapars))
        self.p = len(self.arpars)
        self.q = len(self.mapars) - 1
        # self._roots = _compute_roots(np.append([1 + 0j], self.arpars))

        # combine ar and ma params into one array
        log_pars = np.append(log_arpars, log_mapars)

        # loop over par array to find out how many params
        for i in range(2, self.p + 1):
            arpar_names += (arpar_temp.format(i),)

        for i in range(1, self.q + 1):
            mapar_names += (mapar_temp.format(i),)

        self.parameter_names = arpar_names + mapar_names
        super(CARMA_term, self).__init__(*log_pars, **kwargs)

    def get_real_coefficients(self, params):

        # get roots and acf
        self.arpars = _compute_exp(params[: self.p])
        self.mapars = _compute_exp(params[self.p :])
        self._roots = _compute_roots(np.append([1 + 0j], self.arpars))
        self.acf = acf(self._roots, self.arpars, self.mapars)

        self.mask = np.iscomplex(self._roots)
        acf_real = self.acf[~self.mask]
        roots_real = self._roots[~self.mask]
        num_real = acf_real.shape[0]

        ar = acf_real[:num_real].real
        cr = -roots_real[:num_real].real

        return (ar, cr)

    def get_complex_coefficients(self, params):

        # mask = np.iscomplex(self._roots)
        acf_complex = self.acf[self.mask]
        roots_complex = self._roots[self.mask]

        ac = 2 * acf_complex[::2].real
        bc = 2 * acf_complex[::2].imag
        cc = -roots_complex[::2].real
        dc = -roots_complex[::2].imag

        return (ac, bc, cc, dc)

    def get_rms_amp(self):
        """Return the amplitude of CARMA process."""
        # force to recompute acf
        self.get_all_coefficients()

        return np.sqrt(np.abs(np.sum(self.acf)))


class DHO_term(CARMA_term):
    """Damped Harmonic Oscillator term.

    Args:
        log_a1 (float): The natual logarithm of DHO parameter a1.
        log_a2 (float): The natual logarithm of DHO parameter a2.
        log_b0 (float): The natual logarithm of DHO parameter b0.
        log_b1 (float): The natual logarithm of DHO parameter b1.
    """

    def __init__(self, log_a1, log_a2, log_b0, log_b1, *args, **kwargs):
        """Inits DHO term."""
        super(DHO_term, self).__init__([log_a1, log_a2], [log_b0, log_b1], **kwargs)
