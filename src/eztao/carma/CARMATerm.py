import warnings
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
    roots = roots[roots.real.argsort()]  # acsending sort by real part

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

    @property
    def p(self):
        return 1

    @property
    def q(self):
        return 0


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

        # set order & trigger roots/acf computation
        self._p = len(log_arpars)
        self._q = len(log_mapars) - 1
        self.log_pars = np.append(log_arpars, log_mapars)

        # loop over par array to find out how many params
        for i in range(2, self.p + 1):
            arpar_names += (arpar_temp.format(i),)

        for i in range(1, self.q + 1):
            mapar_names += (mapar_temp.format(i),)

        self.parameter_names = arpar_names + mapar_names
        super(CARMA_term, self).__init__(*self._log_pars, **kwargs)

    @property
    def p(self):
        return self._p

    @property
    def q(self):
        return self._q

    @property
    def log_pars(self):
        return self._log_pars

    @log_pars.setter
    def log_pars(self, value):
        """log_pars setter, will trigger some computation."""
        if value.shape[0] == (self.p + self.q + 1):
            self._log_pars = value
        else:
            raise ValueError("Dimension mismatch!")

        # set pars and compute AR/MA roots, acf and determine real roots
        self._pars = _compute_exp(self._log_pars)
        self._arroots = _compute_roots(np.append([1 + 0j], self._pars[: self.p]))
        self._maroots = (
            _compute_roots(np.array(self._pars[self.p :][::-1], dtype=np.complex128))
            if self.q > 0
            else None
        )
        self.acf = acf(self._arroots, self._pars[: self.p], self._pars[self.p :])
        self.mask = np.iscomplex(self._arroots)

        self._pdef = True
        if (self._arroots.real > 0).any():
            self._pdef = False
            print("Warning: CARMA process is not stationary!")
        if (self._maroots is not None) and (self._maroots.real > 0).any():
            print("Warning: CARMA process is not at minimum phase!")

    def get_real_coefficients(self, params):

        # must trigger here, won't work when overload set_param_vector
        self.log_pars = params

        acf_real = self.acf[~self.mask]
        roots_real = self._arroots[~self.mask]

        ar = acf_real[:].real
        cr = -roots_real[:].real

        return (ar, cr)

    def get_complex_coefficients(self, params):

        acf_complex = self.acf[self.mask]
        roots_complex = self._arroots[self.mask]

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