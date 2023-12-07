"""
A collection of GP kernels that express the autovariance structure of CARMA models using
celerite.
"""

import numpy as np
from celerite import terms
from numba import njit, float64, complex128
import warnings

__all__ = ["acf", "DRW_term", "DHO_term", "CARMA_term"]


@njit(complex128[:](complex128[:]))
def _compute_roots(coeffs):
    """Internal jitted function to compute roots"""

    # find roots using np and make roots that are almost real real
    roots = np.roots(coeffs)
    roots[np.abs(roots.imag) < 1e-10] = roots[np.abs(roots.imag) < 1e-10].real
    roots = roots[roots.real.argsort()]  # ascending sort by real part

    return roots


@njit(float64[:](float64[:]))
def _compute_exp(params):
    """Internal jitted np.exp"""
    return np.exp(params)


@njit(float64[:](float64[:], float64[:]))
def polymul(poly1, poly2):
    """Multiply two polynomials

    Inputs are the coefficients of the polynomials ranked by order (high
    to low). Internally, this function operates from low order to high order.
    """
    ## convert from high->low to low->high
    poly1 = poly1[::-1]
    poly2 = poly2[::-1]
    poly1_len = poly1.shape[0]
    poly2_len = poly2.shape[0]
    c = np.zeros(poly1_len + poly2_len - 1)

    for i in np.arange(poly1_len):
        for j in np.arange(poly2_len):
            c[i + j] += poly1[i] * poly2[j]

    ## convert back to high->low
    return c[::-1]


@njit(float64[:](float64[:]))
def fcoeffs2coeffs(fcoeffs):
    """Convert coeffs of a factored polynomial to the coeffs of the product

    The input fcoeffs follow the notation (index) in Jones et al. (1981) with the last
    element being an additional multiplying factor (the coeff of the highest-order term
    in the final expanded polynomial).

    :meta private:
    """
    size = fcoeffs.shape[0] - 1
    odd = bool(size & 0x1)
    nPair = size // 2
    poly = fcoeffs[-1:]  # The coeff of highest order term in the product

    if odd:
        poly = polymul(poly, np.array([1.0, fcoeffs[-2]]))

    for p in np.arange(nPair):
        poly = polymul(poly, np.array([1.0, fcoeffs[p * 2], fcoeffs[p * 2 + 1]]))

    # the returned is high->low
    return poly


def _roots2coeffs(roots):
    """Generate factored polynomial from roots

    The notation (index) for the factored polynomial follows that in
    Jones et al. (1981). Note: No multiplying is returned.
    """
    coeffs = []
    size = len(roots)
    odd = bool(size & 0x1)
    rootsComp = roots[roots.imag != 0]
    rootsReal = roots[roots.imag == 0]
    nCompPair = len(rootsComp) // 2
    nRealPair = len(rootsReal) // 2

    for i in range(nCompPair):
        root1 = rootsComp[i]
        root2 = rootsComp[i + 1]
        coeffs.append(-(root1.real + root2.real))
        coeffs.append((root1 * root2).real)

    for i in range(nRealPair):
        root1 = rootsReal[i]
        root2 = rootsReal[i + 1]
        coeffs.append(-(root1.real + root2.real))
        coeffs.append((root1 * root2).real)

    if odd:
        coeffs.append(-rootsReal[-1].real)
    return coeffs


@njit(complex128[:](complex128[:], float64[:], float64[:]))
def acf(arroots, arparam, maparam):
    """Get ACVF coefficients given CARMA parameters

    The CARMA noation (index) folows that in Brockwell et al. (2001).

    Args:
        arroots (array(complex)): AR roots in a numpy array
        arparam (array(float)): AR parameters in a numpy array
        maparam (array(float)): MA parameters in a numpy array

    Returns:
        array(complex): ACVF coefficients, each element correspond to a root.
    """
    p = arparam.shape[0]
    q = maparam.shape[0] - 1
    sigma = maparam[0]

    # MA param into Kelly's notation
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

    return sigma**2 * num_left * num_right / denom


class DRW_term(terms.Term):
    r"""
    Damped Random Walk (DRW) term (kernel)

    .. math::

        k_{DRW}(\Delta t) = \sigma^2\,e^{-\Delta t/\tau}

    with the parameters ``log_amp`` and ``log_tau``.

    Args:
        log_amp (float): The natural log of the RMS amplitude of the DRW process.
        log_tau (float): The natural log of the characteristic timescale of the DRW
            process.

    .. note::
        Conversions between EzTao DRW parameters and some other DRW representations
        seen in the literature:

        .. math::

            \mathrm{SF_{\infty}} = 2*\sigma^2

        .. math::

            \sigma^2 = \tau*\sigma_{KBS}^2/2

        .. math::

            \tau = 1/\alpha_1; \,\sigma_{KBS} = \beta_0

        see MacLeod et al. (2010) for SF_{\infty}} and Kelly et al. (2009)
        for \sigma_{KBS}. \alpha_1 and \beta_0 are the AR and MA parameters for a
        CARMA(1,0) model, respectively.
    """

    parameter_names = ("log_amp", "log_tau")

    def get_real_coefficients(self, params):
        """Get ``alpha_real`` and ``beta_real`` (coeffs of celerite's real kernel)

        Args:
            params (array(float)): Parameters of this kernel.

        Returns:
            (``alpha_real``, ``beta_real``).
        """
        log_amp, log_tau = params
        return (np.exp(2 * log_amp), 1 / np.exp(log_tau))

    def get_perturb_amp(self):
        """Get the amplitude of the perturbing noise (beta_0) in DRW

        Returns:
            The amplitude of the perturbing noise (beta_0) in the current DRW.
        """
        log_amp, log_tau = self.get_parameter_vector()
        return self.perturb_amp(log_amp, log_tau)

    @staticmethod
    def perturb_amp(log_amp, log_tau):
        """Compute the amplitude of the perturbing noise (beta_0) in DRW.

        Args:
            log_amp (float): The natural log of the RMS amplitude of the DRW process.
            log_tau (float): The natural log of the characteristic timescale of the DRW
                process.

        Returns:
            The amplitude of the perturbing noise (beta_0) in the DRW specified by the
            input parameters.
        """
        return np.exp((2 * log_amp - np.log(1 / 2) - log_tau) / 2)

    def get_rms_amp(self):
        """Get the RMS amplitude of this DRW process.

        Returns:
            The RMS amplitude of this DRW process.
        """
        log_amp, log_tau = self.get_parameter_vector()
        return np.exp(log_amp)

    def get_carma_parameter(self):
        """Get DRW parameters in CARMA notation (alpha_*/beta_*).

        Returns:
            [alpha_1, beta_0].
        """
        return [1 / np.exp(self.get_parameter("log_tau")), self.get_perturb_amp()]

    @property
    def p(self):
        return 1

    @property
    def q(self):
        return 0


class CARMA_term(terms.Term):
    """A general-purpose CARMA term (kernel)

    Args:
        log_arpars (array(float)): Natural log of the AR coefficients.
        log_mapars (array(float)): Natural log of the MA coefficients.
    """

    def __init__(self, log_arpars, log_mapars, *args, **kwargs):
        arpar_temp = "log_a{}"
        mapar_temp = "log_b{}"
        arpar_names = ("log_a1",)
        mapar_names = ("log_b0",)

        # set order & trigger roots/acf computation
        log_pars = np.append(log_arpars, log_mapars)
        self._p = len(log_arpars)
        self._q = len(log_mapars) - 1
        self._dim = self._p + self._q + 1
        self._compute(log_pars)

        # check if stationary
        self._arroots = _compute_roots(np.append([1 + 0j], self._pars[: self._p]))
        if (self._arroots.real > 0).any():
            print("Warning: CARMA process is not stationary!")

        # loop over par array to find out how many params
        for i in range(2, self._p + 1):
            arpar_names += (arpar_temp.format(i),)

        for i in range(1, self._q + 1):
            mapar_names += (mapar_temp.format(i),)

        self.parameter_names = arpar_names + mapar_names
        super().__init__(*log_pars, **kwargs)

    @property
    def p(self):
        return self._p

    @property
    def q(self):
        return self._q

    def _compute(self, params):
        """Compute important CARMA parameters"""
        self._pars = _compute_exp(params)
        self._arroots = _compute_roots(np.append([1 + 0j], self._pars[: self._p]))
        self.acf = acf(self._arroots, self._pars[: self._p], self._pars[self._p :])
        self.mask = self._arroots.imag != 0

    def set_log_fcoeffs(self, log_fcoeffs):
        """Set kernel parameters

        Use coeffs of the factored polynomial to set CARMA paramters, note that the
        last input coeff is always the coeff for the highest-order MA differential.
        While performing the conversion, 1 is added to the AR coeffs to maintain the
        same formatting (AR polynomials always have the highest order coeff to be 1).

        Args:
            log_fcoeffs (array(float)): Natural log of the coefficients for the
                factored characteristic polynomial, with the last coeff being an
                additional multiplying factor on this polynomial.

        """
        if log_fcoeffs.shape[0] != (self._dim):
            raise ValueError("Dimension mismatch!")

        fcoeffs = _compute_exp(log_fcoeffs)
        ARpars = fcoeffs2coeffs(np.append(fcoeffs[: self._p], [1]))[1:]
        MApars = fcoeffs2coeffs(fcoeffs[self._p :])[::-1]
        self.set_parameter_vector(np.log(np.append(ARpars, MApars)))

    def get_real_coefficients(self, params):
        """
        Get arrays of ``alpha_real`` and ``beta_real`` (coefficients of celerite's
        real kernel)

        Args:
            params (array(float)): Parameters of this kernel.

        Returns:
            Arrays of ``alpha_real`` and ``beta_real``, one for each.
        """

        # trigger re_compute & get celerite coeffs
        self._compute(params)
        acf_real = self.acf[~self.mask]
        roots_real = self._arroots[~self.mask]

        ar = acf_real[:].real
        cr = -roots_real[:].real

        return (ar, cr)

    def get_complex_coefficients(self, params):
        """
        Get arrays of ``alpha_complex_real``, ``alpha_complex_imag``,
        ``beta_complex_real`` and ``beta_complex_imag`` (coefficients of celerite's
        complex kernel)

        Args:
            params (array(float)): Parameters of this kernel.

        Returns:
            Arrays of ``alpha_complex_real``, ``alpha_complex_imag``,
            ``beta_complex_real`` and ``beta_complex_imag``, one for each.
        """

        acf_complex = self.acf[self.mask]
        roots_complex = self._arroots[self.mask]

        ac = 2 * acf_complex[::2].real
        bc = 2 * acf_complex[::2].imag
        cc = -roots_complex[::2].real
        dc = -roots_complex[::2].imag

        return (ac, bc, cc, dc)

    def get_rms_amp(self):
        """Get the RMS amplitude of this CARMA kernel

        Returns:
            The RMS amplitude of this CARMA kernel.
        """
        log_pars = self.get_parameter_vector()
        return self.rms_amp(log_pars[: self.p], log_pars[self.p :])

    @staticmethod
    def rms_amp(log_arpars, log_mapars):
        """Compute the RMS amplitude of a CARMA kernel

        Args:
            log_arpars (array(float)): Natural log of the AR coefficients.
            log_mapars (array(float)): Natural log of the MA coefficients.

        Returns:
            The RMS amplitude of the CARMA kernel specified by the input parameters.
        """
        _p = len(log_arpars)
        _pars = _compute_exp(np.append(log_arpars, log_mapars))
        _arroots = _compute_roots(np.append([1 + 0j], _pars[:_p]))
        _acf = acf(_arroots, _pars[:_p], _pars[_p:])

        return np.sqrt(np.abs(np.sum(_acf)))

    @staticmethod
    def carma2fcoeffs_log(log_arpars, log_mapars):
        """Get the representation of a CARMA kernel in the factored polynomial space

        Args:
            log_arpars (array(float)): Natural log of the AR coefficients.
            log_mapars (array(float)): Natural log of the MA coefficients.

        Returns:
            array(float): The coefficients (in natural log) of the factored polymoical
                for the CARMA kernel specified by the input parameters. The last coeff
                is a multiplying factor of the returned polynomial.
        """

        _p = len(log_arpars)
        _q = len(log_mapars) - 1
        _pars = _compute_exp(np.append(log_arpars, log_mapars))
        _arroots = _compute_roots(np.append([1 + 0j], _pars[:_p]))
        _maroots = _compute_roots(np.array(_pars[_p:][::-1], dtype=np.complex128))
        ma_mult = _pars[-1:]  ## the multiplying factor
        ar_coeffs = _roots2coeffs(_arroots)

        if _q > 0:
            ma_coeffs = _roots2coeffs(_maroots)
            ma_coeffs.append(ma_mult[0])
        else:
            ma_coeffs = ma_mult

        return np.log(np.append(ar_coeffs, ma_coeffs))

    @staticmethod
    def fcoeffs2carma_log(log_fcoeffs, p):
        """Get the representation of a CARMA kernel in the nominal CARMA parameter space

        Args:
            log_coeffs (array(float)): The array of coefficients for the factored
                polynomial with the last coeff being a multiplying factor of the
                polynomial.
            p (int): The p order of the CARMA kernel.

        Returns:
            Natural log of the AR and MA parameters in two separate arrays.
        """

        fcoeffs = np.exp(log_fcoeffs)

        # Append one to AR fcoeffs as the multiplying factor; MA foceffs has that included.
        # Index in CARMA for AR: high -> low; for MA: low -> high
        ARpars = fcoeffs2coeffs(np.append(fcoeffs[:p], [1]))[1:]
        MApars = fcoeffs2coeffs(fcoeffs[p:])[::-1]

        return np.log(ARpars), np.log(MApars)

    @staticmethod
    def carma2fcoeffs(log_arpars, log_mapars):
        """Get the representation of a CARMA kernel in the factored polynomial space

        A wrapper of `CARMA_term.carma2fcoeffs_log` for backward compatibility. This
        function will be deprecated in future releases.
        """
        warnings.warn("Use carma2fcoeffs_log instead", DeprecationWarning)
        log_fcoeffs = CARMA_term.carma2fcoeffs_log(log_arpars, log_mapars)
        return np.exp(log_fcoeffs)

    @staticmethod
    def fcoeffs2carma(log_fcoeffs, p):
        """Get the representation of a CARMA kernel in the nominal CARMA parameter space

        A wrapper of `CARMA_term.fcoeffs2carma_log` for backward compatibility. This
        function will be deprecated in future releases.
        """
        warnings.warn("Use fcoeffs2carma_log instead", DeprecationWarning)
        log_ar, log_ma = CARMA_term.fcoeffs2carma_log(log_fcoeffs, p)
        return np.exp(log_ar), np.exp(log_ma)


class DHO_term(CARMA_term):
    """Damped Harmonic Oscillator (DHO) term (kernel)

    Args:
        log_a1 (float): Natural log of the DHO parameter a1.
        log_a2 (float): Natural log of the DHO parameter a2.
        log_b0 (float): Natural log of the DHO parameter b0.
        log_b1 (float): Natural log of the DHO parameter b1.
    """

    def __init__(self, log_a1, log_a2, log_b0, log_b1, *args, **kwargs):
        """Initiate the DHO term."""
        super(DHO_term, self).__init__([log_a1, log_a2], [log_b0, log_b1], **kwargs)
