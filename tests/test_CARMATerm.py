from eztao.carma.CARMATerm import *
from eztao.carma.CARMATerm import fcoeffs2coeffs
import numpy as np
from numpy.polynomial import polynomial as P
import pytest

# def test_0():
#     assert 0 == 0


def test_drw():
    term = DRW_term(np.log(0.35), np.log(100))

    # test celerite term coeffs
    celerite_coeffs = term.get_all_coefficients()
    assert np.allclose(celerite_coeffs[0][0], [0.1225])
    assert np.allclose(celerite_coeffs[1][0], [0.01])

    # test driving amplitude
    assert np.allclose(term.get_perturb_amp(), [0.04949747468305])


def test_dho():
    # Complex DHO
    term1 = DHO_term(np.log(2), np.log(1.2), np.log(1), np.log(3))
    term1_coeffs = term1.get_all_coefficients()

    assert term1_coeffs[0].size < 1
    assert term1_coeffs[1].size < 1
    assert np.allclose(term1.get_rms_amp(), [1.567907310185565])

    # Real DHO
    term2 = DHO_term(np.log(2), np.log(0.8), np.log(1), np.log(0.5))
    term2_coeffs = term2.get_all_coefficients()

    assert term2_coeffs[0].size > 0
    assert term2_coeffs[1].size > 0
    assert np.allclose(term2.get_rms_amp(), [0.6123724356957945])


def test_carma():
    # 1st CARMA(3,0)
    carma30a = CARMA_term(np.log([3, 2.8, 0.8]), np.log([1]))
    carma30a_coeffs = carma30a.get_all_coefficients()

    assert carma30a_coeffs[2].size < 1
    assert carma30a_coeffs[3].size < 1
    assert carma30a_coeffs[4].size < 1
    assert carma30a_coeffs[5].size < 1
    assert np.allclose(carma30a.get_rms_amp(), [0.496699633899391])

    # 2nd CARMA(3,0)
    carma30b = CARMA_term(np.log([3, 3.2, 1.2]), np.log([1]))
    carma30b_coeffs = carma30b.get_all_coefficients()

    assert carma30b_coeffs[2].size == 1
    assert carma30b_coeffs[3].size == 1
    assert carma30b_coeffs[4].size == 1
    assert carma30b_coeffs[5].size == 1
    assert np.allclose(carma30b.get_rms_amp(), [0.38575837490522974])


def test_acf():
    # CARMA(2,0)
    ar1 = np.array([2, 1.1])
    ma1 = np.array([0.5])

    carma20_acf = acf(np.roots([1, 2, 1.1]), ar1, ma1)
    answers = np.array([0.02840909 - 0.08983743j, 0.02840909 + 0.08983743j])

    assert np.allclose(carma20_acf, answers)


def test_fcoeffs2coeffs():
    """Test the function expanding factored polynomials"""
    assert np.allclose(
        fcoeffs2coeffs(np.array([1.0, 2, 1, 1])), P.polymul([1, 1, 2], [1, 1])
    )
    assert np.allclose(
        fcoeffs2coeffs(np.array([1.0, 2, 1.2])), P.polymul([1, 1, 2], 1.2)
    )
    assert np.allclose(fcoeffs2coeffs(np.array([1.0, 2])), P.polymul([1, 1], 2))


def test_carma_fcoeffs_log():
    """Test the function converting a CARMA model into the factored polynomial space"""
    kernel1 = CARMA_term(np.log([3, 2.8, 0.8]), np.log([1, 5]))
    log_fcoeffs = kernel1.carma2fcoeffs_log(np.log([3, 2.8, 0.8]), np.log([1, 5]))
    fcoeffs = np.exp(log_fcoeffs)

    # test caram2foceffs_log
    assert np.allclose(fcoeffs[-1] * fcoeffs[3], 1.0)

    # test fcoeffs2carma_log
    log_params = np.append(*kernel1.fcoeffs2carma_log(log_fcoeffs, 3))
    assert np.allclose(log_params, kernel1.get_parameter_vector())


def test_carma_fcoeffs():
    """Test the going to be deprecated version of carma_fcoeff version"""
    kernel1 = CARMA_term(np.log([3, 2.8, 0.8]), np.log([1, 5]))
    with pytest.deprecated_call():
        fcoeffs = kernel1.carma2fcoeffs(np.log([3, 2.8, 0.8]), np.log([1, 5]))

    # test caram2foceffs
    assert np.allclose(fcoeffs[-1] * fcoeffs[3], 1.0)

    # test fcoeffs2carma
    with pytest.deprecated_call():
        params = np.append(*kernel1.fcoeffs2carma(np.log(fcoeffs), 3))
    assert np.allclose(np.log(params), kernel1.get_parameter_vector())
