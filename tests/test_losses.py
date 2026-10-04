import numpy as np
import pytest

from sepsis_cdss.losses import focal_weighted_bce, grad_hess, grad_z


@pytest.mark.parametrize("alpha,beta,gamma", [(0.5, 0.025, 0.5), (0.3, 1.0, 2.0), (0.5, 1.0, 0.0)])
def test_gradient_matches_finite_difference(alpha, beta, gamma):
    z = np.linspace(-4, 4, 41)
    for y in (0, 1):
        yy = np.full_like(z, y)
        h = 1e-5
        num = (focal_weighted_bce(z + h, yy, alpha, beta, gamma)
               - focal_weighted_bce(z - h, yy, alpha, beta, gamma)) / (2 * h)
        assert np.allclose(grad_z(z, yy, alpha, beta, gamma), num, atol=1e-5)


def test_hessian_positive_and_finite():
    z = np.linspace(-6, 6, 61)
    for y in (0, 1):
        g, h = grad_hess(z, np.full_like(z, y))
        assert np.all(np.isfinite(g)) and np.all(h > 0)


def test_gamma_zero_alpha_half_reduces_to_weighted_bce():
    z = np.array([-2.0, 0.3, 1.5])
    p = 1 / (1 + np.exp(-z))
    # y=1: loss = -0.5*log p -> grad wrt z = -0.5*(1-p); y=0 with beta: 0.5*beta*p
    assert np.allclose(grad_z(z, np.ones(3), 0.5, 0.2, 0.0), -0.5 * (1 - p))
    assert np.allclose(grad_z(z, np.zeros(3), 0.5, 0.2, 0.0), 0.5 * 0.2 * p)


def test_beta_scales_false_positive_penalty():
    z = np.array([1.0])
    low = abs(grad_z(z, np.zeros(1), beta=0.025))
    high = abs(grad_z(z, np.zeros(1), beta=1.0))
    assert high == pytest.approx(low * 40)
