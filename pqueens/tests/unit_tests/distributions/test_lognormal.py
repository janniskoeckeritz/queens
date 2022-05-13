"""Test-module for lognormal distribution."""

import numpy as np
import pytest
import scipy.stats

from pqueens.distributions import from_config_create_distribution


# ------------- univariate --------------
@pytest.fixture(params=[1.0, [5.0, 0.0, -1.0, 2.0]])
def sample_pos_1d(request):
    """Sample position to be evaluated."""
    return np.array(request.param)


@pytest.fixture(scope='module')
def mean_1d():
    """A possible scalar mean value."""
    return 1.0


@pytest.fixture(scope='module')
def covariance_1d():
    """A possible scalar variance value."""
    return 2.0


@pytest.fixture(scope='module')
def lognormal_1d(mean_1d, covariance_1d):
    """A valid normal distribution."""
    distribution_options = {
        'distribution': 'lognormal',
        'normal_mean': mean_1d,
        'normal_covariance': covariance_1d,
    }
    return from_config_create_distribution(distribution_options)


@pytest.fixture(scope='module')
def uncorrelated_vector_1d(num_draws):
    """A vector of uncorrelated samples from standard normal distribution.

    as expected by a call to a Gaussian random number generator
    """
    vec = [[1.0]]
    return np.tile(vec, num_draws)


# ------------- multivariate --------------
@pytest.fixture(
    params=[[1.0, 2.0], [[2.0, -1.0], [1.0, 0.5], [2.0, -1.0], [0.0, -1.0], [-2.0, -1.0]]]
)
def sample_pos_2d(request):
    """Sample position to be evaluated."""
    return np.array(request.param)


@pytest.fixture(scope='module')
def mean_2d():
    """A possible mean vector."""
    return np.array([1.0, -2.0])


@pytest.fixture(scope='module')
def covariance_2d():
    """Recompose matrix based on given Cholesky decomposition."""
    return np.array([[1.0, 0.0], [0.0, 2.0]])


@pytest.fixture(scope='module')
def lognormal_2d(mean_2d, covariance_2d):
    """A multivariate normal distribution."""
    distribution_options = {
        'distribution': 'lognormal',
        'normal_mean': mean_2d,
        'normal_covariance': covariance_2d,
    }
    return from_config_create_distribution(distribution_options)


@pytest.fixture(scope='module', params=[1, 4])
def num_draws(request):
    """Number of samples to draw from distribution."""
    return request.param


@pytest.fixture(scope='module')
def uncorrelated_vector_2d(num_draws):
    """A vector of uncorrelated samples from standard normal distribution.

    as expected by a call to a Gaussian random number generator
    """
    vec = [[1.0], [-2.0]]
    return np.tile(vec, num_draws)


# -----------------------------------------------------------------------
# ---------------------------- TESTS ------------------------------------
# -----------------------------------------------------------------------

# ------------- univariate --------------
@pytest.mark.unit_tests
def test_init_lognormal_1d(lognormal_1d, mean_1d, covariance_1d):
    """Test init method of LogNormal Distribution class."""
    std = np.sqrt(covariance_1d)
    mean_ref = scipy.stats.lognorm.mean(scale=np.exp(mean_1d), s=std).reshape(1)
    var_ref = scipy.stats.lognorm.var(scale=np.exp(mean_1d), s=std).reshape(1, 1)

    assert lognormal_1d.dimension == 1
    np.testing.assert_equal(lognormal_1d.normal_mean, np.array(mean_1d).reshape(1))
    np.testing.assert_equal(lognormal_1d.normal_covariance, np.array(covariance_1d).reshape(1, 1))
    np.testing.assert_allclose(lognormal_1d.mean, mean_ref)
    np.testing.assert_allclose(lognormal_1d.covariance, var_ref)


@pytest.mark.unit_tests
def test_init_lognormal_1d_incovariance(mean_1d, covariance_1d):
    """Test init method of LogNormal Distribution class."""
    with pytest.raises(np.linalg.LinAlgError, match=r'Cholesky decomposition failed *'):
        distribution_options = {
            'distribution': 'lognormal',
            'normal_mean': mean_1d,
            'normal_covariance': -covariance_1d,
        }
        from_config_create_distribution(distribution_options)


@pytest.mark.unit_tests
def test_cdf_lognormal_1d(lognormal_1d, mean_1d, covariance_1d, sample_pos_1d):
    """Test cdf method of LogNormal Distribution distribution class."""
    std = np.sqrt(covariance_1d)
    ref_sol = scipy.stats.lognorm.cdf(sample_pos_1d, scale=np.exp(mean_1d), s=std).reshape(-1)
    np.testing.assert_allclose(lognormal_1d.cdf(sample_pos_1d), ref_sol)


@pytest.mark.unit_tests
def test_draw_lognormal_1d(lognormal_1d, mean_1d, covariance_1d, uncorrelated_vector_1d, mocker):
    """Test the draw method of normal distribution."""
    mocker.patch('numpy.random.randn', return_value=uncorrelated_vector_1d)
    draw = lognormal_1d.draw()
    ref_sol = np.exp(mean_1d + covariance_1d ** (1 / 2) * uncorrelated_vector_1d.T).reshape(-1, 1)
    np.testing.assert_equal(draw, ref_sol)


@pytest.mark.unit_tests
def test_logpdf_lognormal_1d(lognormal_1d, mean_1d, covariance_1d, sample_pos_1d):
    """Test pdf method of LogNormal Distribution distribution class."""
    std = np.sqrt(covariance_1d)
    ref_sol = scipy.stats.lognorm.logpdf(sample_pos_1d, scale=np.exp(mean_1d), s=std).reshape(-1)
    ref_sol[ref_sol == -np.inf] = np.nan  # Queens Log Normal is not defined for <=0.
    np.testing.assert_allclose(lognormal_1d.logpdf(sample_pos_1d), ref_sol)


@pytest.mark.unit_tests
def test_pdf_lognormal_1d(lognormal_1d, mean_1d, covariance_1d, sample_pos_1d):
    """Test pdf method of LogNormal Distribution distribution class."""
    std = np.sqrt(covariance_1d)
    ref_sol = scipy.stats.lognorm.pdf(sample_pos_1d, scale=np.exp(mean_1d), s=std).reshape(-1)
    ref_sol[ref_sol == 0] = np.nan  # Queens Log Normal is not defined for <=0.
    np.testing.assert_allclose(lognormal_1d.pdf(sample_pos_1d), ref_sol)


@pytest.mark.unit_tests
def test_ppf_lognormal_1d(lognormal_1d, mean_1d, covariance_1d):
    """Test ppf method of LogNormal Distribution distribution class."""
    std = np.sqrt(covariance_1d)
    quantile = 0.5
    ref_sol = scipy.stats.lognorm.ppf(quantile, scale=np.exp(mean_1d), s=std).reshape(-1)
    np.testing.assert_allclose(lognormal_1d.ppf(quantile), ref_sol)


# ------------- multivariate --------------
@pytest.mark.unit_tests
def test_init_lognormal_2d(lognormal_2d, mean_2d, covariance_2d):
    """Test init method of LogNormal Distribution class."""
    std = np.diag(np.sqrt(covariance_2d))
    mean_ref = np.array(
        [
            scipy.stats.lognorm.mean(scale=np.exp(mean_2d[0]), s=std[0]),
            scipy.stats.lognorm.mean(scale=np.exp(mean_2d[1]), s=std[1]),
        ]
    )
    var_ref = np.diag(
        [
            scipy.stats.lognorm.var(scale=np.exp(mean_2d[0]), s=std[0]),
            scipy.stats.lognorm.var(scale=np.exp(mean_2d[1]), s=std[1]),
        ]
    )

    assert lognormal_2d.dimension == 2
    np.testing.assert_equal(lognormal_2d.normal_mean, mean_2d)
    np.testing.assert_equal(lognormal_2d.normal_covariance, covariance_2d)
    np.testing.assert_allclose(lognormal_2d.mean, mean_ref)
    np.testing.assert_allclose(lognormal_2d.covariance, var_ref)


@pytest.mark.unit_tests
def test_cdf_lognormal_2d(lognormal_2d, mean_2d, covariance_2d, sample_pos_2d):
    """Test cdf method of LogNormal Distribution distribution class."""
    sample_pos_2d = sample_pos_2d.reshape(-1, 2)
    std = np.diag(np.sqrt(covariance_2d))
    ref_sol = scipy.stats.lognorm.cdf(
        sample_pos_2d[:, 0], scale=np.exp(mean_2d[0]), s=std[0]
    ) * scipy.stats.lognorm.cdf(sample_pos_2d[:, 1], scale=np.exp(mean_2d[1]), s=std[1])
    ref_sol[ref_sol == 0] = np.nan  # Queens Log Normal is not defined for <=0.
    np.testing.assert_allclose(lognormal_2d.cdf(sample_pos_2d), ref_sol)


@pytest.mark.unit_tests
def test_draw_lognormal_2d(lognormal_2d, mean_2d, covariance_2d, uncorrelated_vector_2d, mocker):
    """Test the draw method of normal distribution."""
    mocker.patch('numpy.random.randn', return_value=uncorrelated_vector_2d)
    draw = lognormal_2d.draw()
    ref_sol = np.exp(mean_2d + np.dot(np.sqrt(covariance_2d), uncorrelated_vector_2d).T)
    np.testing.assert_equal(draw, ref_sol)


@pytest.mark.unit_tests
def test_logpdf_lognormal_2d(lognormal_2d, mean_2d, covariance_2d, sample_pos_2d):
    """Test pdf method of LogNormal Distribution distribution class."""
    sample_pos_2d = sample_pos_2d.reshape(-1, 2)
    std = np.diag(np.sqrt(covariance_2d))
    ref_sol = scipy.stats.lognorm.logpdf(
        sample_pos_2d[:, 0], scale=np.exp(mean_2d[0]), s=std[0]
    ) + scipy.stats.lognorm.logpdf(sample_pos_2d[:, 1], scale=np.exp(mean_2d[1]), s=std[1])
    ref_sol[ref_sol == -np.inf] = np.nan  # Queens Log Normal is not defined for <=0.
    np.testing.assert_allclose(lognormal_2d.logpdf(sample_pos_2d), ref_sol)


@pytest.mark.unit_tests
def test_pdf_lognormal_2d(lognormal_2d, mean_2d, covariance_2d, sample_pos_2d):
    """Test pdf method of LogNormal Distribution distribution class."""
    sample_pos_2d = sample_pos_2d.reshape(-1, 2)
    std = np.diag(np.sqrt(covariance_2d))
    ref_sol = scipy.stats.lognorm.pdf(
        sample_pos_2d[:, 0], scale=np.exp(mean_2d[0]), s=std[0]
    ) * scipy.stats.lognorm.pdf(sample_pos_2d[:, 1], scale=np.exp(mean_2d[1]), s=std[1])
    ref_sol[ref_sol == 0] = np.nan  # Queens Log Normal is not defined for <=0.
    np.testing.assert_allclose(lognormal_2d.pdf(sample_pos_2d), ref_sol)


@pytest.mark.unit_tests
def test_ppf_lognormal_2d(lognormal_2d, mean_2d, covariance_2d):
    """Test ppf method of LogNormal Distribution distribution class."""
    with pytest.raises(ValueError, match='Method does not support multivariate distributions!'):
        lognormal_2d.ppf(np.zeros(2))


@pytest.mark.unit_tests
def test_init_lognormal_wrong_dimension(mean_2d):
    """Test ValueError of init method of LogNormal Distribution class."""
    covariance = np.array([[[1.0, 0.1], [1.0, 0.1]], [[0.2, 2.0], [0.2, 2.0]]])
    with pytest.raises(ValueError, match=r'Provided covariance is not a matrix.*'):
        distribution_options = {
            'distribution': 'lognormal',
            'normal_mean': np.zeros(3),
            'normal_covariance': covariance,
        }
        from_config_create_distribution(distribution_options)


@pytest.mark.unit_tests
def test_init_lognormal_not_quadratic():
    """Test ValueError of init method of LogNormal Distribution class."""
    covariance = np.array([[1.0, 0.1], [0.2, 2.0], [3.0, 0.3]])
    with pytest.raises(ValueError, match=r'Provided covariance matrix is not quadratic.*'):
        distribution_options = {
            'distribution': 'lognormal',
            'normal_mean': np.zeros(3),
            'normal_covariance': covariance,
        }
        from_config_create_distribution(distribution_options)


@pytest.mark.unit_tests
def test_init_lognormal_not_symmetric():
    """Test ValueError of init method of LogNormal Distribution class."""
    covariance = np.array([[1.0, 0.1], [0.2, 2.0]])
    with pytest.raises(ValueError, match=r'Provided covariance matrix is not symmetric.*'):
        distribution_options = {
            'distribution': 'lognormal',
            'normal_mean': np.zeros(3),
            'normal_covariance': covariance,
        }
        from_config_create_distribution(distribution_options)


@pytest.mark.unit_tests
def test_init_lognormal_not_symmetric():
    """Test ValueError of init method of LogNormal Distribution class."""
    covariance = np.array([[1.0, 0.0], [0.0, 2.0]])
    with pytest.raises(ValueError, match=r'Dimension of mean vector and covariance matrix do not*'):
        distribution_options = {
            'distribution': 'lognormal',
            'normal_mean': np.zeros(3),
            'normal_covariance': covariance,
        }
        from_config_create_distribution(distribution_options)
