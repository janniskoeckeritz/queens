"""Mixture distribution."""
import logging

import numpy as np
from scipy.special import logsumexp

from pqueens.distributions import from_config_create_distribution as fcc
from pqueens.distributions.distributions import Distribution

_logger = logging.getLogger(__name__)


class MixtureDistribution(Distribution):
    """Mixture models."""

    def __init__(self, weights, component_distributions):
        """Initialize mixture model.

        Args:
            weights (np.ndarray): weights of the mixtures
            component_distributions (list): component distributions of the mixture
        """
        if len(component_distributions) != len(weights):
            raise ValueError(
                f"The number of weights {len(weights)} does not match the number of distribution"
                f" {len(component_distributions)}"
            )

        super().check_positivity(weights=weights)

        if np.sum(weights) != 1:
            _logger.info("Weights do not sum up to one, they are going to be normalized.")
            weights /= np.sum(weights)

        self.component_distributions = list(component_distributions)
        self.weights = weights
        self.number_of_components = len(weights)

        if len({d.dimension for d in component_distributions}) != 1:
            raise ValueError("Dimensions of the component distributions do not match")

        mean, covariance = self._compute_mean_and_covariance(weights, component_distributions)
        super().__init__(mean, covariance, component_distributions[0].dimension)

    @classmethod
    def from_config_create_distribution(cls, distribution_options):
        """Create mixture model from config.

        Args:
            distribution_options (dict): description of the distribution

        Returns:
            MixtureDistribution: mixture model
        """
        distribution_options_copy = distribution_options.copy()
        distribution_options_copy.pop("type")
        component_keys = distribution_options_copy.pop("component_distributions")
        component_distributions = []
        for key in component_keys:
            component_options = distribution_options_copy.pop(key)
            component_distributions.append(fcc(component_options))
        distribution_options_copy["component_distributions"] = component_distributions
        return cls(**distribution_options_copy)

    @staticmethod
    def _compute_mean_and_covariance(weights, component_distributions):
        """Compute the mean value and covariance of the mixture model.

        Args:
            weights (np.ndarray): Weights of the mixture
            component_distributions (obj): Components of the mixture

        Returns:
            mean (np.ndarray): mean value of the mixture
            covariance (np.ndarray): covariance of the mixture
        """
        mean = 0
        covariance = 0
        for weight, component in zip(weights, component_distributions, strict=True):
            mean += weight * component.mean
            covariance += weight * (component.covariance + component.mean**2)
        covariance -= mean**2
        return mean, covariance

    def draw(self, num_draws=1):
        """Draw *num_draw* samples from the variational distribution.

        Uses a two step process:
            1. From a multinomial distribution, based on the weights, select a component
            2. Sample from the selected component

        Args:
            num_draws (int): Number of samples to draw

        Returns:
            samples (np.ndarray): Row wise samples of the variational distribution
        """
        components = np.random.multinomial(num_draws, self.weights)
        samples = []
        for component, num_draw_component in enumerate(components):
            sample = self.component_distributions[component].draw(num_draw_component)
            samples.append(sample)
        samples = np.concatenate(samples, axis=0)

        # Strictly speaking this is not necessary, however, without it, if you only select x
        # samples, so `samples[:x]`, most samples would originate from the first components and this
        # would be biased
        np.random.shuffle(samples)

        return samples

    def cdf(self, x):
        """Cumulative distribution function.

        Args:
            x (np.ndarray): Positions at which the cdf is evaluated

        Returns:
            np.ndarray: CDF of the mixture model
        """
        cdf = 0
        for weights, component in zip(self.weights, self.component_distributions, strict=True):
            cdf += weights * component.cdf(x)
        return cdf

    def logpdf(self, x):
        """Log of the probability density function.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated
        """
        log_weights = np.log(self.weights)
        weighted_logpdf = []
        for sample in x:
            sample_logpdf = []
            for log_weight, component in zip(
                log_weights, self.component_distributions, strict=True
            ):
                sample_logpdf.append(log_weight + component.logpdf(sample))
            weighted_logpdf.append(sample_logpdf)

        logpdf = logsumexp(weighted_logpdf, axis=1).flatten()

        return logpdf

    def pdf(self, x):
        """Probability density function.

        Args:
            x (np.ndarray): Positions at which the pdf is evaluated
        """
        return np.exp(self.logpdf(x))

    def grad_logpdf(self, x):
        """Gradient of the log pdf with respect to *x*.

        Args:
            x (np.ndarray): Positions at which the gradient of log pdf is evaluated
        """
        responsibilities = self.responsibilities(x)

        grad_logpdf = 0
        for responsibility, component in zip(
            responsibilities.T, self.component_distributions, strict=True
        ):
            grad_logpdf += responsibility.reshape(-1, 1) * component.grad_logpdf(x)

        return np.array(grad_logpdf).reshape(len(x), -1)

    def ppf(self, _):
        """Percent point function (inverse of cdf — quantiles).

        Args:
            q (np.ndarray): Quantiles at which the ppf is evaluated
        """
        raise NotImplementedError("PPF not available for mixture models.")

    def responsibilities(self, x):  # pylint: disable=C0103
        r"""Compute the responsibilities.

        The responsibilities are defined as [1]:

        :math: `\gamma_j(x)=\frac{w_j p_j(x)}{\sum_{i=0}^{n_{components}-1}w_i p_i(x)}`

        [1]: Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.

        Args:
            x (np.ndarray): Positions at which the responsibilities are evaluated

        Returns:
            np.ndarray: responsibilities (number of samples x number of component)
        """
        log_weights = np.log(self.weights)
        inv_log_responsibility = []
        for sample in x:
            data_sample = []
            for log_weight_i, component_i in zip(
                log_weights, self.component_distributions, strict=True
            ):
                data_component_i = []
                for log_weight_j, component_j in zip(
                    log_weights, self.component_distributions, strict=True
                ):
                    log_ratio = (
                        -log_weight_i
                        - component_i.logpdf(sample)
                        + log_weight_j
                        + component_j.logpdf(sample)
                    )
                    data_component_i.append(log_ratio)
                data_sample.append(logsumexp(data_component_i))
            inv_log_responsibility.append(data_sample)

        return np.exp(-np.array(inv_log_responsibility))

    def export_dict(self):
        """Create a dict of the distribution.

        Returns:
            export_dict (dict): Dict containing distribution information
        """
        dictionary = super().export_dict()
        dictionary.pop("component_distributions")
        for i, components in enumerate(self.component_distributions):
            dictionary.update({f"component_{i}": components.export_dict()})
        return dictionary
