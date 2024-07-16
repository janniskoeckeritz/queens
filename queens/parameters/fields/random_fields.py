"""Random fields module."""

import abc

import numpy as np


class RandomField(metaclass=abc.ABCMeta):
    """RandomField meta class.

    Attributes:
            dimension (int): Dimension of the latent space.
            coords (np.ndarray): Coordinates at which the random field is evaluated.
            dim_coords (int): Dimension of the random field (number of coordinates)
            distribution (obj): QUEENS distribution object of latent space variables
    """

    def __init__(self, coords):
        """Initialize random field object.

        Args:
            coords (dict): Dictionary with coordinates of discretized random field and the
                           corresponding keys
        """
        self.coords = coords
        # check if coords are 1D vector
        if np.array(coords["coords"]).ndim == 1:
            self.coords["coords"] = np.array(coords["coords"]).reshape((len(coords["coords"])), 1)

        self.dim_coords = len(coords["keys"])
        self.dimension = None
        self.distribution = None

    @abc.abstractmethod
    def draw(self, num_samples):
        """Draw samples of the latent space.

        Args:
            num_samples (int): Batch size of samples to draw
        """

    @abc.abstractmethod
    def expanded_representation(self, samples):
        """Expand the random field realization.

        Args:
            samples (np.array): Latent space variables to be expanded into a random field
        """

    @abc.abstractmethod
    def logpdf(self, samples):
        """Get joint logpdf of latent space.

        Args:
            samples (np.array): Sample to evaluate logpdf
        """

    @abc.abstractmethod
    def grad_logpdf(self, samples):
        """Get gradient of joint logpdf of latent space.

        Args:
            samples (np.array): Sample to evaluate gradient of logpdf
        """

    def latent_gradient(self, upstream_gradient):
        """Graident of the field with respect to the latent variables."""
        raise NotImplementedError
