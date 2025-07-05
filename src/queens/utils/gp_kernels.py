import jax.numpy as jnp
import jax
from abc import ABC, abstractmethod

def distances(x1, x2):
    """Distance Matrix between two sample sets.

    Args:
         x1 (np.ndarray): input samples
         x2 (np.ndarray): input samples

    Returns:
         dists (np.ndarray): Distance Matrix
    """
    num_dim = x1.shape[1]
    dists = x1.reshape(-1, 1, num_dim) - x2.reshape(1, -1, num_dim)
    return dists


class AbstractKernel(ABC):
    """Abstract Kernel class."""
    @property
    @abstractmethod
    def use_grad_obs(self):
        """Must be implemented in a subclass to return true or false."""
        pass

    @staticmethod
    @abstractmethod
    def gram(points, hyperparameters):
        """Compute the Gram matrix."""
        raise NotImplementedError("This method should be implemented in a subclass.")

    @staticmethod
    @abstractmethod
    def cross(points1, points2, hyperparameters):
        """Compute the cross-covariance matrix."""
        raise NotImplementedError("This method should be implemented in a subclass.")

    @staticmethod
    def cross_gradient(points1, points2, hyperparameters):
        """Compute the cross-covariance matrix for gradient evaluations."""
        raise NotImplementedError("This method should be implemented in a subclass.")


def squared_exponential_point_point(points1, points2, hyperparameters):
    """Compute the squared exponential kernel for point-point evaluations."""
    lengthscales = hyperparameters[:-2]
    signal_std = hyperparameters[-2]

    dists = distances(points1, points2)
    dist_norm = jnp.sum(dists**2 / (2 * lengthscales**2), axis=2)

    return signal_std**2 * jnp.exp(-dist_norm)

def squared_exponential_grad_point(points1, points2, hyperparameters):
    """Compute the squared exponential kernel for gradient-point evaluations."""
    lengthscales = hyperparameters[:-2]

    dists = distances(points1, points2)
    k_x1_x2 = squared_exponential_point_point(points1, points2, hyperparameters)

    length_adjusted_dists = jnp.einsum("ijk,k->ijk", dists, 1/(lengthscales**2))
    k_x1_x2_grad = jnp.einsum("ij,ijk->ijk", k_x1_x2, length_adjusted_dists)
    k_x1_x2_grad = k_x1_x2_grad.reshape((-1, points2.shape[0]))

    return k_x1_x2_grad

def squared_exponential_grad_grad():
    """Compute the squared exponential kernel for gradient evaluations."""
    raise NotImplementedError("This method should be implemented in a subclass.")

class RbfKernel(AbstractKernel):
    """RBF Kernel Class.

    This class implements the RBF kernel for point-point and point-gradient evaluations.
    """
    @property
    def use_grad_obs(self):
        """Set whether to use gradient observations."""
        return False

    @staticmethod
    def gram(points, hyperparameters):
        """Compute the Gram matrix for RBF kernel."""
        num_points = points.shape[0]
        gram_mat = squared_exponential_point_point(points, points, hyperparameters)
        gram_mat += jnp.eye(num_points) * hyperparameters[-1]  # Add noise term
        return gram_mat

    @staticmethod
    def cross(points1, points2, hyperparameters):
        """Compute the cross-covariance matrix for RBF kernel."""
        return squared_exponential_point_point(points1, points2, hyperparameters)

    @staticmethod
    def cross_gradient(points1, points2, hyperparameters):
        """Compute the cross-covariance matrix for gradient evaluations."""
        return squared_exponential_grad_point(points1, points2, hyperparameters)


def flatten_output(func):
    """Takes a function and returns a new function that flattens the output."""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return jnp.ravel(result)
    return wrapper

def flatten_tail(func):
    """Takes a function and returns a new function that flattens all but the first axes of the output."""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return jnp.reshape(result, (result.shape[0], -1))
    return wrapper

def kronecker_delta(x1, x2):
    """Kronecker delta function."""
    return jnp.where(jnp.all(x1 == x2, axis=-1), 1.0, 0.0)

def noise_wrapper(func):
    """
    Wraps a kernel function to include noise in the covariance matrix.
    
    Args:
        func (callable): The kernel function that computes the covariance between two inputs.
    
    Returns:
        callable: A function that computes the covariance matrix with added noise.
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        noise = args[2][-1]
        return result + kronecker_delta(args[0], args[1]) * noise
    return wrapper

class JaxGeneralKernel(AbstractKernel):
    """JaxGeneralKernel Class.

    This class implements a general kernel using JAX for point-point and point-gradient evaluations.
    """
    @property
    def use_grad_obs(self):
        """Set whether to use gradient observations."""
        return False

    def __init__(self, kernel_fn):
        """Initialize JaxGeneralKernel.

        Args:
            kernel_fn (callable): Function to compute the kernel matrix.
        """
        noisy_kernel_fn = noise_wrapper(kernel_fn)
    
        # gram matrix function
        gram = jax.vmap(noisy_kernel_fn, in_axes=(0, None, None), out_axes=0)
        gram = jax.jit(jax.vmap(gram, in_axes=(None, 0, None), out_axes=-1))
        self.gram = lambda points, hypers: gram(points, points, hypers)

        # cross covariance function
        cross = jax.vmap(kernel_fn, in_axes=(0, None, None), out_axes=0)
        self.cross = jax.jit(jax.vmap(cross, in_axes=(None, 0, None), out_axes=-1))
    
    @staticmethod
    def gram(points, hyperparameters):
        """Compute the Gram matrix."""
        raise NotImplementedError(
            "This method is ovewritten in the constructor of JaxGeneralKernel.")

    @staticmethod
    def cross(points1, points2, hyperparameters):
        """Compute the cross-covariance matrix."""
        raise NotImplementedError("This method should be implemented in a subclass.")

    @staticmethod
    def cross_gradient(points1, points2, hyperparameters):
        """Compute the cross-covariance matrix for gradient evaluations."""
        raise NotImplementedError("This method should be implemented in a subclass.")


def jax_kernel_fn_rbf(x1, x2, hyperparams):
    """
    A simple kernel function that computes the squared exponential kernel.

    Args:
        x1 (jnp.ndarray): First input array.
        x2 (jnp.ndarray): Second input array.
        hypers (dict): Hyperparameters for the kernel function, including:
                        - 'length_scales': jnp.ndarray, length scales for the kernel.
                        - 'signal_std': float, standard deviation of the signal.
                        - 'noise': float, noise term.

    Returns:
        jnp.ndarray: The computed kernel value.
    """

    length_scales = hyperparams[:-2]
    signal_std = hyperparams[-2]
    #noise = hyperparams[-1]

    # Compute the squared exponential kernel
    length_scale_operator = jnp.diag(1 / length_scales**2)
    square_dists = jnp.dot((x1 - x2).T, jnp.dot(length_scale_operator, (x1 - x2)))
    return signal_std**2 * jnp.exp(-0.5 * square_dists)