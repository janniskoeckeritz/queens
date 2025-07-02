"""Kernel functions for Gaussian Processes in JAX."""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


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

def not_yet_implemented(*args, **kwargs):
    raise NotImplementedError("Gradient computation is not implemented for this kernel.")

def kernel_fn_rbf(x1, x2, hyperparams):
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

    length_scales = hyperparams['length_scales']
    signal_std = hyperparams['signal_std']
    noise = hyperparams['noise']

    # Compute the squared exponential kernel
    length_scale_operator = jnp.diag(1 / length_scales**2)
    square_dists = jnp.dot((x1 - x2).T, jnp.dot(length_scale_operator, (x1 - x2)))
    return signal_std**2 * jnp.exp(-0.5 * square_dists)


class AbstractKernel:
    """Abstract base class for kernel functions in Gaussian Processes.

    This class is designed to be subclassed with a specific function that constructs a covariance 
    matrix function from a kernel function. A covariance matrix function computes the covariance 
    between two sets of input points.

    Args:
        kernel_fn (callable): The kernel function that computes the covariance between two points  
                              x1 and x2.
        compute_cov_matrix (callable): A function that computes the covariance matrix for given 
                                       inputs.
    Methods:
        kernel_fn_to_cov_mat_fn(kernel_fn): Converts a kernel function to a covariance matrix 
                                            computation function. This method is intended to be 
                                            overridden in subclasses to provide specific kernel 
                                            functionality.
    """
    def __init__(self, kernel_fn, use_noise=True):
        self.kernel_fn = kernel_fn
        self.compute_full_cov_matrix = self.kernel_fn_to_full_cov_mat_fn(
            self.noise_wrapper(kernel_fn)
        )
        self.compute_val_cov_matrix = self.kernel_fn_to_val_cov_mat_fn(kernel_fn)
        self.compute_grad_cov_matrix = self.kernel_fn_to_grad_cov_mat_fn(kernel_fn)

    @staticmethod
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
            noise = args[2]['noise']
            return result + kronecker_delta(args[0], args[1]) * noise
        return wrapper
    

    @staticmethod
    def kernel_fn_to_full_cov_mat_fn(kernel_fn):
        """
        Converts a kernel function to a full covariance matrix computation function.
        
        Args:
            kernel_fn (callable): The kernel function that computes the covariance between two 
                                  inputs.
        
        Returns:
            callable: A function that computes the covariance matrix for given inputs.
        """
        raise NotImplementedError("This method should be overridden in subclasses.")
    
    @staticmethod
    def kernel_fn_to_val_cov_mat_fn(kernel_fn):
        """
        Converts a kernel function to a covariance matrix computation function for the value 
        evaluation of a Gaussian Process. (as opposed to gradient evaluation)
        
        Args:
            kernel_fn (callable): The kernel function that computes the covariance between two 
                                  inputs.
        
        Returns:
            callable: A function that computes the covariance matrix for the value evaluation of a
                      Gaussian Process.
        """
        raise NotImplementedError("This method should be overridden in subclasses.")
    
    @staticmethod
    def kernel_fn_to_grad_cov_mat_fn(kernel_fn):
        """
        Converts a kernel function to a covariance matrix computation function for the gradient
        evaluation of a Gaussian Process.

        Args:
            kernel_fn (callable): The kernel function that computes the covariance between two 
                                  inputs.
        Returns:
            callable: A function that computes the covariance matrix for the gradient evaluation of 
                      a Gaussian Process.
        """
        raise NotImplementedError("This method should be overridden in subclasses.")


class Kernel(AbstractKernel):
    """Kernel class for any kernel without gradient information."""
    use_gradients = False

    def __init__(self, kernel_fn, use_noise=True):
        """
        Initializes the Kernel class with a kernel function.
        
        Args:
            kernel_fn (callable): The kernel function that computes the covariance between two 
                                  inputs.
        """
        self.kernel_fn_to_val_cov_mat_fn = self.kernel_fn_to_full_cov_mat_fn
        self.kernel_fn_to_grad_cov_mat_fn = lambda func: not_yet_implemented
        super().__init__(kernel_fn, use_noise)

    @staticmethod
    def kernel_fn_to_full_cov_mat_fn(kernel_fn):
        """
        Converts a kernel function to a covariance matrix computation function.
        
        Args:
            kernel_fn (callable): The kernel function that computes the covariance between two 
                                  inputs.
        
        Returns:
            callable: A function that computes the covariance matrix for given inputs.
        """
        cov = jax.vmap(kernel_fn, in_axes=(0, None, None), out_axes=0)
        cov = jax.vmap(cov, in_axes=(None, 0, None), out_axes=-1)
        return jax.jit(cov)
    

class GradKernel(AbstractKernel):
    """Gradient Kernel class for kernels with gradient information."""
    use_gradients = True

    def __init__(self, kernel_fn, use_noise=True):
        """
        Initializes the GradKernel class with a kernel function.
        
        Args:
            kernel_fn (callable): The kernel function that computes the covariance between two 
                                  inputs.
        """
        self.kernel_fn_to_grad_cov_mat_fn = lambda func: not_yet_implemented
        super().__init__(kernel_fn, use_noise)

    @staticmethod
    def kernel_fn_to_full_cov_mat_fn(kernel_fn):
        """
        Converts a kernel function to a covariance matrix computation function with gradients.
        
        Args:
            kernel_fn (callable): The kernel function that computes the covariance between two 
                                  inputs.
        
        Returns:
            callable: A function that computes the covariance matrix and its gradients for given    
                      inputs.
        """
        cov = jax.vmap(kernel_fn, in_axes=(0, None, None), out_axes=0)
        cov = jax.vmap(cov, in_axes=(None, 0, None), out_axes=-1)

        grad_cov = jax.grad(kernel_fn, argnums=0)
        grad_cov = flatten_output(jax.vmap(grad_cov, in_axes=(0, None, None), out_axes=0))
        grad_cov = jax.vmap(grad_cov, in_axes=(None, 0, None), out_axes=-1)

        cov_grad = jax.grad(kernel_fn, argnums=1)
        cov_grad = flatten_output(jax.vmap(cov_grad, in_axes=(None, 0, None), out_axes=0))
        cov_grad = jax.vmap(cov_grad, in_axes=(0, None, None), out_axes=0)

        cov_grad_cov = jax.grad(kernel_fn, argnums=0)
        cov_grad_cov = flatten_output(jax.vmap(cov_grad_cov, in_axes=(0, None, None), out_axes=0))
        cov_grad_cov = jax.jacobian(cov_grad_cov, argnums=1)
        cov_grad_cov = flatten_tail(jax.vmap(cov_grad_cov, in_axes=(None, 0, None), out_axes=-2))

        def covariance_mat_from_grad(x1_batch, x2_batch, hypers):
            """
            Computes the Gram matrix and its gradients for the given inputs and hyperparameters.

            Args:
                x1_batch (jnp.ndarray): First input array.
                x2_batch (jnp.nparray): Second input array.
                hypers (dict): Hyperparameters for the kernel function.

            Returns:
                callable: A function that computes the covariance matrix using gradient information.
            """
            mat11 = cov(x1_batch, x2_batch, hypers)
            mat21 = grad_cov(x1_batch, x2_batch, hypers)
            mat12 = cov_grad(x1_batch, x2_batch, hypers)
            mat22 = cov_grad_cov(x1_batch, x2_batch, hypers)

            cov_mat = jnp.vstack(
                [jnp.hstack([mat11, mat12]),
                jnp.hstack([mat21, mat22])]
            )

            return cov_mat

        return jax.jit(covariance_mat_from_grad)

    def kernel_fn_to_val_cov_mat_fn(self, kernel_fn):
        """
        Converts a kernel function to a covariance matrix computation function for value evaluation.
        
        Args:
            kernel_fn (callable): The kernel function that computes the covariance between two 
                                  inputs.
        
        Returns:
            callable: A function that computes the covariance matrix for value evaluation.
        """
        cov = jax.vmap(kernel_fn, in_axes=(0, None, None), out_axes=0)
        cov = jax.vmap(cov, in_axes=(None, 0, None), out_axes=-1)

        cov_grad = jax.grad(kernel_fn, argnums=1)
        cov_grad = flatten_output(jax.vmap(cov_grad, in_axes=(None, 0, None), out_axes=0))
        cov_grad = jax.vmap(cov_grad, in_axes=(0, None, None), out_axes=0)

        def covariance_val_mat_from_grad(x1_batch, x2_batch, hypers):
            """
            Computes the Gram matrix for value evaluation using gradient information.

            Args:
                x1_batch (jnp.ndarray): First input array.
                x2_batch (jnp.ndarray): Second input array.
                hypers (dict): Hyperparameters for the kernel function.

            Returns:
                jnp.ndarray: The computed covariance matrix for value evaluation.
            """
            mat11 = cov(x1_batch, x2_batch, hypers)
            mat12 = cov_grad(x1_batch, x2_batch, hypers)

            cov_mat = jnp.hstack([mat11, mat12])
            return cov_mat

        return jax.jit(covariance_val_mat_from_grad)


def main():
    # Example hyperparameters for the kernel function
    hyperparameters = {
        'length_scales': jnp.array([2.25424250, 3.93871191]),
        'signal_std': 0.610341868,
        'noise': 1e-6,  # Small noise term to avoid numerical issues
    }

    x_train = jnp.array([[1.0, 1.0], [-1.0, -1.0]])

    kernel = GradKernel(kernel_fn_rbf)
    cov_matrix = kernel.compute_cov_matrix(x_train, x_train, hyperparameters)
    print("Covariance Matrix:\n", cov_matrix)
if __name__ == "__main__":
    main()
