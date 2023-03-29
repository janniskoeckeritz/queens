"""Finite difference model."""

import logging

import numpy as np

from pqueens.interfaces import from_config_create_interface
from pqueens.models.model import Model
from pqueens.utils.fd_jacobian import fd_jacobian, get_positions
from pqueens.utils.valid_options_utils import check_if_valid_options

_logger = logging.getLogger(__name__)

VALID_FINITE_DIFFERENCE_METHODS = ["2-point", "3-point"]


class FiniteDifferenceModel(Model):
    """Finite difference model.

    Attributes:
        finite_difference_method (str): Method to calculate a finite difference
                                        based approximation of the Jacobian matrix:
                                         - '2-point': a one-sided scheme by definition
                                         - '3-point': more exact but needs twice as many function
                                                      evaluations
        step_size (float): Step size for the finite difference
                           approximation
        bounds (np.array): Lower and upper bounds on independent variables.
                           Defaults to no bounds meaning: [-inf, inf]
                           Each bound must match the size of *x0* or be a scalar, in the latter case
                           the bound will be the same for all variables. Use it to limit the range
                           of function evaluation.
    """

    def __init__(
        self, model_name, interface, finite_difference_method, step_size=1e-5, bounds=None
    ):
        """Initialize model.

        Args:
            model_name (str): Name of the model
            interface (Interface): Interface object for simulation run
            finite_difference_method (str): Method to calculate a finite difference
                                            based approximation of the Jacobian matrix:
                                             - '2-point': a one-sided scheme by definition
                                             - '3-point': more exact but needs twice as many
                                                          function evaluations
            step_size (float, opt): Step size for the finite difference approximation
            bounds (tuple of array_like, opt): Lower and upper bounds on independent variables.
                                               Defaults to no bounds meaning: [-inf, inf]
                                               Each bound must match the size of *x0* or be a
                                               scalar, in the latter case the bound will be the
                                               same for all variables. Use it to limit the
                                               range of function evaluation.
        """
        super().__init__(model_name)

        check_if_valid_options(VALID_FINITE_DIFFERENCE_METHODS, finite_difference_method)
        self.interface = interface
        self.finite_difference_method = finite_difference_method
        self.step_size = step_size
        _logger.debug(
            "The gradient calculation via finite differences uses a step size of %s.",
            step_size,
        )
        if bounds is None:
            bounds = [-np.inf, np.inf]
        self.bounds = np.array(bounds)

    @classmethod
    def from_config_create_model(
        cls,
        model_name,
        config,
    ):
        """Create Finite difference model from problem description.

        Args:
            model_name (str): Name of the model
            config (dict): Dictionary containing problem description

        Returns:
            instance of FiniteDifferenceModel class
        """
        model_options = config[model_name]
        interface_name = model_options.pop('interface_name')
        interface = from_config_create_interface(interface_name, config)
        model_options.pop('type')

        return cls(model_name=model_name, interface=interface, **model_options)

    def evaluate(self, samples, **kwargs):
        """Evaluate model with current set of samples.

        Args:
            samples (np.ndarray): Evaluated samples

        Returns:
            self.response (np.array): Response of the underlying model at current variables
        """
        if not kwargs.get('gradient', False):
            self.response = self.interface.evaluate(samples)
        else:
            self.response = self.evaluate_finite_differences(samples)
        return self.response

    def grad(self, samples, upstream):
        """Evaluate gradient of model with current set of samples.

        Args:
            samples (np.array): Evaluated samples
            upstream (np.array): Upstream gradient
        """
        return np.sum(upstream[:, :, np.newaxis] * self.response['gradient'], axis=1)

    def evaluate_finite_differences(self, samples):
        """Evaluate model gradient based on FDs.

        Args:
            samples (np.array): Current samples at which model should be evaluated.

        Returns:
            response (np.array): Array with model response for given input samples
            gradient_response (np.array): Array with row-wise model/objective fun gradients for
                                          given samples.
        """
        # check dimensions of samples
        if samples.ndim < 2:
            raise ValueError(
                "The sample dimension must be at least 2D! Columns represent different "
                "variable dimensions and rows different sample realizations."
            )

        num_samples = samples.shape[0]

        # calculate the additional sample points for the stencil per sample
        stencil_samples_lst = []
        delta_positions_lst = []
        for sample in samples:
            stencil_sample, delta_positions = get_positions(
                sample,
                method=self.finite_difference_method,
                rel_step=self.step_size,
                bounds=self.bounds,
            )
            stencil_samples_lst.append(stencil_sample)
            delta_positions_lst.append(delta_positions)

        num_stencil_points_per_sample = stencil_sample.shape[1]
        stencil_samples = np.array(stencil_samples_lst).reshape(-1, num_stencil_points_per_sample)

        # stack samples and stencil points and evaluate entire batch
        combined_samples = np.vstack((samples, stencil_samples))
        all_responses = self.interface.evaluate(combined_samples)['mean'].reshape(
            combined_samples.shape[0], -1
        )

        response = all_responses[:num_samples, :]
        additional_response_lst = np.array_split(all_responses[num_samples:, :], num_samples)

        # calculate the model gradients re-using the already computed model responses
        model_gradients_lst = []
        for output, delta_positions, additional_model_output_stencil in zip(
            response, delta_positions_lst, additional_response_lst
        ):
            model_gradients_lst.append(
                fd_jacobian(
                    output.reshape(1, -1),
                    additional_model_output_stencil,
                    delta_positions,
                    False,
                    method=self.finite_difference_method,
                ).reshape(output.size, -1)
            )

        gradient_response = np.array(model_gradients_lst)

        return {'mean': response, 'gradient': gradient_response}
