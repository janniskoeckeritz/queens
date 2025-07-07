#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Adaptive sampling iterator."""

import logging
import pickle
import types

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from queens.iterators._iterator import Iterator
from queens.iterators.sequential_monte_carlo_chopin import SequentialMonteCarloChopin
from queens.utils.io import load_result
from queens.models.logpdf_gp import LogpdfGP

_logger = logging.getLogger(__name__)
jax.config.update("jax_enable_x64", True)


class AdaptiveSampling(Iterator):
    """Adaptive sampling iterator.

    Attributes:
        likelihood_model (Model): Likelihood model (Only Gaussian Likelihood supported)
        solving_iterator (Iterator): Iterator to solve inverse problem
                                     (only SequentialMonteCarloChopin works out of the box)
        num_new_samples (int): Number of new training samples in each adaptive step
        num_steps (int): Number of adaptive sampling steps
        seed (int, opt): Seed for random number generation
        restart_file (str, opt): Result file path for restarts
        cs_div_criterion (float): Cauchy-Schwarz divergence stopping criterion threshold
        x_train (np.ndarray): Training input samples
        x_train_new (np.ndarray): Newly drawn training samples
        y_train (np.ndarray): Training likelihood output samples
        model_outputs (np.ndarray): Training model output samples
    """

    def __init__(
        self,
        model,
        parameters,
        global_settings,
        likelihood_model,
        initial_train_samples,
        solving_iterator,
        num_new_samples,
        num_steps,
        seed=41,
        restart_file=None,
        cs_div_criterion=0.01,
        verbose=False,
    ):
        """Initialise AdaptiveSampling.

        Args:
            model (Model): Model to be evaluated by iterator.
            parameters (Parameters): Parameters object.
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory.
            likelihood_model (Model): Likelihood model (Only Gaussian Likelihood supported).
            initial_train_samples (np.ndarray): Initial training samples for surrogate model.
            solving_iterator (Iterator): Iterator to solve inverse problem
                                         (only SequentialMonteCarloChopin works out of the box).
            num_new_samples (int): Number of new training samples in each adaptive step.
            num_steps (int): Number of adaptive sampling steps.
            seed (int, opt): Seed for random number generation.
            restart_file (str, opt): Result file path for restarts.
            cs_div_criterion (float): Cauchy-Schwarz divergence stopping criterion threshold.
        """
        super().__init__(model, parameters, global_settings)
        self.seed = seed
        self.likelihood_model = likelihood_model
        self.solving_iterator = solving_iterator
        self.num_new_samples = num_new_samples
        self.num_steps = num_steps
        self.restart_file = restart_file
        self.cs_div_criterion = cs_div_criterion
        self.num_dim = initial_train_samples.shape[1]
        self.verbose = verbose

        self.use_grad_obs = False
        if isinstance(model, LogpdfGP):
            self.use_grad_obs = model.use_grad_obs

        self.x_train_new = initial_train_samples
        self.x_train = np.empty((0, self.parameters.num_parameters))
        self.y_train = np.empty((0, 1))
        self.model_outputs = np.empty((0, self.likelihood_model.y_obs.size))
        self.log_likelihoods = np.empty((0, 1))
        self.y_train_grad = np.empty((0, self.num_dim))
        self.model_gradients = np.empty((0, self.likelihood_model.y_obs.size*self.num_dim))
        self.log_likelihoods_grad = np.empty((0, self.num_dim))

        if self.verbose:
            self.write_results = self.write_results_verbose
        else:
            self.write_results = self.write_results_minimal

    def pre_run(self):
        """Pre run."""
        np.random.seed(self.seed)

        if self.restart_file:
            results = load_result(self.restart_file)
            self.x_train = results["x_train"][-1]
            self.model_outputs = results["model_outputs"][-1]
            self.y_train = results["y_train"][-1]
            self.x_train_new = results["x_train_new"][-1]
            if self.use_grad_obs:
                self.y_train_grad = results["y_train_grad"][-1]
                self.model_gradients = results["model_gradients"][-1]

    def core_run(self):
        """Core run."""
        for i in range(self.num_steps):
            _logger.info("Step: %i / %i", i + 1, self.num_steps)
            self.x_train = np.concatenate([self.x_train, self.x_train_new], axis=0)

            # self.y_train_grad is empty if use_grad_obs is False
            self.y_train, self.y_train_grad = self.eval_log_likelihood()

            _logger.info("Number of solver evaluations: %i", self.x_train.shape[0])
            self.model.initialize(
                self.x_train,
                self.y_train,
                self.likelihood_model.y_obs.size,
                self.y_train_grad,
            )

            random_state = np.random.get_state()
            self.solving_iterator.pre_run()  # We don't want that the random seed is set here.
            np.random.set_state(random_state)

            def _m(self_, _, xp):
                x_train_ml = self.x_train[np.argmax(self.y_train[:, 0])]
                epn = xp.shared["exponents"][-1]
                target = self_.current_target(epn)
                particles = np.lib.recfunctions.structured_to_unstructured(xp.theta)
                if not (particles == x_train_ml).all(-1).any():
                    for j, par in enumerate(xp.theta.dtype.names):
                        xp.theta[par][0] = x_train_ml[j]
                    target(xp)
                return self_.move(xp, target)

            if isinstance(self.solving_iterator, SequentialMonteCarloChopin):
                self.solving_iterator.smc_obj.fk.M = types.MethodType(
                    _m, self.solving_iterator.smc_obj.fk
                )

            self.solving_iterator.core_run()

            particles, weights, log_posterior = self.solving_iterator.get_particles_and_weights()
            self.x_train_new = self.choose_new_samples(particles, weights)

            cs_div = self.write_results(particles, weights, log_posterior, i)

            if cs_div < self.cs_div_criterion:
                break

    def eval_log_likelihood(self):
        """Evaluate log likelihood.

        Returns:
            log_likelihood (np.ndarray): Log likelihood
        """
        if self.use_grad_obs:
            # Evaluate likelihood model with gradients
            result = self.likelihood_model.evaluate_and_gradient(self.x_train_new)
            log_likelihood, log_likelihood_grad = result

            log_likelihood -= self.likelihood_model.normal_distribution.logpdf_const
            log_likelihood = log_likelihood.reshape((-1, 1))

            # Update stored values
            self.log_likelihoods = np.concatenate(
                [self.log_likelihoods, log_likelihood], axis=0
            )
            self.log_likelihoods_grad = np.concatenate(
                [self.log_likelihoods_grad, log_likelihood_grad], axis=0
            )
        else:
            # Evaluate likelihood model without gradients
            log_likelihood = self.likelihood_model.evaluate(self.x_train_new)["result"]
            log_likelihood -= self.likelihood_model.normal_distribution.logpdf_const
            log_likelihood = log_likelihood.reshape((-1, 1))

            # Update stored values
            self.log_likelihoods = np.concatenate(
                [self.log_likelihoods, log_likelihood], axis=0
            )

        return self.log_likelihoods, self.log_likelihoods_grad

    def choose_new_samples(self, particles, weights):
        """Choose new training samples.

        Choose new training samples from approximated posterior distribution.

        Args:
            particles (np.ndarray): Unique particles of approximated posterior.
            weights (np.ndarray): Unique non-zero particle weights of approximated posterior.

        Returns:
            x_train_new (np.ndarray): New training samples
        """
        # Filter particles, that are present in training sample set
        indices = (particles[:, np.newaxis] == self.x_train).all(-1).any(-1)
        particles = particles[~indices]
        weights = weights[~indices]
        weights /= np.sum(weights)

        if len(weights) == 0:
            _logger.warning(
                "Adaptive sampling of new training samples failed. "
                "Drawing new training samples from prior..."
            )
            return self.parameters.draw_samples(self.num_new_samples)

        num_adaptive_samples = min(len(weights), self.num_new_samples)
        indices = np.random.choice(
            np.arange(len(weights)), num_adaptive_samples, p=weights, replace=False
        )
        x_train_new = particles[indices]

        if num_adaptive_samples < self.num_new_samples:
            num_prior_samples = self.num_new_samples - num_adaptive_samples
            _logger.warning(
                "Adaptive sampling of new training samples partly failed. "
                "Drawing %i new training samples from prior...",
                num_prior_samples,
            )
            prior_samples = self.parameters.draw_samples(num_prior_samples)
            x_train_new = np.concatenate([x_train_new, prior_samples], axis=0)

        return x_train_new

    def write_results_verbose(self, particles, weights, log_posterior, iteration):
        """Write results to output file and calculate cs_div.

        Args:
            particles (np.ndarray): Particles of approximated posterior
            weights (np.ndarray): Particle weights of approximated posterior
            log_posterior (np.ndarray): Log posterior value of particles
            iteration (int): Iteration count

        Returns:
            cs_div (float): Maximum Cauchy-Schwarz divergence between marginals of the current and
                            previous step
        """
        result_file = self.global_settings.result_file(".pickle")

        if iteration == 0 and not self.restart_file:
            results = {
                "x_train": [],
                "model_outputs": [],
                "y_train": [],
                "x_train_new": [],
                "particles": [],
                "weights": [],
                "log_posterior": [],
                "cs_div": [],
            }
            cs_div = np.nan
        else:
            results = load_result(result_file)
            particles_prev = results["particles"][-1]
            weights_prev = results["weights"][-1]
            samples_prev = particles_prev[
                np.random.choice(np.arange(weights_prev.size), 5_000, p=weights_prev)
            ]
            samples_curr = particles[np.random.choice(np.arange(weights.size), 5_000, p=weights)]
            cs_div = float(cauchy_schwarz_divergence(samples_prev, samples_curr))
            _logger.info("Cauchy Schwarz divergence: %.2e", cs_div)

        results["x_train"].append(self.x_train)
        results["model_outputs"].append(self.model_outputs)
        results["y_train"].append(self.y_train)
        results["x_train_new"].append(self.x_train_new)
        results["particles"].append(particles)
        results["weights"].append(weights)
        results["log_posterior"].append(log_posterior)
        results["cs_div"].append(cs_div)

        with open(result_file, "wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return cs_div
    
    def write_results_minimal(self, particles, weights, log_posterior, iteration):
        result_file = self.global_settings.result_file(".pickle")

        if iteration == 0 and not self.restart_file:
            results = {
                "x_train": [],
                "log_likelihoods"
                "particles": [],
                "weights": [],
                "cs_div": [],
            }
            if self.use_grad_obs:
                results['log_likelihoods_grad'] = []
            cs_div = np.nan
        else:
            results = load_result(result_file)
            particles_prev = results["particles"]
            weights_prev = results["weights"]
            samples_prev = particles_prev[
                np.random.choice(np.arange(weights_prev.size), 5_000, p=weights_prev)
            ]
            samples_curr = particles[np.random.choice(np.arange(weights.size), 5_000, p=weights)]
            cs_div = float(cauchy_schwarz_divergence(samples_prev, samples_curr))
            _logger.info("Cauchy Schwarz divergence: %.2e", cs_div)

        results["x_train"] = self.x_train
        results["log_likelihoods"] = self.log_likelihoods
        results["particles"] = particles
        results["weights"] = weights
        results["cs_div"].append(cs_div)
        if self.use_grad_obs:
            results["log_likelihoods_grad"] = self.log_likelihoods_grad

        with open(result_file, "wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return cs_div

    def post_run(self):
        """Post run."""


@jit
def cauchy_schwarz_divergence(samples_1, samples_2):
    """Maximum Cauchy-Schwarz divergence between marginals of two sample sets.

    Args:
        samples_1 (np.ndarray): Sample set 1
        samples_2 (np.ndarray): Sample set 2

    Returns:
        cs_div_max (np.ndarray): Maximum Cauchy-Schwarz divergence between marginals of two sample
                                 sets.
    """
    n_1 = samples_1.shape[0]
    n_2 = samples_2.shape[0]

    factor_1 = n_1 ** (-1.0 / 5)
    factor_2 = n_2 ** (-1.0 / 5)

    var_1 = jnp.var(samples_1, axis=0) * factor_1**2
    var_2 = jnp.var(samples_2, axis=0) * factor_2**2

    def normalizing_factor(variance):
        return (2 * jnp.pi * variance) ** (-1 / 2)

    def normal(x_1, x_2, variance):
        d = x_1[:, jnp.newaxis, :] - x_2[jnp.newaxis, :, :]
        norm = d**2 / variance
        return normalizing_factor(variance), -0.5 * norm

    z_1_2 = normal(samples_1, samples_2, var_1 + var_2)
    z_1_1 = normal(samples_1, samples_1, var_1 + var_1)
    z_1_1 = z_1_1[0] * jnp.exp(z_1_1[1])
    z_2_2 = normal(samples_2, samples_2, var_2 + var_2)
    z_2_2 = z_2_2[0] * jnp.exp(z_2_2[1])

    max_1_2 = jnp.max(z_1_2[1], axis=(0, 1))
    term_1 = (
        -jnp.log(jnp.sum(1 / n_1 * 1 / n_2 * z_1_2[0] * jnp.exp(z_1_2[1] - max_1_2), axis=(0, 1)))
        - max_1_2
    )
    term_2 = 0.5 * jnp.log(
        1 / n_1 * normalizing_factor(var_1)
        + jnp.sum(
            2 / n_1**2 * z_1_1 * jnp.tri(*z_1_1.shape[:2], k=-1)[:, :, jnp.newaxis], axis=(0, 1)
        )
    )
    term_3 = 0.5 * jnp.log(
        1 / n_2 * normalizing_factor(var_2)
        + jnp.sum(
            2 / n_2**2 * z_2_2 * jnp.tri(*z_1_1.shape[:2], k=-1)[:, :, jnp.newaxis], axis=(0, 1)
        )
    )
    cs_div_max = jnp.max(term_1 + term_2 + term_3)
    return cs_div_max
