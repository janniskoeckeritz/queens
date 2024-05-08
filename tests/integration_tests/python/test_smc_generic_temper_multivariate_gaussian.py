"""TODO_doc."""

import pickle

import numpy as np
import pandas as pd
import pytest

from queens.distributions.normal import NormalDistribution
from queens.example_simulator_functions.gaussian_logpdf import GAUSSIAN_4D, gaussian_4d_logpdf
from queens.interfaces.direct_python_interface import DirectPythonInterface
from queens.iterators.sequential_monte_carlo_iterator import SequentialMonteCarloIterator
from queens.main import run_iterator
from queens.models.likelihood_models.gaussian_likelihood import GaussianLikelihood
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.utils.experimental_data_reader import ExperimentalDataReader


def test_smc_generic_temper_multivariate_gaussian(
    tmp_path, _create_experimental_data, _initialize_global_settings
):
    """Test SMC with a multivariate Gaussian and generic tempering."""
    # Parameters
    x1 = NormalDistribution(mean=1.0, covariance=5.0)
    x2 = NormalDistribution(mean=3.0, covariance=5.0)
    x3 = NormalDistribution(mean=-3.0, covariance=5.0)
    x4 = NormalDistribution(mean=1.0, covariance=5.0)
    parameters = Parameters(x1=x1, x2=x2, x3=x3, x4=x4)

    # Setup QUEENS stuff
    experimental_data_reader = ExperimentalDataReader(
        file_name_identifier="*.csv",
        csv_data_base_dir=tmp_path,
        output_label="y_obs",
    )
    mcmc_proposal_distribution = NormalDistribution(
        mean=[0.0, 0.0, 0.0, 0.0],
        covariance=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    interface = DirectPythonInterface(function="patch_for_likelihood", parameters=parameters)
    forward_model = SimulationModel(interface=interface)
    model = GaussianLikelihood(
        noise_type="fixed_variance",
        noise_value=1.0,
        nugget_noise_variance=1e-05,
        experimental_data_reader=experimental_data_reader,
        forward_model=forward_model,
    )
    iterator = SequentialMonteCarloIterator(
        seed=42,
        num_particles=200,
        temper_type="generic",
        plot_trace_every=0,
        num_rejuvenation_steps=20,
        result_description={"write_results": True, "plot_results": False, "cov": True},
        mcmc_proposal_distribution=mcmc_proposal_distribution,
        model=model,
        parameters=parameters,
    )

    # Actual analysis
    run_iterator(iterator)

    # Load results
    result_file = tmp_path / "dummy_experiment_name.pickle"
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # note that the analytical solution can be found in multivariate_gaussian_4D_logpdf
    # we only have a very inaccurate approximation here:
    np.testing.assert_array_almost_equal(
        results['mean'], np.array([[0.884713, 2.903405, -3.112647, 1.56134]]), decimal=5
    )

    np.testing.assert_almost_equal(
        results['var'], np.array([[3.255066, 4.143380, 1.838545, 2.834356]]), decimal=5
    )

    np.testing.assert_almost_equal(
        results['cov'],
        np.array(
            [
                [
                    [3.255066, 1.781563, 0.313565, -0.090972],
                    [1.781563, 4.143380, 0.779616, 1.704881],
                    [0.313565, 0.779616, 1.838545, 0.630236],
                    [-0.090972, 1.704881, 0.630236, 2.834356],
                ]
            ]
        ),
        decimal=5,
    )


def target_density(self, samples):  # pylint: disable=unused-argument
    """TODO_doc."""
    samples = np.atleast_2d(samples)
    log_likelihood = gaussian_4d_logpdf(samples).reshape(-1, 1)

    return log_likelihood


@pytest.fixture(name="_create_experimental_data")
def fixture_create_experimental_data(tmp_path):
    """TODO_doc."""
    # generate 10 samples from the same gaussian
    samples = GAUSSIAN_4D.draw(10)
    pdf = gaussian_4d_logpdf(samples)

    # write the data to a csv file in tmp_path
    data_dict = {'y_obs': pdf}
    experimental_data_path = tmp_path / 'experimental_data.csv'
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)
