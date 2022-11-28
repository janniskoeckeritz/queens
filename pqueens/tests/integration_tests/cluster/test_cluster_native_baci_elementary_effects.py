"""Test suite for native cluster tests.

Elementary Effects simulations with BACI using the INVAAA minimal model.
"""
import logging
import pathlib

import pytest

from pqueens import run
from pqueens.utils import injector
from pqueens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "cluster",
    [
        pytest.param("deep", marks=pytest.mark.lnm_cluster_deep_native),
        pytest.param("bruteforce", marks=pytest.mark.lnm_cluster_bruteforce_native),
    ],
    indirect=True,
)
def test_cluster_native_baci_elementary_effects(
    inputdir,
    tmpdir,
    third_party_inputs,
    cluster_queens_testing_folder,
    baci_cluster_paths_native,
    cluster,
    baci_elementary_effects_check_results,
):
    """Test for the Elementary Effects Iterator on the clusters with BACI.

    Args:
        inputdir (str): Path to the JSON input file
        tmpdir (str): Temporary directory in which the pytests are run
        third_party_inputs (str): Path to the BACI input files
        cluster_queens_testing_folder (PosixPath): Path to testing folder on deep
        baci_cluster_paths_native (dict): Paths to baci native on cluster
        baci_elementary_effects_check_results (function): function to check the results

    Returns:
        None
    """
    path_to_executable = baci_cluster_paths_native["path_to_executable"]
    path_to_drt_monitor = baci_cluster_paths_native["path_to_drt_monitor"]

    experiment_name = cluster + "_native_elementary_effects"

    template = pathlib.Path(inputdir, "baci_cluster_native_elementary_effects_template.json")
    input_file = pathlib.Path(tmpdir, f"elementary_effects_{cluster}_invaaa.json")
    cluster_experiment_dir = cluster_queens_testing_folder.joinpath(experiment_name)

    baci_input_filename = "invaaa_ee.dat"
    third_party_input_file_local = pathlib.Path(
        third_party_inputs, "baci_input_files", baci_input_filename
    )

    experiment_dir = cluster_experiment_dir.joinpath("output")

    command_string = f'mkdir -v -p {experiment_dir}'
    returncode, _, stdout, stderr = run_subprocess(
        command_string=command_string,
        subprocess_type='simple',
    )
    _logger.info(stdout)
    if returncode:
        raise Exception(stderr)

    dir_dict = {
        'experiment_name': str(experiment_name),
        'input_template': str(third_party_input_file_local),
        'path_to_executable': str(path_to_executable),
        'path_to_drt_monitor': str(path_to_drt_monitor),
        'experiment_dir': str(experiment_dir),
        'cluster': cluster,
        'scheduler_type': cluster,
    }

    injector.inject(dir_dict, template, input_file)
    run(pathlib.Path(input_file), pathlib.Path(tmpdir))

    result_file = pathlib.Path(tmpdir, experiment_name + '.pickle')
    baci_elementary_effects_check_results(result_file)
