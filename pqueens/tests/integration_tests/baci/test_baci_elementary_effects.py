"""Test suite for integration tests for the Morris-Salib Iterator.

Estimate Elementary Effects for local simulations with BACI using the
INVAAA minimal model.
"""

import json
import os
from pathlib import Path

import pytest

from pqueens import run
from pqueens.utils import injector
from pqueens.utils.run_subprocess import run_subprocess


@pytest.fixture(scope="session")
def output_directory_forward(tmpdir_factory):
    """Create two temporary output directories for test runs with singularity.

        * with singularity (<...>_true)
        * without singularity (<...>_false)

    Args:
        tmpdir_factory: Fixture used to create arbitrary temporary directories

    Returns:
        output_directory_forward (dict): Temporary output directories for simulation without and
        with singularity
    """
    path_singularity_true = tmpdir_factory.mktemp("test_baci_elementary_effects_true")
    path_singularity_false = tmpdir_factory.mktemp("test_baci_elementary_effects_false")

    return {True: path_singularity_true, False: path_singularity_false}


@pytest.fixture()
def experiment_directory(output_directory_forward, singularity_bool):
    """Return experiment directory depending on *singularity_bool*.

    Returns:
        experiment_directory (LocalPath): Experiment directory depending on *singularity_bool*
    """
    return output_directory_forward[singularity_bool]


@pytest.fixture()
def check_experiment_directory(experiment_directory):
    """Check if experiment directory contains subdirectories.

    Raises:
        AssertionError: If experiment directory does not contain subdirectories.
    """
    number_subdirectories = count_subdirectories(experiment_directory)

    assert (
        number_subdirectories != 0
    ), "Empty output directory. Run test_baci_elementary_effects first."


def count_subdirectories(current_directory):
    """Count subdirectories in *current_directory*.

    Returns:
        number_subdirectories (int): Number of subdirectories
    """
    number_subdirectories = 0
    for current_subdirectory in os.listdir(current_directory):
        path_current_subdirectory = os.path.join(current_directory, current_subdirectory)
        if os.path.isdir(path_current_subdirectory):
            number_subdirectories += 1
    return number_subdirectories


def remove_job_output_directory(experiment_directory, jobid):
    """Remove output directory of job #jobid from *experiment_directory*."""
    rm_cmd = "rm -r " + str(experiment_directory) + "/" + str(jobid)
    run_subprocess(rm_cmd)


def test_baci_elementary_effects(
    inputdir,
    third_party_inputs,
    baci_link_paths,
    singularity_bool,
    experiment_directory,
    baci_elementary_effects_check_results,
):
    """Integration test for the Elementary Effects Iterator together with BACI.

    The test runs a local native BACI simulation as well as a local Singularity
    based BACI simulation for elementary effects.

    Args:
        inputdir (str): Path to the JSON input file
        third_party_inputs (str): Path to the BACI input files
        baci_link_paths(str): Path to the links pointing to *baci_release* and *post_drt_monitor*
        singularity_bool (str): String that encodes a boolean that is parsed to the JSON input file
        experiment_directory (LocalPath): Experiment directory depending on *singularity_bool*
        baci_elementary_effects_check_results (function): function to check the results
    """
    template = os.path.join(inputdir, "baci_local_elementary_effects_template.yml")
    input_file = os.path.join(experiment_directory, "elementary_effects_baci_local_invaaa.yml")
    third_party_input_file = os.path.join(third_party_inputs, "baci_input_files", "invaaa_ee.dat")
    experiment_name = "ee_invaaa_local_singularity_" + json.dumps(singularity_bool)

    baci_release, post_drt_monitor, _, _ = baci_link_paths

    dir_dict = {
        'experiment_name': experiment_name,
        'baci_input': third_party_input_file,
        'baci_release': baci_release,
        'post_drt_monitor': post_drt_monitor,
        'singularity_boolean': json.dumps(singularity_bool),
    }

    injector.inject(dir_dict, template, input_file)
    run(Path(input_file), Path(experiment_directory))

    result_file_name = experiment_name + ".pickle"
    result_file = os.path.join(experiment_directory, result_file_name)
    baci_elementary_effects_check_results(result_file)


def test_baci_dask_elementary_effects(
    inputdir,
    third_party_inputs,
    baci_link_paths,
    tmp_path,
    baci_elementary_effects_check_results,
):
    """Integration test for the Elementary Effects Iterator together with BACI.

    The test runs a local native BACI simulation as well as a local Singularity
    based BACI simulation for elementary effects.

    Args:
        inputdir (str): Path to the JSON input file
        third_party_inputs (str): Path to the BACI input files
        baci_link_paths(str): Path to the links pointing to *baci_release* and *post_drt_monitor*
        baci_elementary_effects_check_results (function): function to check the results
    """
    template = os.path.join(inputdir, "baci_dask_local_elementary_effects_template.yml")
    input_file = tmp_path / "elementary_effects_baci_dask_local_invaaa.yml"
    third_party_input_file = os.path.join(third_party_inputs, "baci_input_files", "invaaa_ee.dat")
    experiment_name = "dask_ee_invaaa_local"

    baci_release, post_drt_monitor, _, _ = baci_link_paths

    dir_dict = {
        'experiment_name': experiment_name,
        'baci_input': third_party_input_file,
        'baci_release': baci_release,
        'post_drt_monitor': post_drt_monitor,
    }

    injector.inject(dir_dict, template, input_file)
    run(Path(input_file), tmp_path)

    result_file_name = experiment_name + ".pickle"
    result_file = tmp_path / result_file_name
    baci_elementary_effects_check_results(result_file)
