"""Module to read experimental data."""

import numpy as np

from pqueens.data_processor import from_config_create_data_processor


def get_experimental_data_and_write_to_db(
    config,
    data_processor_name,
    base_dir,
    file_name,
    experiment_name,
    db,
    coordinate_labels,
    time_label,
    output_label,
):
    """Load all experimental data into QUEENS.

    Args:
        config (dict): input json file with problem description
        data_processor_name (str): DataProcessor name
        base_dir (str): Path to base directory containing experimental data
        file_name (str): file name of experimental data
        experiment_name (str): Name of the current experiment in QUEENS
        db (obj): Database object
        coordinate_labels (lst): List of column-wise coordinate labels in csv files
        time_label (str): Name of the time variable in csv file
        output_label (str): Label that marks the output quantity in the csv file

    Returns:
        y_obs_vec (np.array): Column-vector of model outputs which correspond row-wise to
                                observation coordinates
        experimental_coordinates (np.array): Matrix with observation coordinates. One row
                                                corresponds to one coordinate point.
    """
    try:
        # standard initialization for data_processor
        data_processor = from_config_create_data_processor(config, data_processor_name)
    except ValueError:
        # allow short initialization for data_processor
        # only using the 'file_name_identifier'
        short_config = {
            "data_processor": {
                "type": "csv",
                "file_name_identifier": file_name,
                "file_options_dict": {
                    "header_row": 0,
                    "index_column": False,
                    "returned_filter_format": "dict",
                    "filter": {"type": "entire_file"},
                },
            },
        }
        data_processor = from_config_create_data_processor(short_config, "data_processor")

    experimental_data_dict = data_processor.get_data_from_file(base_dir)

    # save experimental data to the database
    # For now we save all data-points to the experimental data slot `1`. This could be
    # extended in the future if we want to read in several different data sources
    db.save(experimental_data_dict, experiment_name, 'experimental_data', '1')

    # arrange the experimental data coordinates
    experimental_coordinates = (
        np.array([experimental_data_dict[coordinate] for coordinate in coordinate_labels]),
    )[0].T

    # get a unique vector of observation times
    if time_label:
        time_vec = np.sort(list(set(experimental_data_dict[time_label])))
    else:
        time_vec = None

    # get the experimental outputs
    y_obs_vec = np.array(experimental_data_dict[output_label]).reshape(
        -1,
    )

    return y_obs_vec, experimental_coordinates, time_vec
