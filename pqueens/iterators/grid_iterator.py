"""Grid Iterator."""
import numpy as np

import pqueens.visualization.grid_iterator_visualization as qvis
from pqueens.models import from_config_create_model
from pqueens.utils.process_outputs import process_ouputs, write_results

from .iterator import Iterator


class GridIterator(Iterator):
    """Grid Iterator to enable meshgrid evaluations.

    Different axis scaling possiblem: as linear, log10 or ln.

    Attributes:
        model (model): Model to be evaluated by iterator
        grid_dict (dict): Dictionary containing grid information
        result_description (dict):  Description of desired results
        num_parameters (int)          :   number of parameters to be varied
        samples (np.array):   Array with all samples
        output (np.array):   Array with all model outputs
        num_grid_points_per_axis (list):  list with number of grid points for each grid axis
        scale_type (list): list with string entries denoting scaling type for each grid axis
    """

    def __init__(
        self,
        model,
        result_description,
        global_settings,
        grid_dict,
        num_parameters,
    ):
        """Initialize grid iterator.

        Args:
            model (model): Model to be evaluated by iterator
            result_description (dict):  Description of desired results
            global_settings (dict, optional): Settings for the QUEENS run.
            grid_dict (dict): Dictionary containing grid information
            num_parameters (int):   number of parameters to be varied
        """
        super().__init__(model, global_settings)
        self.grid_dict = grid_dict
        self.result_description = result_description
        self.samples = None
        self.output = None
        self.num_grid_points_per_axis = []
        self.num_parameters = num_parameters
        self.scale_type = []

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name, model=None):
        """Create grid iterator from problem description.

        Args:
            config (dict):       Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section
                                 in options dict (optional)
            model (model):       Model to use (optional)


        Returns:
            iterator (obj): GridIterator object
        """
        method_options = config[iterator_name]
        if model is None:
            model_name = method_options["model_name"]
            model = from_config_create_model(model_name, config)

        result_description = method_options.get("result_description", None)
        global_settings = config.get("global_settings", None)
        grid_dict = method_options.get("grid_design", None)
        num_parameters = len(grid_dict)

        # take care of wrong user input
        if num_parameters is None:
            raise RuntimeError("Number of parameters (num_parameters) not given by user!")

        # ---------------------- CREATE VISUALIZATION BORG ----------------------------
        qvis.from_config_create(config, iterator_name=iterator_name)

        return cls(model, result_description, global_settings, grid_dict, num_parameters)

    def pre_run(self):
        """Generate samples based on description in grid_dict."""
        # Sanity check for random fields
        if self.parameters.random_field_flag:
            raise RuntimeError(
                "The grid iterator is currently not implemented in conjunction with random fields."
            )

        # pre-allocate empty list for filling up with vectors of grid points as elements
        grid_point_list = []

        #  set up 1D arrays for each parameter (needs bounds and type of axis)
        for index, (parameter_name, parameter) in enumerate(self.parameters.dict.items()):
            start_value = parameter.lower_bound
            stop_value = parameter.upper_bound
            data_type = self.grid_dict[parameter_name].get("data_type", None)
            axis_type = self.grid_dict[parameter_name].get("axis_type", None)
            num_grid_points = self.grid_dict[parameter_name].get("num_grid_points", None)
            self.num_grid_points_per_axis.append(num_grid_points)
            self.scale_type.append(axis_type)

            # check user input
            if axis_type is None:
                raise RuntimeError(
                    "Scaling of axis not given properly by user (possible: 'lin', "
                    "'log10' and 'ln')"
                )

            if num_grid_points is None:
                raise RuntimeError(
                    " Number of grid points ('num_grid_points') not given properly by user "
                )

            if axis_type == 'lin':
                grid_point_list.append(
                    np.linspace(
                        start_value,
                        stop_value,
                        num=num_grid_points,
                        endpoint=True,
                        retstep=False,
                    )
                )
            elif axis_type == 'log10':
                grid_point_list.append(
                    np.logspace(
                        np.log10(start_value),
                        np.log10(stop_value),
                        num=num_grid_points,
                        endpoint=True,
                        base=10,
                    )
                )
            elif axis_type == "ln":
                grid_point_list.append(
                    np.logspace(
                        np.log(start_value),
                        np.log(stop_value),
                        num=num_grid_points,
                        endpoint=True,
                        base=np.e,
                    )
                )
            else:
                raise NotImplementedError(
                    "Invalid option for \'axis_type\'. Valid options are: "
                    f"\'lin\', \'log10\', \'ln\'. You chose {axis_type}."
                )

            # handle data types different from float (default)
            if data_type == 'INT':
                grid_point_list[index] = grid_point_list[index].astype(int)

            elif data_type == 'FLOAT':
                pass

            else:
                raise RuntimeError(
                    " Datatype of parameter / random variable given by user not supported by "
                    " grid iterator (possible: 'FLOAT' or 'INT') "
                )

        grid_coords = np.meshgrid(*grid_point_list)
        self.samples = np.empty([np.prod(self.num_grid_points_per_axis), self.num_parameters])
        for i in range(self.num_parameters):
            self.samples[:, i] = grid_coords[i].flatten()

    def core_run(self):
        """Evaluate the meshgrid on model."""
        self.output = self.model.evaluate(self.samples)

    def post_run(self):
        """Analyze the results."""
        if self.result_description is not None:
            results = process_ouputs(self.output, self.result_description, self.samples)
            if self.result_description["write_results"] is True:
                write_results(
                    results,
                    self.global_settings["output_dir"],
                    self.global_settings["experiment_name"],
                )

        # plot QoI over grid
        qvis.grid_iterator_visualization_instance.plot_QoI_grid(
            self.output,
            self.samples,
            self.num_parameters,
            self.num_grid_points_per_axis,
        )
