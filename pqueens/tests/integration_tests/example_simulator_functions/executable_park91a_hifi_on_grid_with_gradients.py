#!/usr/bin/env python3
"""High-fidelity Park91a with x3 and x4 as fixed coordinates as executable."""
import sys
from pathlib import Path

import numpy as np
from numpy import genfromtxt


def park91a_hifi_coords(x1, x2, x3, x4):
    r"""High-fidelity Park91a function.

    High-fidelity Park91a function with *x3* and *x4* as fixed coordinates.
    Coordinates are prescribed in the main function of this module.

    Simple four-dimensional benchmark function as proposed in [1] to mimic
    a computer model. For the purpose of multi-fidelity simulation, [3]
    defined a corresponding lower fidelity function, which is implemented
    in *park91a_lofi*.

    The high-fidelity version is defined as:

    :math:`f({\bf x}) =
    \frac{x_1}{2} \left[\sqrt{1+(x_2+x_3^2)\frac{x_4}{x_1^2}}-1 \right]+(x_1+3x_4)\exp[1-\sin(x_3)]`

    Args:
        x1 (float): Input parameter 1 [0,1)
        x2 (float): Input parameter 2 [0,1)
        x3 (float): Input parameter 3 [0,1)
        x4 (float): Input parameter 4 [0,1)

    Returns:
        float: Value of function at parameters

    References:
        [1] Park, J.-S.(1991). Tuning complex computer codes to data and optimal
            designs, Ph.D. Thesis

        [2] Cox, D. D., Park, J.-S., & Singer, C. E. (2001). A statistical method
            for tuning a computer code to a database. Computational Statistics &
            Data Analysis, 37(1), 77?92. http://doi.org/10.1016/S0167-9473(00)00057-8

        [3] Xiong, S., Qian, P., & Wu, C. (2013). Sequential design and analysis of
            high-accuracy and low-accuracy computer codes. Technometrics.
            http://doi.org/10.1080/00401706.2012.723572
    """
    # catch values outside of definition
    if x1 <= 0:
        x1 = 0.01
    elif x1 >= 1:
        x1 = 0.99

    if x2 <= 0:
        x2 = 0.01
    elif x2 >= 1:
        x2 = 0.99

    if x3 <= 0:
        x3 = 0.01
    elif x3 >= 1:
        x3 = 0.99

    if x4 <= 0:
        x4 = 0.01
    elif x4 >= 1:
        x4 = 0.99

    term1a = x1 / 2
    term1b = np.sqrt(1 + (x2 + x3**2) * x4 / (x1**2)) - 1
    term1 = term1a * term1b

    term2a = x1 + 3 * x4
    term2b = np.exp(1 + np.sin(x3))
    term2 = term2a * term2b

    y = term1 + term2

    # ----
    term1a = x1 / 2
    d_term1a_dx1 = 1 / 2
    term1b = np.sqrt(1 + (x2 + x3**2) * x4 / (x1**2)) - 1
    d_term1b_dx1 = (
        1
        / (2 * np.sqrt(1 + (x2 + x3**2) * x4 / (x1**2)))
        * (-2 * (x2 + x3**2) * x4 * x1 ** (-3))
    )
    term1 = term1a * term1b
    d_term1_dx1 = d_term1a_dx1 * term1b + term1a * d_term1b_dx1

    term2a = x1 + 3 * x4
    d_term2a_dx1 = 1
    term2b = np.exp(1 + np.sin(x3))
    d_term2b_dx1 = 0
    term2 = term2a * term2b
    d_term2_dx1 = d_term2a_dx1 * term2b + term2a * d_term2b_dx1

    dy_dx1 = d_term1_dx1 + d_term2_dx1

    # ----
    term1a = x1 / 2
    d_term1a_dx2 = 0
    term1b = np.sqrt(1 + (x2 + x3**2) * x4 / (x1**2)) - 1
    d_term1b_dx2 = 1 / (2 * np.sqrt(1 + (x2 + x3**2) * x4 / (x1**2))) * x4 / (x1**2)
    term1 = term1a * term1b
    d_term1_dx2 = d_term1a_dx2 * term1b + term1a * d_term1b_dx2

    term2a = x1 + 3 * x4
    term2b = np.exp(1 + np.sin(x3))
    term2 = term2a * term2b
    d_term2_dx2 = 0

    dy_dx2 = d_term1_dx2 + d_term2_dx2

    return y, (dy_dx1, dy_dx2)


def main(run_type, params):
    """Interface to Park91a test function.

    Args:
        run_type (str): Run type for the main function:

                        - 's': standard run without gradients
                        - 'p': provided gradient run
                        - 'a': adjoint run that solves the adjoint equation
        params (dict): Dictionary with parameters

    Returns:
        function_output (np.array): Value of function input parameters
        evaluated_gradient_expression (np.array): Value of gradient input parameters
                                                  Note: This can be different gradients
                                                  depending on the run_type!
    """
    # use x3 and x4 as coordinates and create coordinate grid
    xx3 = np.linspace(0, 1, 4)
    xx4 = np.linspace(0, 1, 4)
    x3_vec, x4_vec = np.meshgrid(xx3, xx4)
    x3_vec = x3_vec.flatten()
    x4_vec = x4_vec.flatten()

    run_type_dict = {'s': run_type_standard, 'p': run_type_provided_gradient, 'a': run_type_adjoint}
    run_function = run_type_dict.get(run_type)
    if run_function is None:
        raise ValueError(f'Invalid run_type, run_type must be in {run_type_dict.keys}!')

    function_output, evaluated_gradient_expression = run_function(x3_vec, x4_vec, params)

    return function_output, evaluated_gradient_expression


def run_type_standard(x3_vec, x4_vec, params):
    """Run standard function without gradients.

    Args:
        x3_vec (np.array): Vector of x3 values from grid points
        x4_vec (np.array): Vector of x4 values from grid points
        params (dict): Dictionary with input parameters

    Returns:
        y_vec (np.array): Vector of function values at grid points
        y_grad (np.array): Empty dummy vector of gradient values at grid points
    """
    y_vec = []
    y_grad = []
    for x3, x4 in zip(x3_vec, x4_vec):
        y_vec.append(park91a_hifi_coords(params['x1'], params['x2'], x3, x4)[0])
    y_vec = np.array(y_vec)
    y_grad = np.array(y_grad)
    return y_vec, y_grad


def run_type_provided_gradient(x3_vec, x4_vec, params):
    """Run with provided gradients.

    Args:
        x3_vec (np.array): Vector of x3 values from grid points
        x4_vec (np.array): Vector of x4 values from grid points
        params (dict): Dictionary with input parameters

    Returns:
        y_vec (np.array): Vector of function values at grid points
        y_grad (np.array): Vector of gradient values at grid points
    """
    y_vec = []
    y_grad = []
    for x3, x4 in zip(x3_vec, x4_vec):
        y_vec.append(park91a_hifi_coords(params['x1'], params['x2'], x3, x4)[0])
        y_grad.append(park91a_hifi_coords(params['x1'], params['x2'], x3, x4)[1][:])
    y_vec = np.array(y_vec)
    y_grad = np.array(y_grad).T
    return y_vec, y_grad


def run_type_adjoint(x3_vec, x4_vec, params):
    """Run that only solves the adjoint problem.

    Args:
        x3_vec (np.array): Vector of x3 values from grid points
        x4_vec (np.array): Vector of x4 values from grid points
        params (dict): Dictionary with input parameters

    Returns:
        y_vec (np.array): Empty dummy vector of function values at grid points
        do_dx (np.array): Vector of gradient values of the objective function at grid points
    """
    y_vec = []
    y_grad = []
    for x3, x4 in zip(x3_vec, x4_vec):
        y_grad.append(park91a_hifi_coords(params['x1'], params['x2'], x3, x4)[1][:])

    # we define g(y,x1,x2) = y - term1 - term2 = 0
    # as y is explicit in g, dg_dy = 1:
    # --> 1 * lambda = -do_dy, with o being the objective function, here log-likelihood
    # and y the value the output of this function, do_dy is here read-in from a csv file
    # hence: lambda = -do_dy, which we will load in the next lines:
    adjoint_base_path = Path(input_file_path).parent  # not parent but -
    adjoint_path = adjoint_base_path / "grad_objective.csv"
    do_dy = genfromtxt(adjoint_path, delimiter=',')
    lambda_var = np.negative(np.atleast_2d(np.array(do_dy)))

    # now we need to implement g_x, the jacobian of the residuum function w.r.t. the input
    # afterwards we can calculate the final gradient do_dx, the gradient of the objective fun
    # w.r.t. to the model input x; for this simple analytical example g_x is simply the
    # negative gradient/jacobian of the model (for PDEs)
    dg_dx = -np.array(y_grad)

    # now we can finalize the adjoint:
    do_dx = np.array(np.dot(lambda_var, dg_dx))
    y_vec = np.array(y_vec)

    return y_vec, do_dx


def write_results(output, output_path):
    """Write solution to csv files."""
    y_vec, y_grad = output
    output_file = output_path.parent / (output_path.stem + "_output.csv")
    gradient_file = output_path.parent / (output_path.stem + "_gradient.csv")
    if y_vec.shape[0] != 0:
        np.savetxt(output_file, y_vec, delimiter=",")
    if y_grad.shape[0] != 0:
        np.savetxt(gradient_file, np.squeeze(y_grad), delimiter=",")


def read_input_file(input_file_path):
    """Read-in input from csv file."""
    inputs = genfromtxt(input_file_path, delimiter=r',|\s+')
    return inputs


if __name__ == "__main__":
    run_type = sys.argv[1]
    input_file_path = sys.argv[2]
    output_path = Path(sys.argv[3])
    params = read_input_file(input_file_path)
    output = main(run_type, {"x1": params[0], "x2": params[1]})
    write_results(output, output_path)
