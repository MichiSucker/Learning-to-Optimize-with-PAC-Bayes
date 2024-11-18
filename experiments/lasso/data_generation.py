import torch
from typing import Tuple, Callable


def get_dimensions() -> Tuple[int, int]:
    dimension_right_hand_side = 70
    dimension_optimization_variable = 350
    return dimension_right_hand_side, dimension_optimization_variable


def get_distribution_of_right_hand_side() -> torch.distributions.Distribution:
    dimension_right_hand_side, _ = get_dimensions()
    mean = torch.distributions.uniform.Uniform(-5, 5).sample((dimension_right_hand_side,))
    cov = torch.distributions.uniform.Uniform(-5, 5).sample((dimension_right_hand_side, dimension_right_hand_side))
    cov = torch.transpose(cov, 0, 1) @ cov
    return torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)


def get_distribution_of_regularization_parameter() -> torch.distributions.Distribution:
    return torch.distributions.uniform.Uniform(low=5, high=10)


def get_matrix_for_smooth_part() -> torch.Tensor:
    dimension_right_hand_side, dimension_optimization_variable = get_dimensions()
    return torch.distributions.uniform.Uniform(-0.5, 0.5).sample((
        dimension_right_hand_side, dimension_optimization_variable))


def calculate_smoothness_parameter(matrix: torch.Tensor) -> torch.Tensor:
    eigenvalues = torch.linalg.eigvalsh(matrix.T @ matrix)
    return eigenvalues[-1]


def get_loss_function_of_algorithm() -> Tuple[Callable, Callable, Callable]:

    def smooth_part(x, parameter):
        return 0.5 * torch.linalg.norm(torch.matmul(parameter['A'], x) - parameter['b']) ** 2

    def nonsmooth_part(x, parameter):
        return parameter['mu'] * torch.linalg.norm(x, ord=1)

    def loss_function(x, parameter):
        return smooth_part(x, parameter) + nonsmooth_part(x, parameter)

    return loss_function, smooth_part, nonsmooth_part


def check_and_extract_number_of_datapoints(number_of_datapoints_per_dataset: dict) -> Tuple[int, int, int, int]:
    if (('prior' not in number_of_datapoints_per_dataset)
            or ('train' not in number_of_datapoints_per_dataset)
            or ('test' not in number_of_datapoints_per_dataset)
            or ('validation' not in number_of_datapoints_per_dataset)):
        raise ValueError("Missing number of datapoints.")
    else:
        return (number_of_datapoints_per_dataset['prior'],
                number_of_datapoints_per_dataset['train'],
                number_of_datapoints_per_dataset['test'],
                number_of_datapoints_per_dataset['validation'])


def create_parameter(matrix: torch.Tensor,
                     right_hand_side: torch.Tensor,
                     regularization_parameter: torch.Tensor) -> dict:
    return {'A': matrix, 'b': right_hand_side, 'mu': regularization_parameter}


def get_parameters(matrix: torch.Tensor, number_of_datapoints_per_dataset: dict) -> dict:

    n_prior, n_train, n_test, n_validation = check_and_extract_number_of_datapoints(number_of_datapoints_per_dataset)
    distribution_right_hand_side = get_distribution_of_right_hand_side()
    distribution_regularization_parameters = get_distribution_of_regularization_parameter()

    parameters = {}
    for name, number_of_datapoints in [('prior', n_prior), ('train', n_train), ('test', n_test),
                                       ('validation', n_validation)]:
        parameters[name] = [create_parameter(
            matrix=matrix,
            right_hand_side=distribution_right_hand_side.sample((1,)),
            regularization_parameter=distribution_regularization_parameters.sample((1,)))
                            for _ in range(number_of_datapoints)]
    return parameters


def get_data(number_of_datapoints_per_dataset: dict) -> Tuple[dict, Callable, Callable, Callable, torch.Tensor]:

    A = get_matrix_for_smooth_part()
    smoothness_parameter = calculate_smoothness_parameter(matrix=A)
    loss_function_of_algorithm, smooth_part, nonsmooth_part = get_loss_function_of_algorithm()
    parameters = get_parameters(matrix=A, number_of_datapoints_per_dataset=number_of_datapoints_per_dataset)

    return parameters, loss_function_of_algorithm, smooth_part, nonsmooth_part, smoothness_parameter
