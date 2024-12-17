from typing import Tuple, Callable
import torch


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


def get_distribution_of_strong_convexity_parameter() -> Tuple[torch.distributions.Distribution, torch.Tensor]:
    mu_min, mu_max = torch.tensor(1e-3), torch.tensor(5e-3)
    strong_convexity_distribution = torch.distributions.uniform.Uniform(mu_min, mu_max)
    return strong_convexity_distribution, mu_min


def get_distribution_of_smoothness_parameter() -> Tuple[torch.distributions.Distribution, torch.Tensor]:
    L_min, L_max = torch.tensor(1e2), torch.tensor(5e2)
    smoothness_distribution = torch.distributions.uniform.Uniform(L_min, L_max)
    return smoothness_distribution, L_max


def get_distribution_of_right_hand_side() -> Tuple[torch.distributions.Distribution, int]:
    dim = 200
    mean = torch.distributions.uniform.Uniform(-5, 5).sample((dim, ))
    cov = torch.distributions.uniform.Uniform(-5, 5).sample((dim, dim))
    cov = torch.transpose(cov, 0, 1) @ cov
    return torch.distributions.multivariate_normal.MultivariateNormal(mean, cov), dim


def create_parameter(diagonal: torch.Tensor, right_hand_side: torch.Tensor) -> dict:
    return {'A': torch.diag(diagonal), 'b': right_hand_side, 'optimal_loss': torch.tensor(0.0)}


def get_loss_function_of_algorithm() -> Callable:

    def loss_function(x, parameter):
        return 0.5 * torch.linalg.norm(torch.matmul(parameter['A'], x) - parameter['b']) ** 2

    return loss_function


def get_values_of_diagonal(min_value: float, max_value: float, number_of_values: int) -> torch.Tensor:
    return torch.linspace(min_value, max_value, number_of_values)


def get_parameters(number_of_datapoints_per_dataset: dict) -> Tuple[dict, torch.Tensor, torch.Tensor, int]:

    n_prior, n_train, n_test, n_validation = check_and_extract_number_of_datapoints(number_of_datapoints_per_dataset)
    distribution_of_strong_convexity_parameter, mu_min = get_distribution_of_strong_convexity_parameter()
    distribution_of_smoothness_parameter, L_max = get_distribution_of_smoothness_parameter()
    distribution_of_right_hand_side, dim = get_distribution_of_right_hand_side()

    parameters = {}
    for number_of_functions, name in [(n_prior, 'prior'), (n_train, 'train'),
                                      (n_test, 'test'), (n_validation, 'validation')]:
        samples_strong_convexity = distribution_of_strong_convexity_parameter.sample((number_of_functions,))
        samples_smoothness = distribution_of_smoothness_parameter.sample((number_of_functions,))
        samples_right_hand_side = distribution_of_right_hand_side.sample((number_of_functions, ))
        diagonals = [get_values_of_diagonal(
            min_value=torch.sqrt(strong_convexity).item(), max_value=torch.sqrt(smoothness).item(),
            number_of_values=dim)
                     for strong_convexity, smoothness in zip(samples_strong_convexity, samples_smoothness)]
        parameters[name] = [create_parameter(diagonals[i], samples_right_hand_side[i, :])
                            for i in range(number_of_functions)]

    return parameters, mu_min, L_max, dim


def get_data(number_of_datapoints_per_dataset: dict) -> Tuple[dict, Callable, torch.Tensor, torch.Tensor, int]:

    parameters, mu_min, L_max, dim = get_parameters(number_of_datapoints_per_dataset)
    loss_function = get_loss_function_of_algorithm()

    return parameters, loss_function, mu_min, L_max, dim
