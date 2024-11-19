import torch


def evaluate_natural_parameters_at(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError('Input to natural parameters has to be a torch.Tensor.')
    return torch.tensor([x, -0.5 * x ** 2])
