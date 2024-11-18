import torch
import torch.nn as nn
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm


class Dummy(nn.Module):

    def __init__(self):
        super(Dummy, self).__init__()
        self.scale = nn.Parameter(torch.tensor(1e-3))

    def forward(self, optimization_algorithm: OptimizationAlgorithm) -> None:
        gradient = optimization_algorithm.loss_function.compute_gradient(optimization_algorithm.current_iterate)
        return optimization_algorithm.current_iterate + self.scale * gradient

    @staticmethod
    def update_state(optimization_algorithm: OptimizationAlgorithm) -> None:
        optimization_algorithm.current_state = optimization_algorithm.current_iterate.detach().clone().reshape((1, -1))


class DummyWithMoreTrainableParameters(nn.Module):

    def __init__(self):
        super(DummyWithMoreTrainableParameters, self).__init__()
        self.scale = nn.Parameter(torch.tensor(1e-3))
        self.matrix = nn.Parameter(torch.tensor([[1e-3, 1.], [2e-4, 5.]]))
        self.fixed_scale = torch.tensor(1e-3)

    def forward(self, optimization_algorithm: OptimizationAlgorithm) -> torch.Tensor:
        gradient = optimization_algorithm.loss_function.compute_gradient(optimization_algorithm.current_iterate)
        return optimization_algorithm.current_iterate + self.scale * gradient

    @staticmethod
    def update_state(optimization_algorithm: OptimizationAlgorithm) -> None:
        optimization_algorithm.current_state = optimization_algorithm.current_iterate.detach().clone().reshape((1, -1))


class NonTrainableDummy(nn.Module):

    def __init__(self):
        super(NonTrainableDummy, self).__init__()
        self.scale = torch.tensor(1e-4)

    def forward(self, optimization_algorithm: OptimizationAlgorithm) -> torch.Tensor:
        gradient = optimization_algorithm.loss_function.compute_gradient(optimization_algorithm.current_iterate)
        return optimization_algorithm.current_iterate + self.scale * gradient

    @staticmethod
    def update_state(optimization_algorithm: OptimizationAlgorithm) -> None:
        optimization_algorithm.current_state = optimization_algorithm.current_iterate.detach().clone().reshape((1, -1))
