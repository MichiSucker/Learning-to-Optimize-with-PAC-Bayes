import torch
import torch.nn as nn
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm


class GradientDescent(nn.Module):

    def __init__(self, alpha: torch.Tensor):
        super(GradientDescent, self).__init__()
        self.alpha = nn.Parameter(alpha)

    def forward(self, algorithm: OptimizationAlgorithm) -> torch.Tensor:
        return (algorithm.current_state[0]
                - self.alpha * algorithm.loss_function.compute_gradient(algorithm.current_iterate))

    @staticmethod
    def update_state(opt_algo: OptimizationAlgorithm) -> None:
        opt_algo.current_state = opt_algo.current_iterate.detach().clone().reshape((1, -1))
