import torch
import torch.nn as nn

from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm


class HeavyBallWithFriction(nn.Module):

    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor):
        super(HeavyBallWithFriction, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, opt_algo: OptimizationAlgorithm) -> torch.Tensor:
        return (opt_algo.current_state[1]
                - self.alpha * opt_algo.loss_function.compute_gradient(opt_algo.current_state[1])
                + self.beta * (opt_algo.current_state[1] - opt_algo.current_state[0]))

    @staticmethod
    def update_state(opt_algo: OptimizationAlgorithm) -> None:
        opt_algo.current_state[0] = opt_algo.current_state[1].detach().clone()
        opt_algo.current_state[1] = opt_algo.current_iterate.detach().clone()
