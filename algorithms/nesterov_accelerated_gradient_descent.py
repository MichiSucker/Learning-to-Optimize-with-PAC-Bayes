import torch
import torch.nn as nn

from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm


class NesterovAcceleratedGradient(nn.Module):

    def __init__(self, alpha: torch.Tensor):
        super(NesterovAcceleratedGradient, self).__init__()
        self.alpha = nn.Parameter(alpha)

    def forward(self, opt_algo: OptimizationAlgorithm) -> torch.Tensor:
        t_new = 0.5 * (1.0 + torch.sqrt(1.0 + 4 * opt_algo.current_state[0][-1].clone() ** 2))
        y_k = (opt_algo.current_state[2] + ((opt_algo.current_state[0][-1] - 1) / t_new) *
               (opt_algo.current_state[2] - opt_algo.current_state[1]))
        result = y_k - self.alpha * opt_algo.loss_function.compute_gradient(y_k)
        result = torch.maximum(torch.tensor(0.0), result)
        result = torch.minimum(torch.tensor(1.0), result)
        return result

    @staticmethod
    def update_state(opt_algo: OptimizationAlgorithm) -> None:
        opt_algo.current_state[0][-1] = 0.5 * (1.0 + torch.sqrt(1.0 + 4 * opt_algo.current_state[0][-1].clone() ** 2))
        opt_algo.current_state[1] = opt_algo.current_state[2].detach().clone()
        opt_algo.current_state[2] = opt_algo.current_iterate.detach().clone()
