import torch
import torch.nn as nn
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm


def soft_thresholding(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.zeros_like(x), torch.abs(x) - tau) * torch.sign(x)


class FISTA(nn.Module):

    def __init__(self, alpha: torch.Tensor):
        super(FISTA, self).__init__()
        self.alpha: torch.Tensor = nn.Parameter(alpha)

    def forward(self, opt_algo: OptimizationAlgorithm) -> torch.Tensor:

        mu = opt_algo.loss_function.get_parameter()['mu']
        t_new = 0.5 * (1.0 + torch.sqrt(1.0 + 4 * opt_algo.current_state[0][-1].clone() ** 2))
        y_k = (opt_algo.current_state[2] + ((opt_algo.current_state[0][-1] - 1) / t_new) *
               (opt_algo.current_state[2] - opt_algo.current_state[1]))
        result = soft_thresholding(
            x=y_k - self.alpha * opt_algo.loss_function.compute_gradient_of_smooth_part(y_k),
            tau=self.alpha * mu)
        return result

    @staticmethod
    def update_state(opt_algo: OptimizationAlgorithm) -> None:
        opt_algo.current_state[0][-1] = 0.5 * (1.0 + torch.sqrt(1.0 + 4 * opt_algo.current_state[0][-1].clone() ** 2))
        opt_algo.current_state[1] = opt_algo.current_state[2].detach().clone()
        opt_algo.current_state[2] = opt_algo.current_iterate.detach().clone()
