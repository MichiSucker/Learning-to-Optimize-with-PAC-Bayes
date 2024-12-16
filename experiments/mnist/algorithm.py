import torch
import torch.nn as nn

from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm


class MnistOptimizer(nn.Module):

    def __init__(self, dim: int):
        super(MnistOptimizer, self).__init__()

        self.extrapolation = nn.Parameter(0.001 * torch.ones(dim))
        self.gradient = nn.Parameter(0.001 * torch.ones(dim))
        self.extrapolation_2 = nn.Parameter(0.001 * torch.ones(dim))
        self.gradient_2 = nn.Parameter(0.001 * torch.ones(dim))

        size = 5
        self.update_layer = nn.Sequential(
            nn.Conv2d(7, 3 * size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(3 * size, 4 * size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(4 * size, 3 * size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(3 * size, 1, kernel_size=1, bias=False),
        )

        h_size = 5
        self.coefficients = nn.Sequential(
            nn.Linear(4, 3 * h_size, bias=False),
            nn.ReLU(),
            nn.Linear(3 * h_size, 4 * h_size, bias=False),
            nn.ReLU(),
            nn.Linear(4 * h_size, 3 * h_size, bias=False),
            nn.ReLU(),
            nn.Linear(3 * h_size, 7, bias=False),
        )

        # For stability
        self.eps = torch.tensor(1e-16).float()

    def forward(self, opt_algo: OptimizationAlgorithm) -> torch.Tensor:

        loss = opt_algo.loss_function(opt_algo.current_state[1].clone()).reshape((1,))
        old_loss = opt_algo.loss_function(opt_algo.current_state[1].clone()).reshape((1,))
        factor = torch.exp(-(loss/1e-2)**4)

        # Normalize gradient
        grad = opt_algo.loss_function.compute_gradient(opt_algo.current_state[1])
        grad_norm = torch.linalg.norm(grad).reshape((1,))
        if grad_norm > self.eps:
            grad = grad / grad_norm

        # Compute momentum
        diff = opt_algo.current_state[1] - opt_algo.current_state[0]
        diff_norm = torch.linalg.norm(diff).reshape((1,))
        if diff_norm > self.eps:
            diff = diff / diff_norm

        weighting = self.coefficients(
            torch.concat(
                (torch.log(1 + grad_norm.reshape((1,))),
                 torch.log(1 + diff_norm.reshape((1,))),
                 torch.log(1 + loss),
                 torch.log(1 + old_loss)
                 ))
        )

        direction = self.update_layer(torch.concat((
            (1. - factor) * weighting[0] * self.gradient * grad.reshape((1, 1, 1, -1)),
            (1. - factor) * weighting[1] * self.extrapolation * diff.reshape((1, 1, 1, -1)),
            weighting[2] * grad.reshape((1, 1, 1, -1)),
            weighting[3] * diff.reshape((1, 1, 1, -1)),
            weighting[4] * (grad * diff).reshape((1, 1, 1, -1)),
            factor * weighting[5] * self.gradient_2 * grad.reshape((1, 1, 1, -1)),
            factor * weighting[6] * self.extrapolation_2 * diff.reshape((1, 1, 1, -1)),
        ), dim=1)).flatten()

        return opt_algo.current_state[-1] + direction

    @staticmethod
    def update_state(opt_algo: OptimizationAlgorithm) -> None:
        opt_algo.current_state[0] = opt_algo.current_state[1].detach().clone()
        opt_algo.current_state[1] = opt_algo.current_iterate.detach().clone()

