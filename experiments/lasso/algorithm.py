import torch
import torch.nn as nn
from algorithms.fista import soft_thresholding
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm


class SparsityNet(nn.Module):

    def __init__(self, dim: int):
        super(SparsityNet, self).__init__()

        self.dim = dim

        in_size = 4
        hidden_size = 64
        out_size = 1
        self.update_layer = nn.Sequential(
            nn.Conv2d(in_size, hidden_size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_size, out_size, kernel_size=1, bias=False),
        )

        in_size = 3
        hidden_size = 64
        out_size = 1
        self.coefficients = nn.Sequential(
            nn.Linear(in_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size, bias=False),
        )

        in_size = 3
        hidden_size = 64
        out_size = 1
        self.sparsity_layer = nn.Sequential(
            nn.Conv2d(in_size, hidden_size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_size, out_size, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.tau = nn.Parameter(torch.tensor(0.001))

        # For stability
        self.eps = torch.tensor(1e-10).float()

    def forward(self, opt_algo: OptimizationAlgorithm):

        # Normalize gradient
        gradient = opt_algo.loss_function.compute_gradient(opt_algo.current_state[1])
        gradient_norm = torch.linalg.norm(gradient).reshape((1,))
        if gradient_norm > self.eps:
            gradient = gradient / gradient_norm

        # Compute subgradient of L_1-norm
        subgradient = torch.ones(opt_algo.current_state[1].shape)
        subgradient[opt_algo.current_state[1] < 0] = -1
        subgradient[opt_algo.current_state[1] == 0] = 0
        subgradient_norm = torch.linalg.norm(subgradient)
        if subgradient_norm > self.eps:
            subgradient = subgradient / subgradient_norm

        # Compute and normalize momentum term
        diff = opt_algo.current_state[1] - opt_algo.current_state[0]
        diff_norm = torch.linalg.norm(diff).reshape((1,))
        if diff_norm > self.eps:
            diff = diff / diff_norm

        step_size = self.coefficients(
            torch.concat(
                (torch.log(1 + gradient_norm.reshape((1,))),
                 torch.log(1 + subgradient_norm.reshape((1,))),
                 torch.log(1 + diff_norm.reshape((1,)))))
        )

        direction = self.update_layer(torch.concat((
            gradient.reshape((1, 1, 1, -1)),
            diff.reshape((1, 1, 1, -1)),
            (gradient * diff).reshape((1, 1, 1, -1)),
            subgradient.reshape((1, 1, 1, -1)),
        ), dim=1))

        # Apply soft-thresholding
        update = opt_algo.current_state[-1] + step_size * direction.flatten()

        norm = torch.linalg.norm(update)
        sparsity = self.sparsity_layer(
            torch.concat((
                opt_algo.current_state[-1].reshape((1, 1, 1, -1)),
                update.reshape((1, 1, 1, -1)),
                subgradient.reshape((1, 1, 1, -1)),
            ), dim=1)).flatten()
        update = soft_thresholding(sparsity * update, tau=self.tau)
        return update * norm / torch.linalg.norm(update)

    @staticmethod
    def update_state(opt_algo):
        opt_algo.current_state[0] = opt_algo.current_state[1].detach().clone()
        opt_algo.current_state[1] = opt_algo.current_iterate.detach().clone()
