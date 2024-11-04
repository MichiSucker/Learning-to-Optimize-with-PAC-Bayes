import torch
import torch.nn as nn


class Quadratics(nn.Module):

    def __init__(self, dim: int):
        super(Quadratics, self).__init__()

        size = 16
        self.update_layer = nn.Sequential(
            nn.Conv2d(3, size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(size, size, kernel_size=1, bias=False),
            nn.Conv2d(size, size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(size, size, kernel_size=1, bias=False),
            nn.Conv2d(size, size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(size, 1, kernel_size=1, bias=False),
        )

        h_size = 8
        self.coefficients = nn.Sequential(
            nn.Linear(4, h_size, bias=False),
            nn.ReLU(),
            nn.Linear(h_size, h_size, bias=False),
            nn.Linear(h_size, h_size, bias=False),
            nn.ReLU(),
            nn.Linear(h_size, h_size, bias=False),
            nn.Linear(h_size, h_size, bias=False),
            nn.ReLU(),
            nn.Linear(h_size, 1, bias=False),
        )

        # For stability
        self.eps = torch.tensor(1e-24).float()

    def forward(self, opt_algo):

        # Normalize gradient
        grad = opt_algo.loss_function.compute_gradient(opt_algo.current_state[1])
        grad_norm = torch.linalg.norm(grad).reshape((1,))
        if grad_norm > self.eps:
            grad = grad / grad_norm

        # Compute new hidden state
        diff = opt_algo.current_state[1] - opt_algo.current_state[0]
        diff_norm = torch.linalg.norm(diff).reshape((1,))
        if diff_norm > self.eps:
            diff = diff / diff_norm

        loss = opt_algo.loss_function(opt_algo.current_state[1].clone()).reshape((1,))
        old_loss = opt_algo.loss_function(opt_algo.current_state[1].clone()).reshape((1,))
        step_size = self.coefficients(
            torch.concat(
                (torch.log(1 + grad_norm.reshape((1,))),
                 torch.log(1 + diff_norm.reshape((1,))),
                 torch.log(1 + loss),
                 torch.log(1 + old_loss)
                 ))
        )
        direction = self.update_layer(torch.concat((
            grad.reshape((1, 1, 1, -1)),
            diff.reshape((1, 1, 1, -1)),
            (grad * diff).reshape((1, 1, 1, -1)),
        ), dim=1))

        return opt_algo.current_state[-1] + step_size * direction.flatten()

    @staticmethod
    def update_state(opt_algo):
        opt_algo.current_state[0] = opt_algo.current_state[1].detach().clone()
        opt_algo.current_state[1] = opt_algo.current_iterate.detach().clone()