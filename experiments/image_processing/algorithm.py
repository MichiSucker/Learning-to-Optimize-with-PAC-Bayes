import torch
import torch.nn as nn


class ConvNet(nn.Module):

    def __init__(self, img_height: torch.tensor, img_width: torch.tensor, kernel_size: int):
        super(ConvNet, self).__init__()

        self.width = img_width
        self.height = img_height
        self.shape = (1, 1, img_height, img_width)
        self.kernel_size = kernel_size
        padding_mode = 'reflect'

        in_channels = 3
        hidden_channels = 16
        out_channels = 1
        self.update_layer = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode=padding_mode, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode=padding_mode, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode=padding_mode, bias=False)
        )

        in_size = 2
        hidden_size = 8
        out_size = 1
        self.step_size_layer = nn.Sequential(
            nn.Linear(in_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size, bias=False),
        )

        # For stability
        self.eps = torch.tensor(1e-10).float()

    def forward(self, opt_algo):

        # Compute and normalize gradient
        grad = opt_algo.loss_function.grad(opt_algo.current_state[1])
        grad_norm = torch.linalg.norm(grad).reshape((1,))
        if grad_norm > self.eps:
            grad = grad/grad_norm

        # Compute and normalize momentum term
        diff = opt_algo.current_state[1] - opt_algo.current_state[0]
        diff_norm = torch.linalg.norm(diff).reshape((1,))
        if diff_norm > self.eps:
            diff = diff/diff_norm

        update_direction = self.update_layer(torch.concat((
            grad.reshape((1, 1, self.height, self.width)),
            diff.reshape((1, 1, self.height, self.width)),
            (grad * diff).reshape((1, 1, self.height, self.width))), dim=1))

        step_size = self.step_size_layer(torch.concat((
            torch.log(1 + grad_norm),
            torch.log(1 + diff_norm))))

        return opt_algo.current_state[-1] + step_size * update_direction.flatten()

    @staticmethod
    def update_state(opt_algo):
        opt_algo.current_state[0] = opt_algo.current_state[1].detach().clone()
        opt_algo.current_state[1] = opt_algo.current_iterate.detach().clone()
