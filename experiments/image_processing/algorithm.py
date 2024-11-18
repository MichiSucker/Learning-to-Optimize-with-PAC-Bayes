import torch
import torch.nn as nn

from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm


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

    def forward(self, opt_algo: OptimizationAlgorithm) -> torch.Tensor:

        # Compute and normalize gradient
        gradient = opt_algo.loss_function.compute_gradient(opt_algo.current_state[1])
        gradient_norm = torch.linalg.norm(gradient).reshape((1,))
        if gradient_norm > self.eps:
            gradient = gradient / gradient_norm

        # Compute and normalize momentum term
        difference = opt_algo.current_state[1] - opt_algo.current_state[0]
        difference_norm = torch.linalg.norm(difference).reshape((1,))
        if difference_norm > self.eps:
            difference = difference / difference_norm

        update_direction = self.update_layer(torch.concat((
            gradient.reshape((1, 1, self.height, self.width)),
            difference.reshape((1, 1, self.height, self.width)),
            (gradient * difference).reshape((1, 1, self.height, self.width))), dim=1))

        step_size = self.step_size_layer(torch.concat((
            torch.log(1 + gradient_norm),
            torch.log(1 + difference_norm))))

        return opt_algo.current_state[-1] + step_size * update_direction.flatten()

    @staticmethod
    def update_state(opt_algo: OptimizationAlgorithm) -> None:
        opt_algo.current_state[0] = opt_algo.current_state[1].detach().clone()
        opt_algo.current_state[1] = opt_algo.current_iterate.detach().clone()
