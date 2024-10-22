import torch
import torch.nn as nn


class Dummy(nn.Module):

    def __init__(self):
        super(Dummy, self).__init__()
        self.scale = nn.Parameter(torch.tensor(1.))

    def forward(self, optimization_algorithm):
        current_iterate = optimization_algorithm.get_current_iterate()
        return current_iterate + self.scale * torch.randn(size=current_iterate.shape)

    @staticmethod
    def update_state(optimization_algorithm):
        optimization_algorithm.current_state = optimization_algorithm.current_iterate.detach().clone().reshape((1, -1))
