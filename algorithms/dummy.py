import torch
import torch.nn as nn


class Dummy(nn.Module):

    def __init__(self):
        super(Dummy, self).__init__()
        self.scale = nn.Parameter(torch.tensor(1.))

    def forward(self, optimization_algorithm):
        gradient = optimization_algorithm.loss_function.compute_gradient(optimization_algorithm.current_iterate)
        return optimization_algorithm.current_iterate + self.scale * gradient

    @staticmethod
    def update_state(optimization_algorithm):
        optimization_algorithm.current_state = optimization_algorithm.current_iterate.detach().clone().reshape((1, -1))
