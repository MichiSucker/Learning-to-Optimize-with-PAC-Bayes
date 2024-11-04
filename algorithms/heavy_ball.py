import torch
import torch.nn as nn


class HeavyBallWithFriction(nn.Module):

    def __init__(self, alpha: torch.tensor, beta: torch.tensor):
        super(HeavyBallWithFriction, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, opt_algo):
        return (opt_algo.current_state[1]
                - self.alpha * opt_algo.loss_function.compute_gradient(opt_algo.current_state[1])
                + self.beta * (opt_algo.current_state[1] - opt_algo.current_state[0]))

    @staticmethod
    def update_state(opt_algo):
        opt_algo.current_state[0] = opt_algo.current_state[1].detach().clone()
        opt_algo.current_state[1] = opt_algo.current_iterate.detach().clone()
