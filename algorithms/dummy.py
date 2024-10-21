import torch
import torch.nn as nn


class Dummy(nn.Module):

    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self):
        pass

    @staticmethod
    def update_state(opt_algo):
        pass
