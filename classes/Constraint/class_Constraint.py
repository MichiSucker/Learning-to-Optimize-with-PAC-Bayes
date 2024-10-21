from typing import Callable

import torch


class Constraint:

    def __init__(self, function: Callable):
        self.function = function

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> bool:
        return self.function(x, *args, **kwargs)
