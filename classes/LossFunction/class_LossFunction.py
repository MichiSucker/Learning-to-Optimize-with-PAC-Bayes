from typing import Callable
import torch


class LossFunction:

    def __init__(self, function: Callable):
        self.function: Callable = function

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.function(x, *args, **kwargs)

    def compute_gradient(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        y = x.clone().detach().requires_grad_(True)
        function_value = self.function(y, *args, **kwargs)
        # If the gradient of y already got compute inside the function call (which might be necessary sometimes)
        # do not compute it again.
        if y.grad is None:
            function_value.backward()
        return y.grad
