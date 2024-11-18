from typing import Callable
import torch
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction


class NonsmoothParametricLossFunction(ParametricLossFunction):

    def __init__(self, function: Callable, smooth_part: Callable, nonsmooth_part: Callable, parameter: dict):
        self.parameter = parameter
        self.whole_function = function
        self.smooth_part = ParametricLossFunction(function=smooth_part, parameter=self.parameter)
        self.nonsmooth_part = ParametricLossFunction(function=nonsmooth_part, parameter=self.parameter)
        super().__init__(function=self.whole_function, parameter=parameter)

    def compute_gradient_of_smooth_part(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.smooth_part.compute_gradient(x, *args, **kwargs)

    def compute_gradient_of_nonsmooth_part(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.nonsmooth_part.compute_gradient(x, *args, **kwargs)
