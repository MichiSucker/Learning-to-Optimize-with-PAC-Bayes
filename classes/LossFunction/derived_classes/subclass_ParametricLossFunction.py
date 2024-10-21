from classes.LossFunction.class_LossFunction import LossFunction
from typing import Callable


class ParametricLossFunction(LossFunction):

    def __init__(self, function: Callable, parameter: dict):
        self.parameter = parameter
        super().__init__(function=lambda x: function(x, self.parameter))

    def get_parameter(self) -> dict:
        return self.parameter
