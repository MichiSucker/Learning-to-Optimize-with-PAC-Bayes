from typing import Callable
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm


class Constraint:

    def __init__(self, function: Callable):
        self.function = function

    def __call__(self, optimization_algorithm: OptimizationAlgorithm, *args, **kwargs) -> bool:
        return self.function(optimization_algorithm, *args, **kwargs)


def create_list_of_constraints_from_functions(describing_property, list_of_functions):
    list_of_constraints = [
        Constraint(function=lambda opt_algo, i=i: describing_property(function=list_of_functions[i],
                                                                      optimization_algorithm=opt_algo))
        for i in range(len(list_of_functions))]
    return list_of_constraints
