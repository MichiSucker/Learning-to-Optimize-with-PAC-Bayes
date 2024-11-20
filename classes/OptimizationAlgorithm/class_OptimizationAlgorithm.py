import torch
import torch.nn as nn
from typing import Callable, List
from classes.LossFunction.class_LossFunction import LossFunction
from classes.LossFunction.derived_classes.derived_classes.\
    subclass_NonsmoothParametricLossFunction import NonsmoothParametricLossFunction
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction


class OptimizationAlgorithm:

    def __init__(self,
                 implementation: nn.Module,
                 initial_state: torch.Tensor,
                 loss_function: LossFunction | ParametricLossFunction | NonsmoothParametricLossFunction,
                 constraint: Callable = None):
        self.implementation = implementation
        self.loss_function = loss_function
        self.initial_state = initial_state.clone()
        self.current_state = initial_state.clone()
        self.current_iterate = self.current_state[-1]
        self.iteration_counter = 0
        self.constraint = constraint
        self.n_max = None

    def get_initial_state(self) -> torch.Tensor:
        return self.initial_state

    def get_implementation(self) -> nn.Module:
        return self.implementation

    def get_current_state(self) -> torch.Tensor:
        return self.current_state

    def get_current_iterate(self) -> torch.Tensor:
        return self.current_iterate

    def get_iteration_counter(self) -> int:
        return self.iteration_counter

    def set_iteration_counter(self, n: int) -> None:
        if not isinstance(n, int):
            raise TypeError('Iteration counter has to be a non-negative integer.')
        self.iteration_counter = n

    def reset_iteration_counter_to_zero(self) -> None:
        self.iteration_counter = 0

    def reset_to_initial_state(self) -> None:
        self.set_current_state(self.initial_state.clone())

    def reset_state_and_iteration_counter(self) -> None:
        self.reset_to_initial_state()
        self.reset_iteration_counter_to_zero()

    def set_current_state(self, new_state: torch.Tensor) -> None:
        if new_state.shape != self.current_state.shape:
            raise ValueError('Shape of new state does not match shape of current state.')
        self.current_state = new_state.clone()
        self.current_iterate = self.current_state[-1]

    def set_constraint(self, function: Callable) -> None:
        self.constraint = function

    def perform_step(self, return_iterate=False) -> None | torch.Tensor:
        self.iteration_counter += 1
        self.current_iterate = self.implementation.forward(self)
        with torch.no_grad():
            self.implementation.update_state(self)
        if return_iterate:
            return self.current_iterate

    def compute_partial_trajectory(self, number_of_steps: int) -> List[torch.Tensor]:
        trajectory = [self.current_state[-1].clone()] + [self.perform_step(return_iterate=True)
                                                         for _ in range(number_of_steps)]
        return trajectory

    def set_loss_function(self, new_loss_function: Callable) -> None:
        self.loss_function = new_loss_function

    def evaluate_loss_function_at_current_iterate(self) -> torch.Tensor:
        return self.loss_function(self.current_iterate)

    def evaluate_constraint(self) -> bool:
        if self.constraint is not None:
            return self.constraint(self)
