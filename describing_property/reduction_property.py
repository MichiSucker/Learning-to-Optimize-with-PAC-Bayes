import torch
from classes.LossFunction.class_LossFunction import LossFunction
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from typing import Tuple, Callable, List

from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import \
    PacBayesOptimizationAlgorithm


def store_current_loss_function_state_and_iteration_counter(optimization_algorithm: OptimizationAlgorithm
                                                            ) -> Tuple[LossFunction, torch.Tensor, int]:
    return (optimization_algorithm.loss_function,
            optimization_algorithm.current_state,
            optimization_algorithm.iteration_counter)


def reset_loss_function_state_and_iteration_counter(optimization_algorithm: OptimizationAlgorithm,
                                                    loss_function: LossFunction,
                                                    state: torch.Tensor,
                                                    iteration_counter: int) -> None:
    optimization_algorithm.set_loss_function(loss_function)
    optimization_algorithm.set_current_state(state)
    optimization_algorithm.set_iteration_counter(iteration_counter)


def compute_loss_at_beginning_and_end(optimization_algorithm: OptimizationAlgorithm) -> Tuple[float, float]:
    loss_at_beginning = optimization_algorithm.evaluate_loss_function_at_current_iterate().item()
    _ = [optimization_algorithm.perform_step() for _ in range(optimization_algorithm.n_max)]
    loss_at_end = optimization_algorithm.evaluate_loss_function_at_current_iterate().item()
    return loss_at_beginning, loss_at_end


def instantiate_reduction_property_with(factor: float, exponent: float) -> Tuple[Callable, Callable, Callable]:

    def convergence_risk_constraint(loss_at_beginning: float, loss_at_end: float) -> bool:
        return loss_at_end <= factor * loss_at_beginning ** exponent

    def empirical_second_moment(list_of_loss_functions: List[LossFunction], point: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.stack([(factor * loss_function(point) ** exponent) ** 2
                                       for loss_function in list_of_loss_functions]))

    def reduction_property(loss_function_to_test: LossFunction,
                           optimization_algorithm: PacBayesOptimizationAlgorithm) -> bool:

        current_loss_function, current_state, current_iteration_counter = (
            store_current_loss_function_state_and_iteration_counter(optimization_algorithm))

        optimization_algorithm.reset_state_and_iteration_counter()
        optimization_algorithm.set_loss_function(new_loss_function=loss_function_to_test)
        loss_at_beginning, loss_at_end = compute_loss_at_beginning_and_end(optimization_algorithm)

        reset_loss_function_state_and_iteration_counter(optimization_algorithm=optimization_algorithm,
                                                        loss_function=current_loss_function,
                                                        state=current_state,
                                                        iteration_counter=current_iteration_counter)

        return convergence_risk_constraint(loss_at_beginning, loss_at_end)

    return reduction_property, convergence_risk_constraint, empirical_second_moment
