from typing import Callable
import torch
from classes.LossFunction.class_LossFunction import LossFunction
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import \
    PacBayesOptimizationAlgorithm
from exponential_family.describing_property.reduction_property import compute_loss_at_beginning_and_end


def evaluate_sufficient_statistics(optimization_algorithm: PacBayesOptimizationAlgorithm,
                                   loss_function: LossFunction,
                                   constants: torch.Tensor,
                                   convergence_risk_constraint: Callable,
                                   convergence_probability: torch.Tensor) -> torch.Tensor:

    optimization_algorithm.reset_state_and_iteration_counter()
    optimization_algorithm.set_loss_function(loss_function)
    loss_at_beginning, loss_at_end = compute_loss_at_beginning_and_end(optimization_algorithm)

    if convergence_risk_constraint(loss_at_beginning, loss_at_end):
        return torch.tensor([-loss_at_end/convergence_probability, constants / (convergence_probability**2)])
    else:
        # By looking into the proof, we see that we can only need to count
        # the constant corresponding to the problems, where the algorithm has the convergence property.
        return torch.tensor([0.0, 0.0])
