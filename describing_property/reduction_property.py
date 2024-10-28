import torch
from classes.LossFunction.class_LossFunction import LossFunction
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from typing import Tuple

from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import \
    PacBayesOptimizationAlgorithm


def store_current_loss_function_state_and_iteration_counter(optimization_algorithm):
    return (optimization_algorithm.loss_function,
            optimization_algorithm.current_state,
            optimization_algorithm.iteration_counter)


def reset_loss_function_state_and_iteration_counter(optimization_algorithm, loss_function, state, iteration_counter):
    optimization_algorithm.set_loss_function(loss_function)
    optimization_algorithm.set_current_state(state)
    optimization_algorithm.set_iteration_counter(iteration_counter)


def compute_loss_at_beginning_and_end(optimization_algorithm):
    loss_at_beginning = optimization_algorithm.evaluate_loss_function_at_current_iterate().item()
    _ = [optimization_algorithm.perform_step() for _ in range(optimization_algorithm.n_max)]
    loss_at_end = optimization_algorithm.evaluate_loss_function_at_current_iterate().item()
    return loss_at_beginning, loss_at_end


def instantiate_reduction_property_with(factor, exponent):

    def convergence_risk_constraint(loss_at_beginning, loss_at_end):
        return loss_at_end <= factor * loss_at_beginning ** exponent

    def empirical_second_moment(list_of_loss_functions, point):
        return torch.mean(torch.stack([(factor * loss_function(point) ** exponent) ** 2
                                       for loss_function in list_of_loss_functions]))

    def reduction_property(loss_function_to_test, optimization_algorithm: PacBayesOptimizationAlgorithm):

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

# def get_reducing_property(factor: float, exponent: float) -> Tuple:
#
#     def convergence_risk_constraint(losses):
#         return losses[-1] <= factor * losses[0] ** exponent
#
#     def empirical_second_moment(list_of_loss_functions, point):
#         return torch.mean(torch.stack([
#                     (factor * loss_function(point) ** exponent) ** 2 for loss_function in list_of_loss_functions]))
#
#     def reducing_property(f: LossFunction, opt_algo: OptimizationAlgorithm) -> bool:
#
#         # Store current state, loss function, etc.
#         cur_state, cur_loss = opt_algo.current_state, opt_algo.loss_function
#         cur_iteration_count = opt_algo.iteration_counter
#         opt_algo.reset_state()
#
#         # Set new loss function and compute corresponding losses
#         opt_algo.set_loss_function(f)
#         # losses = [f(opt_algo.current_state[-1]).item()] + [f(it).item() for it in opt_algo]
#         iterates, did_converge = opt_algo.compute_trajectory(num_steps=opt_algo.n_max, check_convergence=True)
#
#         # Take losses until convergence
#         losses = []
#         for k, x in enumerate(iterates):
#             losses.append(f(x).item())
#             if did_converge[k]:
#                 break
#         losses = torch.tensor(losses)
#
#         # Reset current state, loss function, etc.
#         opt_algo.set_current_state(cur_state)
#         opt_algo.set_loss_function(cur_loss)
#         opt_algo.set_iteration_counter(cur_iteration_count)
#
#         return convergence_risk_constraint(losses)
#
#     return reducing_property, convergence_risk_constraint, empirical_second_moment