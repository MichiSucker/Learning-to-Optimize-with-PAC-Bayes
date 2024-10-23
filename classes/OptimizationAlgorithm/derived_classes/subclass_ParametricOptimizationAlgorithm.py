import torch

from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.Constraint.class_Constraint import Constraint
from classes.LossFunction.class_LossFunction import LossFunction
import copy
from tqdm import tqdm
import numpy as np


class TrainingAssistant:

    def __init__(self, printing_enabled, print_update_every, maximal_number_of_iterations, update_stepsize_every,
                 factor_update_stepsize, bins=None):
        self.printing_enabled = printing_enabled
        self.print_update_every = print_update_every
        self.maximal_number_of_iterations = maximal_number_of_iterations
        self.update_stepsize_every = update_stepsize_every
        self.factor_update_stepsize = factor_update_stepsize
        self.running_loss = 0
        self.loss_histogram = []
        if not bins:
            self.bins = [1e0, 1e-4, 1e-8, 1e-12, 1e-16, 1e-20, 1e-24, 1e-28][::-1]
        else:
            self.bins = bins

    def starting_message(self):
        if self.printing_enabled:
            print("---------------------------------------------------------------------------------------------------")
            print("Fit Algorithm:")
            print("---------------------------------------------------------------------------------------------------")
            print(f"\t-Optimizing for {self.maximal_number_of_iterations} iterations.")
            print(f"\t-Updating step-size every {self.update_stepsize_every} iterations.")

    def final_message(self):
        if self.printing_enabled:
            print("---------------------------------------------------------------------------------------------------")
            print("End Fitting Algorithm.")

    def should_update_stepsize_of_optimizer(self, iteration):
        if (iteration >= 1) and (iteration % self.update_stepsize_every == 0):
            return True
        else:
            return False

    def update_stepsize_of_optimizer(self, optimizer):
        for g in optimizer.param_groups:
            g['lr'] = self.factor_update_stepsize * g['lr']

    def get_bins(self):
        return self.bins

    def set_bins(self, new_bins):
        self.bins = new_bins

    def should_print_update(self, iteration):
        if iteration >= 1 and self.printing_enabled and iteration % self.print_update_every == 0:
            return True
        else:
            return False

    def print_update(self, iteration, constraint_checker):
        print("\t\t\t-----------------------------------------------------------------------------------------")
        print(f"\t\t\tIteration: {iteration}; Found point inside constraint: "
              f"{constraint_checker.found_point_inside_constraint}")
        vals, bins = np.histogram(self.loss_histogram, bins=self.bins)
        print(f"\t\t\t\tLosses:")
        for j in range(len(vals) - 1, -1, -1):
            print(f"\t\t\t\t\t[{bins[j + 1]:.0e}, {bins[j]:.0e}] : {vals[j]}/{self.print_update_every}")
        print("\t\t\t-----------------------------------------------------------------------------------------")

    def reset_running_loss_and_loss_histogram(self):
        self.loss_histogram = []
        self.running_loss = 0


class ConstraintChecker:

    def __init__(self, check_constraint_every, there_is_a_constraint):
        self.check_constraint_every = check_constraint_every
        self.there_is_a_constraint = there_is_a_constraint
        self.found_point_inside_constraint = False
        self.point_inside_constraint = None

    def should_check_constraint(self, iteration_number):
        if (self.there_is_a_constraint
            and (iteration_number >= 1)
                and (iteration_number % self.check_constraint_every == 0)):
            return True
        else:
            return False

    def set_there_is_a_constraint(self, new_bool):
        self.there_is_a_constraint = new_bool

    def set_check_constraint_every(self, new_number):
        self.check_constraint_every = new_number

    def update_point_inside_constraint_or_reject(self, optimization_algorithm):
        satisfies_constraint = check_constraint(optimization_algorithm)

        if satisfies_constraint:
            self.found_point_inside_constraint = True
            self.point_inside_constraint = copy.deepcopy(optimization_algorithm.implementation.state_dict())

        elif self.found_point_inside_constraint and (not satisfies_constraint):
            optimization_algorithm.implementation.load_state_dict(self.point_inside_constraint)

    def final_check(self, optimization_algorithm):
        satisfies_constraint = check_constraint(optimization_algorithm)
        if satisfies_constraint:
            return
        elif self.found_point_inside_constraint and (not satisfies_constraint):
            optimization_algorithm.implementation.load_state_dict(self.point_inside_constraint)
        else:
            raise Exception("Did not find a point that lies within the constraint!")


def check_constraint(optimization_algorithm):
    return optimization_algorithm.constraint(optimization_algorithm)


class TrajectoryRandomizer:

    def __init__(self, should_restart, restart_probability, length_partial_trajectory):
        self.should_restart = should_restart
        self.restart_probability = restart_probability
        self.length_partial_trajectory = length_partial_trajectory

    def get_should_restart(self):
        return self.should_restart

    def set_should_restart(self, should_restart):
        if not isinstance(should_restart, bool):
            raise TypeError("Type of 'should_restart' has to be bool.")
        self.should_restart = should_restart

    def get_restart_probability(self):
        return self.restart_probability


class ParametricOptimizationAlgorithm(OptimizationAlgorithm):

    def __init__(self,
                 initial_state: torch.Tensor,
                 implementation,
                 loss_function: LossFunction,
                 constraint: Constraint = None):

        super().__init__(initial_state=initial_state, implementation=implementation, loss_function=loss_function,
                         constraint=constraint)

    def fit(self,
            loss_functions: list,
            fitting_parameters: dict,
            constraint_parameters: dict,
            update_parameters: dict
            ) -> None:

        optimizer, training_assistant, trajectory_randomizer, constraint_checker = self.initialize_helpers_for_training(
            fitting_parameters=fitting_parameters,
            constraint_parameters=constraint_parameters,
            update_parameters=update_parameters
        )
        training_assistant.starting_message()

        pbar = tqdm(range(training_assistant.maximal_number_of_iterations))
        pbar.set_description('Fit Algorithm')
        for i in pbar:

            if training_assistant.should_update_stepsize_of_optimizer(iteration=i):
                training_assistant.update_stepsize_of_optimizer(optimizer)

            self.update_hyperparameters(optimizer=optimizer,
                                        trajectory_randomizer=trajectory_randomizer,
                                        loss_functions=loss_functions,
                                        training_assistant=training_assistant)

            if training_assistant.should_print_update(i):
                training_assistant.print_update(iteration=i, constraint_checker=constraint_checker)
                training_assistant.reset_running_loss_and_loss_histogram()

            if constraint_checker.should_check_constraint(i):
                constraint_checker.update_point_inside_constraint_or_reject(self)

        constraint_checker.final_check(self)
        self.reset_state_and_iteration_counter()
        training_assistant.final_message()

    def initialize_helpers_for_training(self, fitting_parameters, constraint_parameters, update_parameters):

        trajectory_randomizer = TrajectoryRandomizer(
            should_restart=True,
            restart_probability=fitting_parameters['restart_probability'],
            length_partial_trajectory=fitting_parameters['length_trajectory']
        )

        constraint_checker = ConstraintChecker(
            check_constraint_every=constraint_parameters['num_iter_update_constraint'],
            there_is_a_constraint=self.constraint is not None
        )

        training_assistant = TrainingAssistant(
            printing_enabled=update_parameters['with_print'],
            print_update_every=update_parameters['num_iter_print_update'],
            maximal_number_of_iterations=fitting_parameters['n_max'],
            update_stepsize_every=fitting_parameters['num_iter_update_stepsize'],
            factor_update_stepsize=fitting_parameters['factor_stepsize_update'],
            bins=update_parameters['bins']
        )

        optimizer = torch.optim.Adam(self.implementation.parameters(), lr=fitting_parameters['lr'])

        return optimizer, training_assistant, trajectory_randomizer, constraint_checker

    def update_hyperparameters(self, optimizer, trajectory_randomizer, loss_functions, training_assistant):
        optimizer.zero_grad()
        self.determine_next_starting_point(
            trajectory_randomizer=trajectory_randomizer, loss_functions=loss_functions)
        predicted_iterates = self.compute_partial_trajectory(
            number_of_steps=trajectory_randomizer.length_partial_trajectory)
        ratios_of_losses = self.compute_ratio_of_losses(predicted_iterates=predicted_iterates)
        if losses_are_invalid(ratios_of_losses):
            print('Invalid losses.')
            return
        sum_losses = torch.sum(torch.stack(ratios_of_losses))
        sum_losses.backward()
        optimizer.step()

        with torch.no_grad():
            training_assistant.loss_histogram.append(self.loss_function(predicted_iterates[-1]).item())
            training_assistant.running_loss += sum_losses

    def determine_next_starting_point(self, trajectory_randomizer, loss_functions):
        if trajectory_randomizer.should_restart:
            self.restart_with_new_loss(loss_functions=loss_functions)
            trajectory_randomizer.set_should_restart(False)
        else:
            self.detach_current_state_from_computational_graph()
            trajectory_randomizer.set_should_restart((torch.rand(1) <= trajectory_randomizer.restart_probability).item())

    def restart_with_new_loss(self, loss_functions):
        self.reset_state_and_iteration_counter()
        self.set_loss_function(np.random.choice(loss_functions))

    def detach_current_state_from_computational_graph(self):
        x_0 = self.current_state.detach().clone()
        self.current_state = x_0

    def compute_ratio_of_losses(self, predicted_iterates):
        ratios = [self.loss_function(predicted_iterates[k]) / self.loss_function(predicted_iterates[k - 1])
                  for k in range(1, len(predicted_iterates))]
        return ratios


def losses_are_invalid(losses):
    if (len(losses) == 0) or (None in losses) or (torch.inf in losses):
        return True
    return False

