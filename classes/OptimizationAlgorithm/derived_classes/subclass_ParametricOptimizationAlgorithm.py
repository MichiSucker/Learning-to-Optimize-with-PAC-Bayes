import torch

from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.Constraint.class_Constraint import Constraint
from classes.LossFunction.class_LossFunction import LossFunction
import copy
from tqdm import tqdm
import numpy as np


class TrainingAssistant:
    # To-Do: Collect all print statements and stuff inside here. This is mainly for cleaning-up.
    pass


class ConstraintChecker:

    def __init__(self, check_constraint_every, there_is_a_constraint):
        self.check_constraint_every = check_constraint_every
        self.there_is_a_constraint = there_is_a_constraint
        self.found_point_inside_constraint = False
        self.point_inside_constraint = None

    def constraint_has_to_be_checked(self, iteration_number):
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

    def __init__(self, should_restart, restart_probability):
        self.should_restart = should_restart
        self.restart_probability = restart_probability

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

        # Extract parameters for prints during training
        num_iter_print_update = update_parameters['num_iter_print_update']
        with_print = update_parameters['with_print']
        bins = update_parameters['bins']

        # Extract parameters for fitting
        num_iter_max = fitting_parameters['n_max']
        num_iter_update_stepsize = fitting_parameters['num_iter_update_stepsize']
        lr = fitting_parameters['lr']
        length_trajectory = fitting_parameters['length_trajectory']
        factor_stepsize_update = fitting_parameters['factor_stepsize_update']

        trajectory_randomizer = TrajectoryRandomizer(
            should_restart=True, restart_probability=fitting_parameters['restart_probability'])
        constraint_checker = ConstraintChecker(
            check_constraint_every=constraint_parameters['num_iter_update_constraint'],
            there_is_a_constraint=self.constraint is not None)

        if bins is None:
            bins = [1e0, 1e-4, 1e-8, 1e-12, 1e-16, 1e-20, 1e-24, 1e-28][::-1]
        optimizer = torch.optim.Adam(self.implementation.parameters(), lr=lr)
        i, running_loss = 0, 0
        if with_print:
            print("---------------------------------------------------------------------------------------------------")
            print("Fit Algorithm:")
            print("---------------------------------------------------------------------------------------------------")
            print(f"\t-Optimizing for {num_iter_max} iterations.")
            print(f"\t-Updating step-size every {num_iter_update_stepsize} iterations.")

        update_histogram = []

        i = 0
        pbar = tqdm(total=num_iter_max)
        pbar.set_description('Fit Algorithm')

        while i < num_iter_max:

            if should_update_stepsize_of_optimizer(i=i, update_stepsize_every=num_iter_update_stepsize):
                update_stepsize_of_optimizer(optimizer, factor=factor_stepsize_update)

            self.update_hyperparameters(optimizer=optimizer, trajectory_randomizer=trajectory_randomizer,
                                        loss_functions=loss_functions, length_trajectory=length_trajectory)

            with torch.no_grad():
                update_histogram.append(self.loss_function(predicted_iterates[-1]).item())
                running_loss += sum_losses

            # Update Statements
            if i >= 1 and with_print and i % num_iter_print_update == 0:
                print("\t\t\t-----------------------------------------------------------------------------------------")
                print(f"\t\t\tIteration: {i}; Found point inside constraint: "
                      f"{constraint_checker.found_point_inside_constraint}")
                print("\t\t\t\tAvg. Loss = {:.2f}".format(running_loss / (num_iter_print_update * length_trajectory)))
                vals, bins = np.histogram(update_histogram, bins=bins)
                print(f"\t\t\t\tRatios:")
                for j in range(len(vals) - 1, -1, -1):
                    print(f"\t\t\t\t\t[{bins[j + 1]:.0e}, {bins[j]:.0e}] : {vals[j]}/{num_iter_print_update}")
                print("\t\t\t-----------------------------------------------------------------------------------------")

                # Reset Variables
                update_histogram = []
                running_loss = 0

            if constraint_checker.constraint_has_to_be_checked(i):
                constraint_checker.update_point_inside_constraint_or_reject(self)

            # Update Iteration Counter
            i += 1
            pbar.update(1)

        constraint_checker.final_check(self)
        self.reset_state_and_iteration_counter()

        if with_print:
            print("---------------------------------------------------------------------------------------------------")
            print("End Fitting Algorithm.")

    def update_hyperparameters(self, optimizer, trajectory_randomizer, loss_functions, length_trajectory):
        optimizer.zero_grad()
        self.determine_next_starting_point(
            trajectory_randomizer=trajectory_randomizer, loss_functions=loss_functions)
        predicted_iterates = self.compute_trajectory(number_of_steps=length_trajectory)
        ratios_of_losses = self.compute_ratio_of_losses(predicted_iterates=predicted_iterates)
        if losses_are_invalid(ratios_of_losses):
            print('Invalid losses.')
            return
        sum_losses = torch.sum(torch.stack(ratios_of_losses))
        sum_losses.backward()
        optimizer.step()

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


def should_update_stepsize_of_optimizer(i, update_stepsize_every):
    if (i >= 1) and (i % update_stepsize_every == 0):
        return True
    return False


def update_stepsize_of_optimizer(optimizer, factor):
    for g in optimizer.param_groups:
        g['lr'] = factor * g['lr']


def losses_are_invalid(losses):
    if (len(losses) == 0) or (None in losses) or (torch.inf in losses):
        return True
    return False

