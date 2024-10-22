import torch
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.Constraint.class_Constraint import Constraint
from classes.LossFunction.class_LossFunction import LossFunction
import copy
from tqdm import tqdm
import numpy as np


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
            update_parameters: dict,
            found_point_inside_constraint: bool = False
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

        trajectory_randomizer = TrajectoryRandomizer(should_restart=True,
                                                     restart_probability=fitting_parameters['restart_probability'])

        # Extract parameters of constraint set
        num_iter_update_constraint = constraint_parameters['num_iter_update_constraint']

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

        if found_point_inside_constraint:
            old_state_dict = copy.deepcopy(self.implementation.state_dict())
        else:
            old_state_dict = None

        rejected, num_iter_update_rejection_rate = 0, 20
        i, t = 0, 0
        pbar = tqdm(total=num_iter_max)
        pbar.set_description('Fit Algorithm')

        while i < num_iter_max:

            if (t >= 1) and (t % num_iter_update_rejection_rate == 0):
                if rejected / num_iter_update_rejection_rate >= 0.5:
                    print("Decrease Learning Rate.")
                    for g in optimizer.param_groups:
                        g['lr'] = 0.1 * g['lr']
                rejected = 0

            # Update Stepsize
            if t >= 1 and t % num_iter_update_stepsize == 0:
                if with_print:
                    print("\t\t\t------------------")
                    print("\t\t\tUpdating Stepsize.")
                    print("\t\t\t------------------")
                for g in optimizer.param_groups:
                    g['lr'] = factor_stepsize_update * g['lr']

            # Increase Counter for t
            # This should prevent getting stuck, because the step-size gets decreased
            t += 1

            # Reset optimizer
            optimizer.zero_grad()
            self.determine_next_starting_point(
                trajectory_randomizer=trajectory_randomizer, loss_functions=loss_functions)
            predicted_iterates = self.compute_trajectory(number_of_steps=length_trajectory)

            new_loss = self.loss_function(predicted_iterates[-1]).item()
            update_histogram.append(new_loss)

            ratios_of_losses = self.compute_ratio_of_losses(predicted_iterates=predicted_iterates)
            if losses_are_invalid(ratios_of_losses):
                print('Invalid losses.')
                continue
            sum_losses = torch.sum(torch.stack(ratios_of_losses))
            sum_losses.backward()

            with torch.no_grad():
                running_loss += sum_losses

            # Update Statements
            if i >= 1 and with_print and i % num_iter_print_update == 0:
                print("\t\t\t-----------------------------------------------------------------------------------------")
                print(f"\t\t\tIteration: {i}; Found point inside constraint: {found_point_inside_constraint}")
                print("\t\t\t\tAvg. Loss = {:.2f}".format(running_loss / (num_iter_print_update * length_trajectory)))
                vals, bins = np.histogram(update_histogram, bins=bins)
                print(f"\t\t\t\tRatios:")
                for j in range(len(vals) - 1, -1, -1):
                    print(f"\t\t\t\t\t[{bins[j + 1]:.0e}, {bins[j]:.0e}] : {vals[j]}/{num_iter_print_update}")
                print("\t\t\t-----------------------------------------------------------------------------------------")

                # Reset Variables
                update_histogram = []
                running_loss = 0

            # Update hyperparameters
            optimizer.step()

            # Check constraint. Reject current step if it is not satisfied.
            if (i >= 1) and (self.constraint is not None) and (i % num_iter_update_constraint == 0):
                print("Checking Constraint.")
                satisfies_constraint = self.constraint(self)
                print(f"Check over. Constraint satisfied: {satisfies_constraint}.")

                # If constraint is satisfied, just update
                if satisfies_constraint:
                    print("Found new point that satisfies constraint.")
                    found_point_inside_constraint = True
                    old_state_dict = copy.deepcopy(self.implementation.state_dict())

                # If constrained is not satisfied, and one has found a point before that does satisfy the constraint,
                # then reject the new point.
                elif found_point_inside_constraint and (not satisfies_constraint):
                    self.implementation.load_state_dict(old_state_dict)
                    rejected += 1
                    print("Reject.")
                    #continue  # Why do we continue here again? It only removes the two lines below.

                # If constraint is not satisfied and one has not found a point before that does satisfy the constraint,
                # just do nothing here, as this corresponds to accepting the new point.
            #
            # pr.disable()
            # pr.print_stats()

            # Update Iteration Counter
            i += 1
            pbar.update(1)

        # Final check of constraint. If it is not satisfied, print a warning (without stopping the program).
        satisfies_constraint = self.constraint(self)
        if satisfies_constraint:
            print("Final point lies within the constraint set.")
        elif found_point_inside_constraint and (not satisfies_constraint):
            self.implementation.load_state_dict(old_state_dict)
            print("Took point from before that satisfies constraint.")
        else:
            raise Exception("Did not find a point that lies within the constraint!")

        # Reset algorithm
        self.reset_state_and_iteration_counter()
        if with_print:
            print("---------------------------------------------------------------------------------------------------")
            print("End Fitting Algorithm.")

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

