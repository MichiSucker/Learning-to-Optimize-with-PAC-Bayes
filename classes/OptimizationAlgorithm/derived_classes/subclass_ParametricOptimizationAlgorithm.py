import torch
import torch.nn as nn
from typing import Tuple, List
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.Constraint.class_Constraint import Constraint
from classes.LossFunction.class_LossFunction import LossFunction
import copy
from tqdm import tqdm
import numpy as np
from torch.distributions import MultivariateNormal


class SamplingAssistant:

    def __init__(self, learning_rate, desired_number_of_samples, number_of_iterations_burnin):
        self.initial_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.number_of_correct_samples = 0
        self.desired_number_of_samples = desired_number_of_samples
        self.number_of_iterations_burnin = number_of_iterations_burnin
        self.samples = []
        self.samples_state_dict = []
        self.estimated_probabilities = []

    def decay_learning_rate(self, iteration):
        self.current_learning_rate = self.initial_learning_rate / iteration

    def get_progressbar(self):
        pbar = tqdm(total=self.desired_number_of_samples + self.number_of_iterations_burnin)
        pbar.set_description('Sampling')
        return pbar

    def should_continue(self):
        return self.number_of_correct_samples < self.desired_number_of_samples + self.number_of_iterations_burnin

    def should_store_sample(self, iteration):
        return iteration >= self.number_of_iterations_burnin

    def store_sample(self, implementation, estimated_probability):
        self.samples.append([p.detach().clone() for p in implementation.parameters() if p.requires_grad])
        self.samples_state_dict.append(copy.deepcopy(implementation.state_dict()))
        self.estimated_probabilities.append(estimated_probability)
        self.number_of_correct_samples += 1

    def prepare_output(self):
        if any((len(self.samples) < self.desired_number_of_samples,
                len(self.samples_state_dict) < self.desired_number_of_samples,
                len(self.estimated_probabilities) < self.desired_number_of_samples)):
            raise Exception("Did not find enough samples to prepare output.")

        if ((len(self.samples) != len(self.samples_state_dict))
                or (len(self.samples) != len(self.estimated_probabilities))):
            raise Exception("Something went wrong: Number of samples do not match up.")

        return (self.samples[-self.desired_number_of_samples:],
                self.samples_state_dict[-self.desired_number_of_samples:],
                self.estimated_probabilities[-self.desired_number_of_samples:])


class ConstraintChecker:

    def __init__(self,
                 check_constraint_every: int,
                 there_is_a_constraint: bool):
        self.check_constraint_every = check_constraint_every
        self.there_is_a_constraint = there_is_a_constraint
        self.found_point_inside_constraint = False
        self.point_inside_constraint = None

    def should_check_constraint(self, iteration_number: int) -> bool:
        if (self.there_is_a_constraint
            and (iteration_number >= 1)
                and (iteration_number % self.check_constraint_every == 0)):
            return True
        else:
            return False

    def set_there_is_a_constraint(self, new_bool: bool):
        self.there_is_a_constraint = new_bool

    def set_check_constraint_every(self, new_number: int):
        self.check_constraint_every = new_number

    def update_point_inside_constraint_or_reject(self, optimization_algorithm: OptimizationAlgorithm):
        satisfies_constraint = check_constraint(optimization_algorithm)

        if satisfies_constraint:
            self.found_point_inside_constraint = True
            self.point_inside_constraint = copy.deepcopy(optimization_algorithm.implementation.state_dict())

        elif self.found_point_inside_constraint and (not satisfies_constraint):
            optimization_algorithm.implementation.load_state_dict(self.point_inside_constraint)

    def final_check(self, optimization_algorithm: OptimizationAlgorithm):
        satisfies_constraint = check_constraint(optimization_algorithm)
        if satisfies_constraint:
            return
        elif self.found_point_inside_constraint and (not satisfies_constraint):
            optimization_algorithm.implementation.load_state_dict(self.point_inside_constraint)
        else:
            raise Exception("Did not find a point that lies within the constraint!")


def check_constraint(optimization_algorithm: OptimizationAlgorithm) -> bool:
    return optimization_algorithm.constraint(optimization_algorithm)


class TrainingAssistant:

    def __init__(self,
                 printing_enabled: bool,
                 print_update_every: int,
                 maximal_number_of_iterations: int,
                 update_stepsize_every: int,
                 factor_update_stepsize: float,
                 bins: list = None):
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

    def get_progressbar(self):
        pbar = tqdm(range(self.maximal_number_of_iterations))
        pbar.set_description('Fit algorithm')
        return pbar

    def should_update_stepsize_of_optimizer(self, iteration: int) -> bool:
        if (iteration >= 1) and (iteration % self.update_stepsize_every == 0):
            return True
        else:
            return False

    def update_stepsize_of_optimizer(self, optimizer: torch.optim.Optimizer):
        for g in optimizer.param_groups:
            g['lr'] = self.factor_update_stepsize * g['lr']

    def get_bins(self) -> list:
        return self.bins

    def set_bins(self, new_bins: list):
        self.bins = new_bins

    def should_print_update(self, iteration: int) -> bool:
        if iteration >= 1 and self.printing_enabled and iteration % self.print_update_every == 0:
            return True
        else:
            return False

    def print_update(self, iteration: int, constraint_checker: ConstraintChecker):
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


class TrajectoryRandomizer:

    def __init__(self,
                 should_restart: bool,
                 restart_probability: float,
                 length_partial_trajectory: int):
        self.should_restart = should_restart
        self.restart_probability = restart_probability
        self.length_partial_trajectory = length_partial_trajectory

    def get_should_restart(self):
        return self.should_restart

    def set_should_restart(self, should_restart: bool):
        if not isinstance(should_restart, bool):
            raise TypeError("Type of 'should_restart' has to be bool.")
        self.should_restart = should_restart

    def get_restart_probability(self):
        return self.restart_probability


class InitializationAssistant:

    def __init__(self, printing_enabled, maximal_number_of_iterations, update_stepsize_every, factor_update_stepsize,
                 print_update_every):
        self.printing_enabled = printing_enabled
        self.maximal_number_of_iterations = maximal_number_of_iterations
        self.update_stepsize_every = update_stepsize_every
        self.print_update_every = print_update_every
        self.factor_update_stepsize = factor_update_stepsize
        self.running_loss = 0

    def starting_message(self):
        if self.printing_enabled:
            print("Init. network to mimic algorithm.")
            print(f"Optimizing for {self.maximal_number_of_iterations} iterations.")

    def final_message(self):
        if self.printing_enabled:
            print("Finished initialization.")

    def get_progressbar(self):
        pbar = tqdm(range(self.maximal_number_of_iterations))
        pbar.set_description('Initialize algorithm')
        return pbar

    def should_update_stepsize_of_optimizer(self, iteration):
        return (iteration >= 1) and (iteration % self.update_stepsize_every == 0)

    def update_stepsize_of_optimizer(self, optimizer):
        for g in optimizer.param_groups:
            g['lr'] = self.factor_update_stepsize * g['lr']

    def should_print_update(self, iteration):
        return (iteration >= 1) and self.printing_enabled and (iteration % self.print_update_every == 0)

    def print_update(self, iteration):
        print(f"Iteration: {iteration}")
        print("\tAvg. Loss = {:.2f}".format(self.running_loss / self.print_update_every))
        self.running_loss = 0


class ParametricOptimizationAlgorithm(OptimizationAlgorithm):

    def __init__(self,
                 initial_state: torch.Tensor,
                 implementation: nn.Module,
                 loss_function: LossFunction,
                 constraint: Constraint = None):

        super().__init__(initial_state=initial_state, implementation=implementation, loss_function=loss_function,
                         constraint=constraint)

    def initialize_with_other_algorithm(self,
                                        other_algorithm: OptimizationAlgorithm,
                                        loss_functions: List[LossFunction],
                                        parameters: dict) -> None:

        optimizer, initialization_assistant, trajectory_randomizer = self.initialize_helpers_for_initialization(
            parameters=parameters)
        initialization_assistant.starting_message()
        pbar = initialization_assistant.get_progressbar()
        for i in pbar:

            self.update_initialization_of_hyperparameters(optimizer=optimizer,
                                                          other_algorithm=other_algorithm,
                                                          trajectory_randomizer=trajectory_randomizer,
                                                          loss_functions=loss_functions,
                                                          initialization_assistant=initialization_assistant)

            if initialization_assistant.should_print_update(iteration=i):
                initialization_assistant.print_update(iteration=i)

            if initialization_assistant.should_update_stepsize_of_optimizer(iteration=i):
                initialization_assistant.update_stepsize_of_optimizer(optimizer=optimizer)

        self.reset_state_and_iteration_counter()
        other_algorithm.reset_state_and_iteration_counter()
        initialization_assistant.final_message()

    def initialize_helpers_for_initialization(self, parameters):

        initialization_assistant = InitializationAssistant(
            printing_enabled=parameters['with_print'],
            maximal_number_of_iterations=parameters['num_iter_max'],
            update_stepsize_every=parameters['num_iter_update_stepsize'],
            print_update_every=parameters['num_iter_print_update'],
            factor_update_stepsize=0.5
        )

        trajectory_randomizer = TrajectoryRandomizer(
            should_restart=True,
            restart_probability=0.05,
            length_partial_trajectory=5
        )
        optimizer = torch.optim.Adam(self.implementation.parameters(), lr=parameters['lr'])

        return optimizer, initialization_assistant, trajectory_randomizer

    def update_initialization_of_hyperparameters(
            self,
            optimizer,
            other_algorithm,
            trajectory_randomizer,
            loss_functions,
            initialization_assistant):

        optimizer.zero_grad()
        self.determine_next_starting_point_for_both_algorithms(
            trajectory_randomizer=trajectory_randomizer,
            other_algorithm=other_algorithm,
            loss_functions=loss_functions)
        iterates_other = other_algorithm.compute_partial_trajectory(
            number_of_steps=trajectory_randomizer.length_partial_trajectory)
        iterates_self = self.compute_partial_trajectory(
            number_of_steps=trajectory_randomizer.length_partial_trajectory)
        loss = compute_initialization_loss(iterates_learned_algorithm=iterates_self,
                                           iterates_standard_algorithm=iterates_other)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            initialization_assistant.running_loss += loss

    def determine_next_starting_point_for_both_algorithms(self,
                                                          trajectory_randomizer: TrajectoryRandomizer,
                                                          other_algorithm: OptimizationAlgorithm,
                                                          loss_functions: List[LossFunction]):
        if trajectory_randomizer.should_restart:
            self.restart_with_new_loss(loss_functions=loss_functions)
            other_algorithm.reset_state_and_iteration_counter()
            other_algorithm.loss_function = self.loss_function
            trajectory_randomizer.set_should_restart(False)
        else:
            self.detach_current_state_from_computational_graph()
            trajectory_randomizer.set_should_restart(
                (torch.rand(1) <= trajectory_randomizer.restart_probability).item()
            )

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
        pbar = training_assistant.get_progressbar()
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

    def initialize_helpers_for_training(
            self,
            fitting_parameters: dict,
            constraint_parameters: dict,
            update_parameters: dict
    ) -> Tuple[torch.optim.Optimizer, TrainingAssistant, TrajectoryRandomizer, ConstraintChecker]:

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

    def update_hyperparameters(self,
                               optimizer: torch.optim.Optimizer,
                               trajectory_randomizer: TrajectoryRandomizer,
                               loss_functions: List[LossFunction],
                               training_assistant: TrainingAssistant):

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
            training_assistant.running_loss += sum_losses.item()

    def determine_next_starting_point(self,
                                      trajectory_randomizer: TrajectoryRandomizer,
                                      loss_functions: List[LossFunction]):
        if trajectory_randomizer.should_restart:
            self.restart_with_new_loss(loss_functions=loss_functions)
            trajectory_randomizer.set_should_restart(False)
        else:
            self.detach_current_state_from_computational_graph()
            trajectory_randomizer.set_should_restart(
                (torch.rand(1) <= trajectory_randomizer.restart_probability).item()
            )

    def restart_with_new_loss(self, loss_functions: List[LossFunction]):
        self.reset_state_and_iteration_counter()
        self.set_loss_function(np.random.choice(loss_functions))

    def detach_current_state_from_computational_graph(self):
        x_0 = self.current_state.detach().clone()
        self.set_current_state(x_0)

    def compute_ratio_of_losses(self, predicted_iterates: List[torch.Tensor]) -> List:
        ratios = [self.loss_function(predicted_iterates[k]) / self.loss_function(predicted_iterates[k - 1])
                  for k in range(1, len(predicted_iterates))]
        return ratios

    def sample_with_sgld(self,
                         loss_functions: list,
                         parameters: dict
                         ) -> Tuple[list, list, list]:

        sampling_assistant = SamplingAssistant(
            learning_rate=parameters['lr'],
            desired_number_of_samples=parameters['num_samples'],
            number_of_iterations_burnin=parameters['num_iter_burnin']
        )

        trajectory_randomizer = TrajectoryRandomizer(
            should_restart=True,
            restart_probability=parameters['restart_probability'],
            length_partial_trajectory=parameters['length_trajectory']
        )

        # Setup distributions for noise
        noise_distributions = self.set_up_noise_distributions()

        # For rejection procedure
        # This assumes that the initialization of the sampling algorithm lies withing the constraint!
        # Since 'fit' should be called before this method, the final output either got rejected or does indeed ly
        # inside the constraint.
        old_state_dict = copy.deepcopy(self.implementation.state_dict())
        t = 1
        pbar = sampling_assistant.get_progressbar()
        while sampling_assistant.should_continue():

            sampling_assistant.decay_learning_rate(iteration=t)

            # Note that this initialization refers to the optimization space: This is different from the
            # hyperparameter-space, which is the one for sampling!
            # Further: This restarting procedure is only a heuristic from our training-procedure.
            # It is not inherent in SGLD!
            self.determine_next_starting_point(
                trajectory_randomizer=trajectory_randomizer, loss_functions=loss_functions)
            self.set_loss_function(np.random.choice(loss_functions))    # For SGLD, we always sample a new loss-function
            predicted_iterates = self.compute_partial_trajectory(
                number_of_steps=trajectory_randomizer.length_partial_trajectory)
            ratios_of_losses = self.compute_ratio_of_losses(predicted_iterates=predicted_iterates)
            if losses_are_invalid(ratios_of_losses):
                print('Invalid losses.')
                trajectory_randomizer.set_should_restart(True)
                add_noise(self, lr=sampling_assistant.current_learning_rate, noise_distributions=noise_distributions)
                continue
            sum_losses = torch.sum(torch.stack(ratios_of_losses))
            sum_losses.backward()
            self.perform_noisy_gradient_step_on_hyperparameters(lr=sampling_assistant.current_learning_rate,
                                                                noise_distributions=noise_distributions)

            # Check constraint. Reject current step if it is not satisfied.
            # Note that for the sampling procedure here, it is assumed that one starts with a point inside the
            # constraint, i.e. one already has found a point inside.
            if self.constraint is not None:
                satisfies_constraint, estimated_prob = self.constraint(self, return_val=True)

                # If constraint is satisfied, update the point.
                if satisfies_constraint:
                    old_state_dict = copy.deepcopy(self.implementation.state_dict())

                    if sampling_assistant.should_store_sample(iteration=t):
                        sampling_assistant.store_sample(implementation=self.implementation,
                                                        estimated_probability=estimated_prob)
                        pbar.update(1)

                # Otherwise, reject the new point.
                else:
                    self.implementation.load_state_dict(old_state_dict)

            # Update iteration counter (for reduction of step-size, to prevent getting stuck)
            t += 1

        self.reset_state_and_iteration_counter()
        return sampling_assistant.prepare_output()

    def set_up_noise_distributions(self):
        noise_distributions = {}
        for name, parameter in self.implementation.named_parameters():
            if parameter.requires_grad:
                dim = len(parameter.flatten())
                noise_distributions[name] = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
        return noise_distributions

    def perform_noisy_gradient_step_on_hyperparameters(self, lr, noise_distributions):
        for p in self.implementation.parameters():
            if p.requires_grad:
                noise = lr ** 2 * noise_distributions[p].sample()
                with torch.no_grad():
                    p.add_(-0.5 * lr * p.grad + noise.reshape(p.shape))


def add_noise(opt_algo: OptimizationAlgorithm,
              lr: float,
              noise_distributions: dict
              ) -> None:
    # Add noise to every hyperparameter that requires gradient
    with torch.no_grad():
        for p in opt_algo.implementation.parameters():
            if p.requires_grad:
                noise = lr ** 2 * noise_distributions[p].sample()
                p.add_(noise.reshape(p.shape))


def losses_are_invalid(losses: List) -> bool:
    if (len(losses) == 0) or (None in losses) or (torch.inf in losses):
        return True
    return False


def compute_initialization_loss(iterates_learned_algorithm, iterates_standard_algorithm):
    if len(iterates_standard_algorithm) != len(iterates_learned_algorithm):
        raise ValueError("Number of iterates does not match.")
    criterion = torch.nn.MSELoss()
    return torch.sum(torch.stack(
        [criterion(prediction, x) for prediction, x in zip(iterates_learned_algorithm[1:],
                                                           iterates_standard_algorithm[1:])])
    )
