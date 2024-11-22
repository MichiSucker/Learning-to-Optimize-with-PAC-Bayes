import torch
import torch.nn as nn
from typing import Tuple, List, Dict
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.Constraint.class_Constraint import Constraint
from classes.LossFunction.class_LossFunction import LossFunction
import numpy as np
from torch.distributions import MultivariateNormal
from classes.Helpers.class_InitializationAssistant import InitializationAssistant
from classes.Helpers.class_SamplingAssistant import SamplingAssistant
from classes.Helpers.class_ConstraintChecker import ConstraintChecker
from classes.Helpers.class_TrainingAssistant import TrainingAssistant
from classes.Helpers.class_TrajectoryRandomizer import TrajectoryRandomizer


class ParametricOptimizationAlgorithm(OptimizationAlgorithm):

    def __init__(self,
                 initial_state: torch.Tensor,
                 implementation: nn.Module,
                 loss_function: LossFunction,
                 constraint: Constraint = None):

        super().__init__(initial_state=initial_state, implementation=implementation, loss_function=loss_function,
                         constraint=constraint)

    def set_hyperparameters_to(self, new_hyperparameters: dict) -> None:
        self.implementation.load_state_dict(new_hyperparameters)

    def initialize_with_other_algorithm(self,
                                        other_algorithm: OptimizationAlgorithm,
                                        loss_functions: List[LossFunction],
                                        parameters_of_initialization: dict) -> None:

        optimizer, initialization_assistant, trajectory_randomizer = self.initialize_helpers_for_initialization(
            parameters=parameters_of_initialization)
        initialization_assistant.print_starting_message()
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
        initialization_assistant.print_final_message()

    def initialize_helpers_for_initialization(
            self, parameters: dict) -> Tuple[torch.optim.Optimizer, InitializationAssistant, TrajectoryRandomizer]:

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
            optimizer: torch.optim.Optimizer,
            other_algorithm: OptimizationAlgorithm,
            trajectory_randomizer: TrajectoryRandomizer,
            loss_functions: List[LossFunction],
            initialization_assistant: InitializationAssistant) -> None:

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
                                                          loss_functions: List[LossFunction]) -> None:
        if trajectory_randomizer.should_restart:
            self.restart_with_new_loss(loss_functions=loss_functions)
            other_algorithm.reset_state_and_iteration_counter()
            other_algorithm.loss_function = self.loss_function
            trajectory_randomizer.set_variable__should_restart__to(False)
        else:
            self.detach_current_state_from_computational_graph()
            trajectory_randomizer.set_variable__should_restart__to(
                (torch.rand(1) <= trajectory_randomizer.restart_probability).item())

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

        training_assistant.print_starting_message()
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
        training_assistant.print_final_message()

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
                               training_assistant: TrainingAssistant) -> None:

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
                                      loss_functions: List[LossFunction]) -> None:
        if trajectory_randomizer.should_restart:
            self.restart_with_new_loss(loss_functions=loss_functions)
            trajectory_randomizer.set_variable__should_restart__to(False)
        else:
            self.detach_current_state_from_computational_graph()
            trajectory_randomizer.set_variable__should_restart__to(
                (torch.rand(1) <= trajectory_randomizer.restart_probability).item())

    def restart_with_new_loss(self, loss_functions: List[LossFunction]) -> None:
        self.reset_state_and_iteration_counter()
        self.set_loss_function(np.random.choice(loss_functions))

    def detach_current_state_from_computational_graph(self) -> None:
        x_0 = self.current_state.detach().clone()
        self.set_current_state(x_0)

    def compute_ratio_of_losses(self, predicted_iterates: List[torch.Tensor]) -> List[torch.Tensor]:
        ratios = [self.loss_function(predicted_iterates[k]) / self.loss_function(predicted_iterates[k - 1])
                  if self.loss_function(predicted_iterates[k - 1]) > 1e-12
                  else self.loss_function(predicted_iterates[k]) - self.loss_function(predicted_iterates[k])
                  for k in range(1, len(predicted_iterates))]
        return ratios

    def fit_with_function_values(self,
                                 loss_functions: list,
                                 fitting_parameters: dict,
                                 constraint_parameters: dict,
                                 update_parameters: dict) -> None:

        # (!) Note: This function is solely used for the additional_experiments to show that it is not good to train
        # on function values (!)

        optimizer, training_assistant, trajectory_randomizer, constraint_checker = self.initialize_helpers_for_training(
            fitting_parameters=fitting_parameters,
            constraint_parameters=constraint_parameters,
            update_parameters=update_parameters
        )

        training_assistant.print_starting_message()
        pbar = training_assistant.get_progressbar()
        for i in pbar:
            if training_assistant.should_update_stepsize_of_optimizer(iteration=i):
                training_assistant.update_stepsize_of_optimizer(optimizer)

            self.update_hyperparameters_based_on_function_values(optimizer=optimizer,
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
        training_assistant.print_final_message()

    def update_hyperparameters_based_on_function_values(self,
                                                        optimizer: torch.optim.Optimizer,
                                                        trajectory_randomizer: TrajectoryRandomizer,
                                                        loss_functions: List[LossFunction],
                                                        training_assistant: TrainingAssistant) -> None:

        # (!) Note: This function is solely used for the additional_experiments to show that it is not good to train
        # on function values (!)

        optimizer.zero_grad()
        self.determine_next_starting_point(
            trajectory_randomizer=trajectory_randomizer, loss_functions=loss_functions)
        predicted_iterates = self.compute_partial_trajectory(
            number_of_steps=trajectory_randomizer.length_partial_trajectory)
        losses = [self.loss_function(predicted_iterates[k]) for k in range(1, len(predicted_iterates))]
        if losses_are_invalid(losses):
            print('Invalid losses.')
            return
        sum_losses = torch.sum(torch.stack(losses))
        sum_losses.backward()
        optimizer.step()

        with torch.no_grad():
            training_assistant.loss_histogram.append(self.loss_function(predicted_iterates[-1]).item())
            training_assistant.running_loss += sum_losses.item()

    def sample_with_sgld(self,
                         loss_functions: List[LossFunction],
                         parameters: dict
                         ) -> Tuple[list, list, list]:

        sampling_assistant, trajectory_randomizer = self.initialize_helpers_for_sampling(parameters=parameters)
        t = 1
        while sampling_assistant.should_continue():

            sampling_assistant.decay_learning_rate(iteration=t)
            self.compute_next_possible_sample(
                loss_functions=loss_functions,
                trajectory_randomizer=trajectory_randomizer,
                sampling_assistant=sampling_assistant
            )

            if self.constraint is not None:
                self.accept_or_reject_based_on_constraint(sampling_assistant=sampling_assistant, iteration=t)
            else:
                self.update_point(sampling_assistant=sampling_assistant, iteration=t, estimated_probability=1.)

            t += 1

        self.reset_state_and_iteration_counter()
        return sampling_assistant.prepare_output()

    def initialize_helpers_for_sampling(self, parameters: dict) -> Tuple[SamplingAssistant, TrajectoryRandomizer]:

        trajectory_randomizer = TrajectoryRandomizer(
            should_restart=True,
            restart_probability=parameters['restart_probability'],
            length_partial_trajectory=parameters['length_trajectory']
        )

        sampling_assistant = SamplingAssistant(
            learning_rate=parameters['lr'],
            desired_number_of_samples=parameters['num_samples'],
            number_of_iterations_burnin=parameters['num_iter_burnin']
        )
        sampling_assistant.set_noise_distributions(self.set_up_noise_distributions())
        # For rejection procedure
        # This assumes that the initialization of the sampling algorithm lies withing the constraint!
        # Since 'fit' should be called before this method, the final output either got rejected or does indeed ly
        # inside the constraint.
        sampling_assistant.set_point_that_satisfies_constraint(state_dict=self.implementation.state_dict())
        return sampling_assistant, trajectory_randomizer

    def compute_next_possible_sample(self,
                                     loss_functions: List[LossFunction],
                                     trajectory_randomizer: TrajectoryRandomizer,
                                     sampling_assistant: SamplingAssistant) -> None:
        # Note that this initialization refers to the optimization space: This is different from the
        # hyperparameter-space, which is the one for sampling!
        # Further: This restarting procedure is only a heuristic from our training-procedure.
        # It is not inherent in SGLD!
        self.determine_next_starting_point(
            trajectory_randomizer=trajectory_randomizer, loss_functions=loss_functions)
        self.set_loss_function(np.random.choice(loss_functions))  # For SGLD, we always sample a new loss-function
        predicted_iterates = self.compute_partial_trajectory(
            number_of_steps=trajectory_randomizer.length_partial_trajectory)
        ratios_of_losses = self.compute_ratio_of_losses(predicted_iterates=predicted_iterates)
        if losses_are_invalid(ratios_of_losses):
            print('Invalid losses.')
            trajectory_randomizer.set_variable__should_restart__to(True)
            add_noise_to_every_parameter_that_requires_grad(self, sampling_assistant=sampling_assistant)
            return
        sum_losses = torch.sum(torch.stack(ratios_of_losses))
        sum_losses.backward()
        self.perform_noisy_gradient_step_on_hyperparameters(sampling_assistant)

    def accept_or_reject_based_on_constraint(self, sampling_assistant: SamplingAssistant, iteration: int) -> None:
        # TODO: For more readability, adjust that to probabilistic constraint.
        #  Like this, its not obvious from the code that self.constraint has to be create from a ProbabilisticConstraint
        satisfies_constraint, estimated_prob = self.constraint(self, also_return_value=True)
        if satisfies_constraint:
            self.update_point(sampling_assistant=sampling_assistant, iteration=iteration,
                              estimated_probability=estimated_prob)
        else:
            sampling_assistant.reject_sample(self)

    def update_point(self, sampling_assistant: SamplingAssistant, iteration: int, estimated_probability: float):
        sampling_assistant.set_point_that_satisfies_constraint(state_dict=self.implementation.state_dict())
        if sampling_assistant.should_store_sample(iteration=iteration):
            sampling_assistant.store_sample(implementation=self.implementation,
                                            estimated_probability=estimated_probability)

    def set_up_noise_distributions(self) -> Dict:
        noise_distributions = {}
        for name, parameter in self.implementation.named_parameters():
            if parameter.requires_grad:
                dim = len(parameter.flatten())
                noise_distributions[name] = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
        return noise_distributions

    def perform_noisy_gradient_step_on_hyperparameters(self, sampling_assistant: SamplingAssistant) -> None:
        for name, parameter in self.implementation.named_parameters():
            if parameter.requires_grad:
                noise = (sampling_assistant.current_learning_rate ** 0.5
                         * sampling_assistant.noise_distributions[name].sample())
                with torch.no_grad():
                    parameter.add_(
                        -0.5 * sampling_assistant.current_learning_rate * parameter.grad
                        + noise.reshape(parameter.shape)
                    )


def add_noise_to_every_parameter_that_requires_grad(
        opt_algo: OptimizationAlgorithm, sampling_assistant: SamplingAssistant) -> None:
    with torch.no_grad():
        for name, parameter in opt_algo.implementation.named_parameters():
            if parameter.requires_grad:
                noise = (sampling_assistant.current_learning_rate ** 0.5
                         * sampling_assistant.noise_distributions[name].sample())
                parameter.add_(noise.reshape(parameter.shape))


def losses_are_invalid(losses: List) -> bool:
    if (len(losses) == 0) or (None in losses) or (torch.inf in losses):
        return True
    return False


def compute_initialization_loss(iterates_learned_algorithm: List[torch.Tensor],
                                iterates_standard_algorithm: List[torch.Tensor]) -> torch.Tensor:
    if len(iterates_standard_algorithm) != len(iterates_learned_algorithm):
        raise ValueError("Number of iterates does not match.")
    criterion = torch.nn.MSELoss()
    return torch.sum(torch.stack(
        [criterion(prediction, x) for prediction, x in zip(iterates_learned_algorithm[1:],
                                                           iterates_standard_algorithm[1:])])
    )
