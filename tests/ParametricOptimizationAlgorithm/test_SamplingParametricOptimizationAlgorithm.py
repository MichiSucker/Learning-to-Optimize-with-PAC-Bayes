import unittest
from distutils.dep_util import newer

from classes.Constraint.class_Constraint import Constraint
from classes.OptimizationAlgorithm.derived_classes.subclass_ParametricOptimizationAlgorithm import (
    ParametricOptimizationAlgorithm, TrajectoryRandomizer, SamplingAssistant)
import torch
from algorithms.dummy import Dummy, DummyWithMoreTrainableParameters
from classes.LossFunction.class_LossFunction import LossFunction
from torch.distributions import MultivariateNormal
import copy

from tests.OptimizationAlgorithm.test_OptimizationAlgorithm import dummy_constraint


def dummy_function(x):
    return 0.5 * torch.linalg.norm(x) ** 2


class TestSamplingParametricOptimizationAlgorithm(unittest.TestCase):

    def setUp(self):
        self.dim = torch.randint(low=1, high=1000, size=(1,)).item()
        self.length_state = 1  # Take one, because it has to be compatible with Dummy()
        self.initial_state = torch.randn(size=(self.length_state, self.dim))
        self.current_state = self.initial_state.clone()
        self.loss_function = LossFunction(function=dummy_function)
        self.optimization_algorithm = ParametricOptimizationAlgorithm(implementation=Dummy(),
                                                                      initial_state=self.initial_state,
                                                                      loss_function=self.loss_function)

    # def test_create_next_sample(self):
    #     self.optimization_algorithm.create_next_sample()
    #     self.assertTrue()

    def test_perform_noisy_gradient_step_on_hyperparameters(self):
        # This is a weak test: We only check whether the hyperparameters do change or not, depending on the learning
        # rate. For a stronger test, one would have to do a statistical test I guess.
        learning_rate = 1e-4
        number_of_iterations_burnin = 100
        desired_number_of_samples = 10
        sampling_assistant = SamplingAssistant(learning_rate=learning_rate,
                                               desired_number_of_samples=desired_number_of_samples,
                                               number_of_iterations_burnin=number_of_iterations_burnin)

        noise_distributions = {}
        for name, parameter in self.optimization_algorithm.implementation.named_parameters():
            if parameter.requires_grad:
                dim = len(parameter.flatten())
                noise_distributions[name] = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
                parameter.grad = torch.randn(size=parameter.shape)
        sampling_assistant.set_noise_distributions(noise_distributions)
        old_hyperparameters = copy.deepcopy(self.optimization_algorithm.implementation.state_dict())
        self.optimization_algorithm.perform_noisy_gradient_step_on_hyperparameters(sampling_assistant)
        self.assertNotEqual(self.optimization_algorithm.implementation.state_dict(), old_hyperparameters)

        sampling_assistant.current_learning_rate = 0
        old_hyperparameters = copy.deepcopy(self.optimization_algorithm.implementation.state_dict())
        self.optimization_algorithm.perform_noisy_gradient_step_on_hyperparameters(sampling_assistant)
        self.assertEqual(self.optimization_algorithm.implementation.state_dict(), old_hyperparameters)

    def test_initialize_helpers_for_sampling(self):

        parameters = {'restart_probability': 0.9, 'length_trajectory': 10, 'lr': 1e-4, 'num_samples': 100,
                      'num_iter_burnin': 100}

        sampling_assistant, trajectory_randomizer = self.optimization_algorithm.initialize_helpers_for_sampling(
            parameters=parameters
        )
        self.assertIsInstance(sampling_assistant, SamplingAssistant)
        self.assertIsInstance(trajectory_randomizer, TrajectoryRandomizer)
        self.assertIsInstance(sampling_assistant.noise_distributions, dict)
        self.assertIsInstance(sampling_assistant.point_that_satisfies_constraint, dict)

    def test_set_up_noise_distributions(self):
        self.optimization_algorithm.implementation = DummyWithMoreTrainableParameters()
        noise_distributions = self.optimization_algorithm.set_up_noise_distributions()
        self.assertIsInstance(noise_distributions, dict)
        for name, parameter in self.optimization_algorithm.implementation.named_parameters():
            if parameter.requires_grad:
                self.assertTrue(name in list(noise_distributions.keys()))
                self.assertIsInstance(noise_distributions[name], MultivariateNormal)
                self.assertEqual(noise_distributions[name].loc.shape, parameter.reshape((-1,)).shape)
            else:
                self.assertFalse(name in list(noise_distributions.keys()))

    def test_compute_next_possible_sample(self):
        self.optimization_algorithm.implementation = DummyWithMoreTrainableParameters()
        noise_distributions = self.optimization_algorithm.set_up_noise_distributions()
        loss_functions = [LossFunction(dummy_function) for _ in range(10)]
        trajectory_randomizer = TrajectoryRandomizer(should_restart=True, restart_probability=1.,
                                                     length_partial_trajectory=1)
        sampling_assistant = SamplingAssistant(learning_rate=1e-4,
                                               desired_number_of_samples=10,
                                               number_of_iterations_burnin=10)
        sampling_assistant.set_noise_distributions(noise_distributions=noise_distributions)
        for name, parameter in self.optimization_algorithm.implementation.named_parameters():
            if parameter.requires_grad:
                dim = len(parameter.flatten())
                noise_distributions[name] = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
                parameter.grad = torch.randn(size=parameter.shape)

        old_hyperparameters = copy.deepcopy(self.optimization_algorithm.implementation.state_dict())
        self.optimization_algorithm.compute_next_possible_sample(
            loss_functions=loss_functions,
            trajectory_randomizer=trajectory_randomizer,
            sampling_assistant=sampling_assistant
        )
        new_hyperparameters = copy.deepcopy(self.optimization_algorithm.implementation.state_dict())
        self.assertNotEqual(old_hyperparameters, new_hyperparameters)

    def test_accept_or_reject_based_on_constraint(self):
        dummy_constraint_always_false = Constraint(function=lambda x: False)
        sampling_assistant = SamplingAssistant(learning_rate=1e-4,
                                               desired_number_of_samples=10,
                                               number_of_iterations_burnin=10)
        self.optimization_algorithm.set_constraint(dummy_constraint_always_false)
        sampling_assistant.set_point_that_satisfies_constraint(self.optimization_algorithm.implementation.state_dict())
        self.optimization_algorithm.implementation.state_dict()['scale'] -= 0.1
        self.assertNotEqual(sampling_assistant.point_that_satisfies_constraint,
                            self.optimization_algorithm.implementation.state_dict())
        self.optimization_algorithm.accept_or_reject_based_on_constraint(sampling_assistant=sampling_assistant,
                                                                         iteration=10)
        self.assertEqual(sampling_assistant.point_that_satisfies_constraint,
                         self.optimization_algorithm.implementation.state_dict())
