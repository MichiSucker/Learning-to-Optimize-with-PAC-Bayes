import copy
import unittest
from types import NoneType

import torch
from classes.LossFunction.class_LossFunction import LossFunction
from classes.OptimizationAlgorithm.derived_classes.subclass_ParametricOptimizationAlgorithm import (
    ParametricOptimizationAlgorithm)
from algorithms.dummy import Dummy
from classes.Helpers.class_SamplingAssistant import SamplingAssistant


class TestSamplingAssistant(unittest.TestCase):

    def setUp(self):

        def dummy_function(x):
            return 0.5 * torch.linalg.norm(x) ** 2

        self.dummy_function = dummy_function
        self.learning_rate = 1
        self.number_of_iterations_burnin = 100
        self.desired_number_of_samples = 10
        self.sampling_assistant = SamplingAssistant(learning_rate=self.learning_rate,
                                                    desired_number_of_samples=self.desired_number_of_samples,
                                                    number_of_iterations_burnin=self.number_of_iterations_burnin)

    def test_creation(self):
        self.assertIsInstance(self.sampling_assistant, SamplingAssistant)

    def test_decay_learning_rate(self):
        iteration = 10
        self.assertEqual(self.sampling_assistant.current_learning_rate, self.sampling_assistant.initial_learning_rate)
        self.sampling_assistant.decay_learning_rate(iteration=iteration)
        self.assertEqual(self.sampling_assistant.current_learning_rate,
                         self.sampling_assistant.initial_learning_rate / iteration)

    def test_set_point_that_satisfies_constraint(self):
        self.assertIsInstance(self.sampling_assistant.point_that_satisfies_constraint, NoneType)
        implementation = Dummy()
        self.sampling_assistant.set_point_that_satisfies_constraint(implementation.state_dict())
        self.assertEqual(self.sampling_assistant.point_that_satisfies_constraint, implementation.state_dict())

        # Also check that one cannot just set anything.
        with self.assertRaises(TypeError):
            self.sampling_assistant.set_point_that_satisfies_constraint(1.)

    def test_set_noise_distributions(self):
        self.assertIsInstance(self.sampling_assistant.noise_distributions, NoneType)
        noise_distributions = {'scale': 0.1}
        self.sampling_assistant.set_noise_distributions(noise_distributions)
        self.assertEqual(self.sampling_assistant.noise_distributions, noise_distributions)

        # Also check that only dictionaries can be set.
        with self.assertRaises(TypeError):
            self.sampling_assistant.set_noise_distributions(1)

    def test_should_continue(self):
        self.assertIsInstance(self.sampling_assistant.should_continue(), bool)

        # Check scenario where the algorithm should continue.
        self.sampling_assistant.number_of_correct_samples = 1
        self.sampling_assistant.desired_number_of_samples = 1
        self.sampling_assistant.number_of_iterations_burnin = 1
        self.assertTrue(self.sampling_assistant.should_continue())

        # Check scenario where the algorithm should stop.
        self.sampling_assistant.number_of_correct_samples = 1
        self.sampling_assistant.desired_number_of_samples = 1
        self.sampling_assistant.number_of_iterations_burnin = 0
        self.assertFalse(self.sampling_assistant.should_continue())

    def test_should_store_sample(self):
        # Samples should be stored after burn-in phase.
        self.assertIsInstance(self.sampling_assistant.should_store_sample(iteration=10), bool)
        self.assertTrue(self.sampling_assistant.should_store_sample(iteration=self.number_of_iterations_burnin + 1))
        self.assertFalse(self.sampling_assistant.should_store_sample(iteration=self.number_of_iterations_burnin - 1))

    def test_reject_sample(self):
        # Initialize setting.
        dim = torch.randint(low=1, high=1000, size=(1,)).item()
        length_state = 1
        initial_state = torch.randn(size=(length_state, dim))
        loss_function = LossFunction(function=self.dummy_function)
        optimization_algorithm = ParametricOptimizationAlgorithm(implementation=Dummy(),
                                                                 initial_state=initial_state,
                                                                 loss_function=loss_function)
        self.sampling_assistant.set_point_that_satisfies_constraint(optimization_algorithm.implementation.state_dict())
        old_point = copy.deepcopy(self.sampling_assistant.point_that_satisfies_constraint)
        optimization_algorithm.implementation.state_dict()['scale'] -= 0.1

        # Check that point got rejected.
        self.assertNotEqual(optimization_algorithm.implementation.state_dict(), old_point)
        self.sampling_assistant.reject_sample(optimization_algorithm)
        self.assertEqual(optimization_algorithm.implementation.state_dict(), old_point)

    def test_store_sample(self):
        # Initialize setting.
        old_number_of_samples = self.sampling_assistant.number_of_correct_samples
        old_length_samples = len(self.sampling_assistant.samples)
        old_length_samples_state_dict = len(self.sampling_assistant.samples_state_dict)
        old_length_estimated_probabilities = len(self.sampling_assistant.estimated_probabilities)
        implementation = Dummy()
        estimated_probability = 0.9

        # Check that storing the sample updates all three lists related to that.
        self.sampling_assistant.store_sample(implementation=implementation, estimated_probability=estimated_probability)
        self.assertEqual(self.sampling_assistant.number_of_correct_samples, old_number_of_samples + 1)
        self.assertEqual(len(self.sampling_assistant.samples), old_length_samples + 1)
        self.assertEqual(len(self.sampling_assistant.samples_state_dict), old_length_samples_state_dict + 1)
        self.assertEqual(len(self.sampling_assistant.estimated_probabilities), old_length_estimated_probabilities + 1)
        self.assertTrue(self.sampling_assistant.estimated_probabilities[-1] == estimated_probability)
        self.assertTrue(self.sampling_assistant.samples_state_dict[-1] == implementation.state_dict())
        self.assertTrue(self.sampling_assistant.samples[-1] == [p.detach().clone() for p in implementation.parameters()
                                                                if p.requires_grad])

    def test_prepare_output(self):
        # Raise exception if there is no sample, or if the something went wrong during updating
        # (length of lists do not match).
        with self.assertRaises(Exception):
            self.sampling_assistant.prepare_output()
        self.sampling_assistant.desired_number_of_samples = 1
        self.sampling_assistant.samples.append(1)
        with self.assertRaises(Exception):
            self.sampling_assistant.prepare_output()
        self.sampling_assistant.samples_state_dict.append(1)
        with self.assertRaises(Exception):
            self.sampling_assistant.prepare_output()
        self.sampling_assistant.estimated_probabilities.append(1)

        # Check correct preparation of output.
        samples, state_dict_samples, estimated_probabilities = self.sampling_assistant.prepare_output()
        self.assertIsInstance(samples, list)
        self.assertIsInstance(state_dict_samples, list)
        self.assertIsInstance(estimated_probabilities, list)
        self.assertEqual(len(samples), len(state_dict_samples))
        self.assertEqual(len(state_dict_samples), len(estimated_probabilities))
