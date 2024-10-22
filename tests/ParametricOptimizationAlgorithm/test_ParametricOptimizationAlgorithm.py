import unittest
import torch
from algorithms.dummy import Dummy
from classes.LossFunction.class_LossFunction import LossFunction
from classes.OptimizationAlgorithm.derived_classes.subclass_ParametricOptimizationAlgorithm import (
    ParametricOptimizationAlgorithm, TrajectoryRandomizer, losses_are_invalid)


def dummy_function(x):
    return 0.5 * torch.linalg.norm(x) ** 2


class TestFitOfParametricOptimizationAlgorithm(unittest.TestCase):

    def setUp(self):
        self.dim = torch.randint(low=1, high=1000, size=(1,)).item()
        self.length_state = torch.randint(low=1, high=5, size=(1,)).item()
        self.initial_state = torch.randn(size=(self.length_state, self.dim))
        self.current_state = self.initial_state.clone()
        self.loss_function = LossFunction(function=dummy_function)
        self.optimization_algorithm = ParametricOptimizationAlgorithm(implementation=Dummy(),
                                                                      initial_state=self.initial_state,
                                                                      loss_function=self.loss_function)

    def test_creation(self):
        self.assertIsInstance(self.optimization_algorithm, ParametricOptimizationAlgorithm)

    def test_restart_with_new_loss(self):
        self.optimization_algorithm.set_iteration_counter(10)
        loss_functions = [LossFunction(dummy_function) for i in range(10)]
        old_loss_function = self.optimization_algorithm.loss_function
        self.optimization_algorithm.restart_with_new_loss(loss_functions)
        self.assertEqual(self.optimization_algorithm.iteration_counter, 0)
        self.assertNotEqual(old_loss_function, self.optimization_algorithm.loss_function)
        self.assertTrue(self.optimization_algorithm.loss_function in loss_functions)

    def test_detach_current_state_from_computational_graph(self):
        self.optimization_algorithm.current_state.requires_grad = True
        current_state = self.optimization_algorithm.current_state.clone()
        self.assertTrue(self.optimization_algorithm.current_state.requires_grad)
        self.optimization_algorithm.detach_current_state_from_computational_graph()
        self.assertFalse(self.optimization_algorithm.current_state.requires_grad)
        self.assertTrue(torch.equal(current_state, self.optimization_algorithm.current_state))

    def test_determine_next_starting_point(self):
        restart_probability = 0.65
        trajectory_randomizer = TrajectoryRandomizer(should_restart=True,
                                                     restart_probability=restart_probability)
        self.optimization_algorithm.set_iteration_counter(10)
        loss_functions = [LossFunction(dummy_function) for i in range(10)]
        old_loss_function = self.optimization_algorithm.loss_function
        self.optimization_algorithm.set_current_state(torch.randn(size=self.optimization_algorithm.initial_state.shape))
        self.optimization_algorithm.determine_next_starting_point(trajectory_randomizer, loss_functions=loss_functions)
        self.assertFalse(trajectory_randomizer.should_restart)
        self.assertEqual(self.optimization_algorithm.iteration_counter, 0)
        self.assertTrue(torch.equal(self.optimization_algorithm.current_state,
                                    self.optimization_algorithm.initial_state))
        self.assertNotEqual(old_loss_function, self.optimization_algorithm.loss_function)
        self.assertTrue(self.optimization_algorithm.loss_function in loss_functions)

        trajectory_randomizer.set_should_restart(False)
        current_loss_function = self.optimization_algorithm.loss_function
        current_state = self.optimization_algorithm.current_state.clone()
        self.optimization_algorithm.current_state.requires_grad = True
        self.optimization_algorithm.set_iteration_counter(10)
        self.optimization_algorithm.determine_next_starting_point(trajectory_randomizer, loss_functions=loss_functions)
        self.assertFalse(self.optimization_algorithm.current_state.requires_grad)
        self.assertEqual(self.optimization_algorithm.iteration_counter, 10)
        self.assertTrue(torch.equal(self.optimization_algorithm.current_state, current_state))
        self.assertEqual(current_loss_function, self.optimization_algorithm.loss_function)

    def test_compute_ratio_of_losses(self):
        predicted_iterates = [1., 2., 3., 4., 5.]
        self.optimization_algorithm.set_loss_function(lambda x: x)
        ratio_of_losses = self.optimization_algorithm.compute_ratio_of_losses(predicted_iterates)
        self.assertTrue(len(ratio_of_losses) == len(predicted_iterates) - 1)
        self.assertEqual(ratio_of_losses, [2./1., 3./2., 4./3., 5./4.])

    def test_losses_are_invalid(self):
        self.assertFalse(losses_are_invalid([1., 2., 3.]))
        self.assertTrue(losses_are_invalid([]))
        self.assertTrue(losses_are_invalid([1., None]))
        self.assertTrue(losses_are_invalid([1., torch.inf]))
