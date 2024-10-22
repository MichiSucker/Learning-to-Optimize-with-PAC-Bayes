import unittest
import torch
from algorithms.dummy import Dummy
from classes.LossFunction.class_LossFunction import LossFunction
from classes.OptimizationAlgorithm.derived_classes.subclass_ParametricOptimizationAlgorithm import (
    ParametricOptimizationAlgorithm)


def dummy_function(x):
    return 0.5 * torch.linalg.norm(x) ** 2


class TestParametricOptimizationAlgorithm(unittest.TestCase):

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
        start_again_from_initial_state = True
        self.optimization_algorithm.set_iteration_counter(10)
        loss_functions = [LossFunction(dummy_function) for i in range(10)]
        old_loss_function = self.optimization_algorithm.loss_function
        self.optimization_algorithm.set_current_state(torch.randn(size=self.optimization_algorithm.initial_state.shape))
        restart = self.optimization_algorithm.determine_next_starting_point(start_again_from_initial_state,
                                                                            loss_functions=loss_functions,
                                                                            restart_probability=restart_probability)
        self.assertFalse(restart)
        self.assertEqual(self.optimization_algorithm.iteration_counter, 0)
        self.assertTrue(torch.equal(self.optimization_algorithm.current_state,
                                    self.optimization_algorithm.initial_state))
        self.assertNotEqual(old_loss_function, self.optimization_algorithm.loss_function)
        self.assertTrue(self.optimization_algorithm.loss_function in loss_functions)

        start_again_from_initial_state = False
        current_loss_function = self.optimization_algorithm.loss_function
        current_state = self.optimization_algorithm.current_state.clone()
        self.optimization_algorithm.current_state.requires_grad = True
        self.optimization_algorithm.set_iteration_counter(10)
        _ = self.optimization_algorithm.determine_next_starting_point(start_again_from_initial_state,
                                                                      loss_functions=loss_functions,
                                                                      restart_probability=restart_probability)
        self.assertFalse(self.optimization_algorithm.current_state.requires_grad)
        self.assertEqual(self.optimization_algorithm.iteration_counter, 10)
        self.assertTrue(torch.equal(self.optimization_algorithm.current_state, current_state))
        self.assertEqual(current_loss_function, self.optimization_algorithm.loss_function)
