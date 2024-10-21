import unittest

import algorithms.dummy
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.LossFunction.class_LossFunction import LossFunction
from algorithms.dummy import Dummy
import torch


def dummy_function(x):
    return 0.5 * torch.linalg.norm(x) ** 2


class TestClassOptimizationAlgorithm(unittest.TestCase):

    def setUp(self):
        self.dim = torch.randint(low=1, high=1000, size=(1,)).item()
        self.length_state = torch.randint(low=1, high=5, size=(1,)).item()
        self.initial_state = torch.randn(size=(self.length_state, self.dim))
        self.current_state = self.initial_state.clone()
        self.loss_function = LossFunction(function=dummy_function)
        self.optimization_algorithm = OptimizationAlgorithm(implementation=Dummy(),
                                                            initial_state=self.initial_state,
                                                            loss_function=self.loss_function)

    def test_creation(self):
        self.assertIsInstance(self.optimization_algorithm, OptimizationAlgorithm)

    def test_attributes(self):
        self.assertTrue(hasattr(self.optimization_algorithm, 'initial_state'))
        self.assertTrue(hasattr(self.optimization_algorithm, 'current_state'))
        self.assertTrue(hasattr(self.optimization_algorithm, 'current_iterate'))
        self.assertTrue(hasattr(self.optimization_algorithm, 'implementation'))
        self.assertTrue(hasattr(self.optimization_algorithm, 'iteration_counter'))
        self.assertTrue(hasattr(self.optimization_algorithm, 'loss_function'))

    def test_get_initial_state(self):
        self.assertIsInstance(self.optimization_algorithm.get_initial_state(), torch.Tensor)
        self.assertTrue(torch.equal(self.optimization_algorithm.get_initial_state(), self.initial_state))

    def test_get_implementation(self):
        self.assertIsInstance(self.optimization_algorithm.get_implementation(), algorithms.dummy.Dummy)

    def test_get_current_state(self):
        self.assertIsInstance(self.optimization_algorithm.get_current_state(), torch.Tensor)
        self.assertTrue(torch.equal(self.optimization_algorithm.get_current_state(), self.current_state))

    def test_get_current_iterate(self):
        self.assertIsInstance(self.optimization_algorithm.get_current_iterate(), torch.Tensor)
        self.assertTrue(torch.equal(self.optimization_algorithm.get_current_iterate(),
                                    self.optimization_algorithm.get_current_state()[-1]))

    def test_get_iteration_counter(self):
        self.assertIsInstance(self.optimization_algorithm.get_iteration_counter(), int)

    def test_set_iteration_counter(self):
        random_number = torch.randint(low=0, high=100, size=(1,)).item()
        self.optimization_algorithm.set_iteration_counter(n=random_number)
        self.assertEqual(self.optimization_algorithm.get_iteration_counter(), random_number)

    def test_reset_iteration_counter_to_zero(self):
        self.optimization_algorithm.set_iteration_counter(10)
        self.assertNotEqual(self.optimization_algorithm.get_iteration_counter(), 0)
        self.optimization_algorithm.reset_iteration_counter_to_zero()
        self.assertEqual(self.optimization_algorithm.get_iteration_counter(), 0)

    def test_reset_to_initial_state(self):
        self.optimization_algorithm.reset_to_initial_state()
        self.assertTrue(torch.equal(self.optimization_algorithm.current_state,
                                    self.optimization_algorithm.initial_state))
        self.assertTrue(torch.equal(self.optimization_algorithm.initial_state, self.initial_state))

    def test_set_current_state(self):
        old_state = self.optimization_algorithm.get_current_state()
        new_state = torch.randn(size=(self.length_state, self.dim))
        self.optimization_algorithm.set_current_state(new_state)
        self.assertFalse(torch.equal(old_state, self.optimization_algorithm.get_current_state()))
        self.assertTrue(torch.equal(new_state, self.optimization_algorithm.get_current_state()))
        self.assertTrue(torch.equal(new_state[-1], self.optimization_algorithm.get_current_iterate()))

    def test_update_state(self):
        self.assertTrue(hasattr(self.optimization_algorithm.implementation, 'forward'))
        self.assertTrue(hasattr(self.optimization_algorithm.implementation, 'update_state'))
        current_state = self.optimization_algorithm.get_current_state().clone()
        current_iteration_counter = self.optimization_algorithm.get_iteration_counter()
        self.optimization_algorithm.perform_step()
        self.assertNotEqual(self.optimization_algorithm.iteration_counter, current_iteration_counter)
        self.assertFalse(torch.equal(self.optimization_algorithm.get_current_state(), current_state))

    def test_evaluate_loss_function_at_current_iterate(self):
        self.assertEqual(self.loss_function(self.optimization_algorithm.current_iterate),
                         self.optimization_algorithm.evaluate_loss_function_at_current_iterate())