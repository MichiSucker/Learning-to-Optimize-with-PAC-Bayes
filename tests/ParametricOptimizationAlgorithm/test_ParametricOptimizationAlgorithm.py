import unittest
from classes.OptimizationAlgorithm.derived_classes.subclass_ParametricOptimizationAlgorithm import (
    ParametricOptimizationAlgorithm)
from classes.LossFunction.class_LossFunction import LossFunction
import torch
from algorithms.dummy import Dummy
import copy


class TestParametricOptimizationAlgorithm(unittest.TestCase):

    def setUp(self):
        def dummy_function(x):
            return 0.5 * torch.linalg.norm(x) ** 2

        self.dim = torch.randint(low=1, high=1000, size=(1,)).item()
        self.length_state = 1  # Take one, because it has to be compatible with Dummy()
        self.initial_state = torch.randn(size=(self.length_state, self.dim))
        self.current_state = self.initial_state.clone()
        self.loss_function = LossFunction(function=dummy_function)
        self.optimization_algorithm = ParametricOptimizationAlgorithm(implementation=Dummy(),
                                                                      initial_state=self.initial_state,
                                                                      loss_function=self.loss_function)

    def test_set_hyperparameters_to(self):
        new_hyperparameters = copy.deepcopy(self.optimization_algorithm.implementation.state_dict())
        new_hyperparameters['scale'] -= 0.1
        self.assertNotEqual(self.optimization_algorithm.implementation.state_dict(), new_hyperparameters)
        self.optimization_algorithm.set_hyperparameters_to(new_hyperparameters)
        self.assertEqual(self.optimization_algorithm.implementation.state_dict(), new_hyperparameters)
