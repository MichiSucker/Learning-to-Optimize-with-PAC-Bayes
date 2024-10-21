import unittest

import algorithms.dummy
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from algorithms.dummy import Dummy
import torch


class TestClassOptimizationAlgorithm(unittest.TestCase):

    def setUp(self):
        self.dim = torch.randint(low=1, high=1000, size=(1,)).item()
        self.length_state = torch.randint(low=1, high=5, size=(1,)).item()
        self.initial_state = torch.randn(size=(self.length_state, self.dim))
        self.optimization_algorithm = OptimizationAlgorithm(implementation=Dummy(), initial_state=self.initial_state)

    def test_creation(self):
        self.assertIsInstance(self.optimization_algorithm, OptimizationAlgorithm)

    def test_get_initial_state(self):
        self.assertIsInstance(self.optimization_algorithm.get_initial_state(), torch.Tensor)
        self.assertTrue(torch.equal(self.optimization_algorithm.get_initial_state(), self.initial_state))

    def test_get_implementation(self):
        self.assertIsInstance(self.optimization_algorithm.get_implementation(), algorithms.dummy.Dummy)
