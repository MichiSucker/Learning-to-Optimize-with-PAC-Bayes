import unittest
from classes.OptimizationAlgorithm.derived_classes.subclass_ParametricOptimizationAlgorithm import (
    ParametricOptimizationAlgorithm, compute_initialization_loss)
import torch


class TestInitParametricOptimizationAlgorithm(unittest.TestCase):

    def test_compute_initialization_loss(self):
        with self.assertRaises(ValueError):
            iterates_1 = [torch.randn(size=(3,)) for _ in range(3)]
            iterates_2 = [torch.randn(size=(3,)) for _ in range(2)]
            compute_initialization_loss(iterates_1, iterates_2)
        iterates_1 = [torch.randn(size=(3,)) for _ in range(3)]
        iterates_2 = [torch.randn(size=(3,)) for _ in range(3)]
        loss = compute_initialization_loss(iterates_1, iterates_2)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss >= 0)