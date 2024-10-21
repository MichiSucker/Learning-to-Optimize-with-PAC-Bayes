import unittest
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm


class TestClassOptimizationAlgorithm(unittest.TestCase):

    def setUp(self):
        self.optimization_algorithm = OptimizationAlgorithm()

    def test_creation(self):
        self.assertIsInstance(self.optimization_algorithm, OptimizationAlgorithm)
