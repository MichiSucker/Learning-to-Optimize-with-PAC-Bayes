import unittest
import torch
from classes.LossFunction.class_LossFunction import LossFunction


class TestLossFunction(unittest.TestCase):

    def setUp(self):

        def dummy_function(x):
            return 0.5 * torch.linalg.norm(x) ** 2
        self.dummy_function = dummy_function

        self.loss_function = LossFunction(function=dummy_function)

    def test_creation(self):
        self.assertIsInstance(self.loss_function, LossFunction)

    def test_call_function(self):
        dimension = torch.randint(low=1, high=50, size=(1,)).item()
        random_point = torch.randn(size=(dimension,))
        self.assertTrue(torch.equal(self.loss_function(random_point), self.dummy_function(random_point)))

    def test_compute_gradient(self):
        dimension = torch.randint(low=1, high=50, size=(1,)).item()
        random_point = torch.randn(size=(dimension,))
        self.assertTrue(torch.allclose(self.loss_function.compute_gradient(random_point), random_point))
