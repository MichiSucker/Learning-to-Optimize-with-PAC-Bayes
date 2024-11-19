import unittest
from classes.LossFunction.derived_classes.derived_classes.\
    subclass_NonsmoothParametricLossFunction import NonsmoothParametricLossFunction
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
import torch


class TestParametricLossFunction(unittest.TestCase):

    def setUp(self):

        def dummy_smooth_part(x, parameter):
            return 0.5 * parameter['p'] * torch.linalg.norm(x) ** 2

        def dummy_nonsmooth_part(x, parameter):
            return 0.1 * torch.linalg.norm(x, ord=1)

        def dummy_function(x, parameter):
            return dummy_smooth_part(x, parameter) + dummy_nonsmooth_part(x, parameter)

        self.parameter = {'p': 2.}
        self.loss_function = NonsmoothParametricLossFunction(
            function=dummy_function, nonsmooth_part=dummy_nonsmooth_part, smooth_part=dummy_smooth_part,
            parameter=self.parameter)

    def test_creation(self):
        self.assertIsInstance(self.loss_function, NonsmoothParametricLossFunction)
        self.assertIsInstance(self.loss_function, ParametricLossFunction)

    def test_attributes(self):
        self.assertTrue(hasattr(self.loss_function, 'parameter'))
        self.assertTrue(hasattr(self.loss_function, 'whole_function'))
        self.assertTrue(hasattr(self.loss_function, 'smooth_part'))
        self.assertTrue(hasattr(self.loss_function, 'nonsmooth_part'))

    def test_gradient_computation(self):
        # Compute all three separate 'gradients'.
        # Make sure that they add-up correctly.
        dim = torch.randint(low=1, high=10, size=(1,)).item()
        x = torch.randn((dim,))
        nonsmooth_grad = self.loss_function.compute_gradient_of_nonsmooth_part(x)
        smooth_grad = self.loss_function.compute_gradient_of_smooth_part(x)
        grad = self.loss_function.compute_gradient(x)
        self.assertTrue(torch.allclose(nonsmooth_grad, self.loss_function.nonsmooth_part.compute_gradient(x)))
        self.assertTrue(torch.allclose(smooth_grad, self.loss_function.smooth_part.compute_gradient(x)))
        self.assertTrue(torch.allclose(grad, smooth_grad + nonsmooth_grad))
