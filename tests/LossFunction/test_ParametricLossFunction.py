import unittest
from classes.LossFunction.class_LossFunction import LossFunction
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
import torch


class TestParametricLossFunction(unittest.TestCase):

    def setUp(self):

        def dummy_function(x, parameter):
            return 0.5 * parameter['p'] * torch.linalg.norm(x) ** 2

        self.parameter = {'p': 2.}
        self.loss_function = ParametricLossFunction(dummy_function, parameter=self.parameter)

    def test_creation(self):
        self.assertIsInstance(self.loss_function, ParametricLossFunction)
        self.assertIsInstance(self.loss_function, LossFunction)
        self.assertIsInstance(self.loss_function.get_parameter(), dict)

    def test_attributes(self):
        self.assertTrue(hasattr(self.loss_function, 'parameter'))

    def test_get_parameter(self):
        self.assertEqual(self.loss_function.get_parameter(), self.parameter)
        self.assertIsInstance(self.loss_function.get_parameter(), dict)
