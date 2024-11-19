import unittest
import torch
from exponential_family.natural_parameters.natural_parameters import evaluate_natural_parameters_at


class TestNaturalParameters(unittest.TestCase):

    def test_evaluate_natural_parameters_at(self):
        # Here, we always have the same natural parameters.
        x = torch.rand(size=(1,))
        result = evaluate_natural_parameters_at(x)
        self.assertTrue(torch.equal(result, torch.tensor([x, -0.5 * x ** 2])))
        with self.assertRaises(TypeError):
            evaluate_natural_parameters_at(1)
        with self.assertRaises(ValueError):
            evaluate_natural_parameters_at(torch.tensor([1., 2.]))
