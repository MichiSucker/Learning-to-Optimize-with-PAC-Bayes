import unittest
from classes.Constraint.class_Constraint import Constraint
import torch


def dummy_constraint(x):
    if torch.all(x >= 0):
        return True
    else:
        return False


class TestConstraint(unittest.TestCase):

    def setUp(self):
        self.constraint = Constraint(dummy_constraint)

    def test_creation(self):
        self.assertIsInstance(self.constraint, Constraint)

    def test_call_constraint(self):
        self.assertTrue(self.constraint(torch.tensor([1., 0.])))
        self.assertFalse(self.constraint(torch.tensor([-1., 0.])))
        self.assertIsInstance(self.constraint(torch.tensor([1., 0.])), bool)
