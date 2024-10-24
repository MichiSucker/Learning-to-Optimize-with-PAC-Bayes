import unittest
from classes.Constraint.class_ProbabilisticConstraint import ProbabilisticConstraint
from classes.Constraint.class_Constraint import Constraint
from classes.Constraint.class_BayesianProbabilityEstimator import BayesianProbabilityEstimator
import torch


class TestProbabilisticConstraint(unittest.TestCase):

    def setUp(self):
        self.list_of_constraints = []
        self.parameters_estimation = {'quantile_distance': 0.05,
                                      'quantiles': (0.01, 0.99),
                                      'probabilities': (0.85, 0.95)}
        self.probabilistic_constraint = ProbabilisticConstraint(self.list_of_constraints, self.parameters_estimation)

    def test_creation(self):
        self.assertIsInstance(self.probabilistic_constraint, ProbabilisticConstraint)
        self.assertIsInstance(self.probabilistic_constraint.constraint, Constraint)
        self.assertIsInstance(self.probabilistic_constraint.bayesian_estimator, BayesianProbabilityEstimator)

    def test_create_constraint(self):
        constraint = self.probabilistic_constraint.create_constraint()
        self.assertIsInstance(constraint, Constraint)

        # Test case 1: True probability lies within the specified interval
        true_probability = torch.distributions.uniform.Uniform(0, 1).sample((1,)).item()
        list_of_constraints = [
            Constraint(lambda opt_algo: True)
            if torch.distributions.uniform.Uniform(0, 1).sample((1,)).item() < true_probability
            else Constraint(lambda opt_algo: False) for _ in range(100)
        ]
        self.parameters_estimation['probabilities'] = (true_probability - 0.1, true_probability + 0.1)
        self.probabilistic_constraint = ProbabilisticConstraint(list_of_constraints,
                                                                self.parameters_estimation)
        constraint = self.probabilistic_constraint.create_constraint()
        # Here, since the constraints to not really need an optimization algorithm, we can just call the constraint
        # in any way we want.
        result = constraint(1, also_return_value=False)
        self.assertTrue(result)

        result, estimation = constraint(1, also_return_value=True)
        self.assertTrue(result)
        self.assertTrue(self.parameters_estimation['probabilities'][0]
                        <= estimation <= self.parameters_estimation['probabilities'][1])

        # Test case 2: True probability does not lie within the specified interval
        true_probability = torch.distributions.uniform.Uniform(0.75, 1).sample((1,)).item()
        list_of_constraints = [
            Constraint(lambda opt_algo: True)
            if torch.distributions.uniform.Uniform(0, 1).sample((1,)).item() < true_probability
            else Constraint(lambda opt_algo: False) for _ in range(100)
        ]
        self.parameters_estimation['probabilities'] = (0.0, 0.1)
        self.probabilistic_constraint = ProbabilisticConstraint(list_of_constraints,
                                                                self.parameters_estimation)
        constraint = self.probabilistic_constraint.create_constraint()
        result = constraint(1, also_return_value=False)
        self.assertFalse(result)

        result, estimation = constraint(1, also_return_value=True)
        self.assertFalse(result)
        self.assertFalse(self.parameters_estimation['probabilities'][0]
                         <= estimation <= self.parameters_estimation['probabilities'][1])

