import unittest
import torch
from exponential_family.sufficient_statistics.sufficient_statistics import evaluate_sufficient_statistics
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import (
    PacBayesOptimizationAlgorithm)
from algorithms.dummy import Dummy


class TestSufficientStatistics(unittest.TestCase):

    def setUp(self):

        def dummy_function(x, parameter):
            return parameter['p'] * torch.linalg.norm(x) ** 2

        self.dummy_function = dummy_function
        dim = torch.randint(low=1, high=1000, size=(1,)).item()
        length_state = 1  # Take one, because it has to be compatible with Dummy()
        self.initial_state = torch.randn(size=(length_state, dim))
        self.parameter = {'p': 1}
        self.loss_function = ParametricLossFunction(function=dummy_function, parameter=self.parameter)
        self.n_max = torch.randint(low=2, high=25, size=(1,)).item()
        self.pac_parameters = {'sufficient_statistics': None,
                               'natural_parameters': None,
                               'covering_number': None,
                               'epsilon': None,
                               'n_max': self.n_max}
        self.pac_algorithm = PacBayesOptimizationAlgorithm(implementation=Dummy(),
                                                           initial_state=self.initial_state,
                                                           loss_function=self.loss_function,
                                                           pac_parameters=self.pac_parameters)

    def test_constraint_not_satisfied(self):

        def dummy_constraint(loss_at_beginning, loss_at_end):
            return False

        values = evaluate_sufficient_statistics(
            optimization_algorithm=self.pac_algorithm,
            loss_function=ParametricLossFunction(function=self.dummy_function, parameter={'p': 1}),
            constants=torch.tensor(1),
            convergence_risk_constraint=dummy_constraint,
            convergence_probability=torch.tensor(1.)
        )

        # In case the constraint is not satisfied, the function should return [0, 0].
        self.assertTrue(torch.equal(values, torch.tensor([0.0, 0.0])))

    def test_constraint_satisfied(self):

        def dummy_constraint(loss_at_beginning, loss_at_end):
            return True

        constants = torch.randint(low=1, high=1000, size=(1,))
        convergence_probability = torch.rand((1,))
        values = evaluate_sufficient_statistics(
            optimization_algorithm=self.pac_algorithm,
            loss_function=ParametricLossFunction(function=self.dummy_function, parameter={'p': 1}),
            constants=constants,
            convergence_risk_constraint=dummy_constraint,
            convergence_probability=convergence_probability
        )
        loss_at_end = self.pac_algorithm.loss_function(self.pac_algorithm.current_iterate)

        # In case the constraint is satisfied, the function should actually compute the values.
        self.assertTrue(torch.equal(
            values, torch.tensor([-loss_at_end/convergence_probability, constants / (convergence_probability**2)])
        ))
