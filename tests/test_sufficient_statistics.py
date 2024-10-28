import unittest
import torch
from sufficient_statistics.sufficient_statistics import evaluate_sufficient_statistics
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import (
    PacBayesOptimizationAlgorithm)
from algorithms.dummy import Dummy
from describing_property.reduction_property import instantiate_reduction_property_with


def dummy_function(x, parameter=None):
    return 0.5 * torch.linalg.norm(x) ** 2


class TestSufficientStatistics(unittest.TestCase):

    def setUp(self):
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
            parameter_of_loss_function={'p': 1},
            template_for_loss_function=dummy_function,
            constants=1,
            convergence_risk_constraint=dummy_constraint,
            convergence_probability=1.
        )
        self.assertTrue(torch.equal(values, torch.tensor([0.0, 0.0])))

    def test_constraint_satisfied(self):

        def dummy_constraint(loss_at_beginning, loss_at_end):
            return True

        constants = torch.randint(low=1, high=1000, size=(1,)).item()
        convergence_probability = torch.rand((1,)).item()
        values = evaluate_sufficient_statistics(
            optimization_algorithm=self.pac_algorithm,
            parameter_of_loss_function={'p': 1},
            template_for_loss_function=dummy_function,
            constants=constants,
            convergence_risk_constraint=dummy_constraint,
            convergence_probability=convergence_probability
        )
        loss_at_end = self.pac_algorithm.loss_function(self.pac_algorithm.current_iterate)
        self.assertTrue(torch.equal(
            values, torch.tensor([-loss_at_end/convergence_probability, constants / (convergence_probability**2)])
        ))
