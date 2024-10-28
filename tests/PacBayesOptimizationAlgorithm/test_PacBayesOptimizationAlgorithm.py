import unittest
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import (
    PacBayesOptimizationAlgorithm)
import torch
from typing import Callable
from classes.LossFunction.class_LossFunction import LossFunction
from algorithms.dummy import Dummy
import copy


def dummy_function(x):
    return 0.5 * torch.linalg.norm(x) ** 2


class TestPacBayesOptimizationAlgorithm(unittest.TestCase):

    def setUp(self):
        self.dim = torch.randint(low=1, high=1000, size=(1,)).item()
        self.length_state = 1  # Take one, because it has to be compatible with Dummy()
        self.initial_state = torch.randn(size=(self.length_state, self.dim))
        self.current_state = self.initial_state.clone()
        self.loss_function = LossFunction(function=dummy_function)
        self.pac_parameters = {'sufficient_statistics': None,
                               'natural_parameters': None,
                               'covering_number': None,
                               'epsilon': None,
                               'n_max': None}
        self.pac_algorithm = PacBayesOptimizationAlgorithm(implementation=Dummy(),
                                                           initial_state=self.initial_state,
                                                           loss_function=self.loss_function,
                                                           pac_parameters=self.pac_parameters)

    def test_creation(self):
        self.assertIsInstance(self.pac_algorithm, PacBayesOptimizationAlgorithm)

    def test_evaluate_sufficient_statistics_on_all_parameters_and_hyperparameters(self):

        def sufficient_statistics(optimization_algorithm, parameter, probability):
            return torch.tensor([parameter['p']/probability, (parameter['p']/probability) ** 2])

        self.pac_algorithm.sufficient_statistics = sufficient_statistics
        number_of_parameters = torch.randint(low=1, high=10, size=(1,)).item()
        number_of_hyperparameters = torch.randint(low=1, high=10, size=(1,)).item()
        parameters = [{'p': torch.randn((1,)).item()} for _ in range(number_of_parameters)]
        hyperparameters = [copy.deepcopy(self.pac_algorithm.implementation.state_dict())
                           for _ in range(number_of_hyperparameters)]
        estimated_convergence_probabilities = [torch.randn((1,)).item() for _ in range(number_of_hyperparameters)]
        values_of_sufficient_statistics = (
            self.pac_algorithm.evaluate_sufficient_statistics_on_all_parameters_and_hyperparameters(
                list_of_parameters=parameters,
                list_of_hyperparameters=hyperparameters,
                estimated_convergence_probabilities=estimated_convergence_probabilities))
        self.assertEqual(values_of_sufficient_statistics.shape, torch.Size((len(hyperparameters), 2)))

        desired_values = torch.zeros((len(parameters), len(hyperparameters), 2))
        for j, current_hyperparameters in enumerate(hyperparameters):
            self.pac_algorithm.set_hyperparameters_to(current_hyperparameters)
            for i, current_parameters in enumerate(parameters):
                desired_values[i, j, :] = sufficient_statistics(
                    self.pac_algorithm, parameter=current_parameters, probability=estimated_convergence_probabilities[j])

        self.assertTrue(torch.equal(values_of_sufficient_statistics, torch.mean(desired_values, dim=0)))

    def test_get_upper_bound_as_function_of_lambda(self):
        def potentials(lamb):
            return torch.exp(lamb)

        def natural_parameters(lamb):
            return torch.tensor([lamb, -0.5 * lamb ** 2])

        self.pac_algorithm.epsilon = torch.rand(size=(1,))
        self.pac_algorithm.covering_number = torch.randint(low=1, high=100, size=(1,))
        self.pac_algorithm.natural_parameters = natural_parameters

        upper_bound = self.pac_algorithm.get_upper_bound_as_function_of_lambda(potentials=potentials)
        self.assertIsInstance(upper_bound, Callable)
        lamb = torch.rand(size=(1,))
        self.assertIsInstance(upper_bound(lamb), torch.Tensor)
        self.assertEqual(upper_bound(lamb),
                         -(torch.logsumexp(potentials(lamb), dim=0)
                           + torch.log(self.pac_algorithm.epsilon)
                           - torch.log(self.pac_algorithm.covering_number))
                         / (self.pac_algorithm.natural_parameters(lamb)[0]))
