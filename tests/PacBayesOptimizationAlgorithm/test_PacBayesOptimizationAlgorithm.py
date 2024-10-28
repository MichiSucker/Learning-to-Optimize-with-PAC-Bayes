import unittest
from types import NoneType

from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import (
    PacBayesOptimizationAlgorithm)
import torch
from typing import Callable
from classes.LossFunction.class_LossFunction import LossFunction
from algorithms.dummy import Dummy
import copy
import io
import sys


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
                    self.pac_algorithm, parameter=current_parameters,
                    probability=estimated_convergence_probabilities[j])

        self.assertTrue(torch.equal(values_of_sufficient_statistics, torch.mean(desired_values, dim=0)))

    def test_get_upper_bound_as_function_of_lambda(self):
        def potentials(x):
            return torch.exp(x)

        def natural_parameters(x):
            return torch.tensor([x, -0.5 * x ** 2])

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

    def test_minimize_upper_bound_in_lambda(self):
        self.pac_algorithm.covering_number = 100

        # Test case 1: Upper bound minimized on the left end.
        def upper_bound(x):
            return x**2

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        best_value, best_lambda = self.pac_algorithm.minimize_upper_bound_in_lambda(upper_bound=upper_bound)
        sys.stdout = sys.__stdout__

        self.assertTrue(len(capturedOutput.getvalue()) > 0)
        self.assertEqual(best_lambda, 1e-8)
        self.assertEqual(best_value, upper_bound(1e-8))

        # Test case 2: Upper bound minimized on the right end.
        def upper_bound(x):
            return -x**2

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        best_value, best_lambda = self.pac_algorithm.minimize_upper_bound_in_lambda(upper_bound=upper_bound)
        sys.stdout = sys.__stdout__

        self.assertTrue(len(capturedOutput.getvalue()) > 0)
        self.assertEqual(best_lambda, 1e2)
        self.assertEqual(best_value, upper_bound(1e2))

        # Test case 3: Upper bound minimized in-between
        def upper_bound(x):
            return (x-50)**2

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        best_value, best_lambda = self.pac_algorithm.minimize_upper_bound_in_lambda(upper_bound=upper_bound)
        sys.stdout = sys.__stdout__

        capital_lambda = torch.linspace(start=1e-8, end=1e2, steps=int(self.pac_algorithm.covering_number))
        values_upper_bound = torch.stack([upper_bound(lamb) for lamb in capital_lambda])
        best_control_lambda = capital_lambda[torch.argmin(values_upper_bound)]
        best_control_value = torch.min(values_upper_bound)

        self.assertTrue(len(capturedOutput.getvalue()) == 0)
        self.assertEqual(best_lambda, best_control_lambda)
        self.assertEqual(best_value, best_control_value)

    def test_set_variable__pac_bound__to(self):
        self.assertIsInstance(self.pac_algorithm.pac_bound, NoneType)
        new_pac_bound = torch.randn(size=(1,))
        self.pac_algorithm.set_variable__pac_bound__to(new_pac_bound)
        self.assertEqual(self.pac_algorithm.pac_bound, new_pac_bound)
        with self.assertRaises(Exception):
            self.pac_algorithm.set_variable__pac_bound__to(10)

    def test_set_variable__optimal_lambda__to(self):
        self.assertIsInstance(self.pac_algorithm.optimal_lambda, NoneType)
        new_optimal_lambda = torch.randn(size=(1,))
        self.pac_algorithm.set_variable__optimal_lambda__to(new_optimal_lambda)
        self.assertEqual(self.pac_algorithm.optimal_lambda, new_optimal_lambda)
        with self.assertRaises(Exception):
            self.pac_algorithm.set_variable__optimal_lambda__to(10)

    def test_get_posterior_potentials_as_function_of_lambda(self):
        def sufficient_statistics(optimization_algorithm, parameter, probability):
            return torch.tensor([parameter['p']/probability, (parameter['p']/probability) ** 2])

        def natural_parameters(x):
            return torch.tensor([x, -0.5 * x**2])
        self.pac_algorithm.sufficient_statistics = sufficient_statistics
        self.pac_algorithm.natural_parameters = natural_parameters
        number_of_parameters = torch.randint(low=1, high=10, size=(1,)).item()
        number_of_hyperparameters = torch.randint(low=1, high=10, size=(1,)).item()
        parameters = [{'p': torch.randn((1,)).item()} for _ in range(number_of_parameters)]
        hyperparameters = [copy.deepcopy(self.pac_algorithm.implementation.state_dict())
                           for _ in range(number_of_hyperparameters)]
        estimated_convergence_probabilities = [torch.randn((1,)).item() for _ in range(number_of_hyperparameters)]
        potentials_prior = torch.rand(size=(len(hyperparameters),))
        potentials_posterior = self.pac_algorithm.get_posterior_potentials_as_function_of_lambda(
            list_of_parameters_train=parameters,
            samples_prior=hyperparameters,
            estimated_convergence_probabilities=estimated_convergence_probabilities,
            potentials_prior=potentials_prior
        )
        self.assertIsInstance(potentials_posterior, Callable)
        self.assertIsInstance(potentials_posterior(1.), torch.Tensor)
        self.assertEqual(len(hyperparameters), len(potentials_posterior(1.)))

    def test_compute_pac_bound(self):

        def sufficient_statistics(optimization_algorithm, parameter, probability):
            return torch.tensor([parameter['p']/probability, (parameter['p']/probability) ** 2])

        def natural_parameters(x):
            return torch.tensor([x, -0.5 * x ** 2])

        self.pac_algorithm.sufficient_statistics = sufficient_statistics
        self.pac_algorithm.natural_parameters = natural_parameters
        self.pac_algorithm.covering_number = torch.tensor(100)
        self.pac_algorithm.epsilon = torch.tensor(0.05)
        number_of_parameters = torch.randint(low=1, high=10, size=(1,)).item()
        number_of_hyperparameters = torch.randint(low=1, high=10, size=(1,)).item()
        parameters = [{'p': torch.randn((1,)).item()} for _ in range(number_of_parameters)]
        hyperparameters = [copy.deepcopy(self.pac_algorithm.implementation.state_dict())
                           for _ in range(number_of_hyperparameters)]
        estimated_convergence_probabilities = [torch.randn((1,)).item() for _ in range(number_of_hyperparameters)]
        potentials_prior = torch.rand(size=(len(hyperparameters),))

        self.assertIsInstance(self.pac_algorithm.pac_bound, NoneType)
        self.assertIsInstance(self.pac_algorithm.optimal_lambda, NoneType)

        optimal_values_potentials_posterior = self.pac_algorithm.compute_posterior_potentials_and_pac_bound(
            samples_prior=hyperparameters, potentials_prior=potentials_prior,
            estimated_convergence_probabilities=estimated_convergence_probabilities,
            list_of_parameters_train=parameters)

        self.assertIsInstance(self.pac_algorithm.pac_bound, torch.Tensor)
        self.assertIsInstance(self.pac_algorithm.optimal_lambda, torch.Tensor)
        self.assertTrue(len(optimal_values_potentials_posterior), len(hyperparameters))
