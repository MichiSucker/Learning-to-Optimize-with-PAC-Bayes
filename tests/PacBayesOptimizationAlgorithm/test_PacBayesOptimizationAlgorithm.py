import unittest
from types import NoneType

from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import (
    PacBayesOptimizationAlgorithm, compute_loss_at_end)
import torch
from typing import Callable
from classes.LossFunction.class_LossFunction import LossFunction
from classes.Constraint.class_ProbabilisticConstraint import Constraint, ProbabilisticConstraint
from algorithms.dummy import Dummy
from exponential_family.describing_property.reduction_property import instantiate_reduction_property_with
from experiments.quadratics.training import get_sufficient_statistics
import copy
import io
import sys


def dummy_function(x, parameter=None):
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

        # Instantiate setting
        sufficient_statistics = get_sufficient_statistics(constants=torch.tensor(1.))
        self.pac_algorithm.sufficient_statistics = sufficient_statistics
        self.pac_algorithm.n_max = 10
        number_of_hyperparameters = torch.randint(low=1, high=10, size=(1,)).item()
        loss_functions = [LossFunction(lambda x: 0.5 * torch.linalg.norm(x) ** 2) for _ in range(10)]
        hyperparameters = [copy.deepcopy(self.pac_algorithm.implementation.state_dict())
                           for _ in range(number_of_hyperparameters)]
        estimated_convergence_probabilities = [torch.randn((1,)) for _ in range(number_of_hyperparameters)]

        # Compute values
        values_of_sufficient_statistics = (
            self.pac_algorithm.evaluate_sufficient_statistics_on_all_parameters_and_hyperparameters(
                list_of_loss_functions=loss_functions,
                list_of_hyperparameters=hyperparameters,
                estimated_convergence_probabilities=estimated_convergence_probabilities))

        # Check that sizes do match-up: Note that we take the mean over loss_functions, so this is summed-up.
        self.assertEqual(values_of_sufficient_statistics.shape,
                         torch.Size((len(hyperparameters), 2)))

        # Check values
        desired_values = torch.zeros((len(loss_functions), len(hyperparameters), 2))
        for j, current_hyperparameters in enumerate(hyperparameters):
            self.pac_algorithm.set_hyperparameters_to(current_hyperparameters)
            for i, loss_func in enumerate(loss_functions):
                desired_values[i, j, :] = sufficient_statistics(
                    self.pac_algorithm, loss_func, probability=estimated_convergence_probabilities[j])

        self.assertTrue(torch.equal(values_of_sufficient_statistics, torch.mean(desired_values, dim=0)))

    def test_compute_posterior_potentials_and_pac_bound(self):

        # Initialize setting
        def natural_parameters(x):
            return torch.tensor([x, -0.5 * x ** 2])

        sufficient_statistics = get_sufficient_statistics(constants=torch.tensor(1.))
        self.pac_algorithm.sufficient_statistics = sufficient_statistics
        self.pac_algorithm.natural_parameters = natural_parameters
        self.pac_algorithm.covering_number = torch.tensor(100)
        self.pac_algorithm.epsilon = torch.tensor(0.05)
        self.pac_algorithm.n_max = 10

        number_of_loss_functions = torch.randint(low=1, high=10, size=(1,)).item()
        number_of_hyperparameters = torch.randint(low=1, high=10, size=(1,)).item()
        loss_functions = [LossFunction(lambda x: torch.linalg.norm(x)) for _ in range(number_of_loss_functions)]
        hyperparameters = [copy.deepcopy(self.pac_algorithm.implementation.state_dict())
                           for _ in range(number_of_hyperparameters)]
        estimated_convergence_probabilities = [torch.randn((1,)) for _ in range(number_of_hyperparameters)]
        potentials_prior = torch.rand(size=(len(hyperparameters),))

        # Make sure that values are set to None
        self.assertIsInstance(self.pac_algorithm.pac_bound, NoneType)
        self.assertIsInstance(self.pac_algorithm.optimal_lambda, NoneType)

        # Compute optimal values. Check that PAC-bound and optimal lambda are set correctly afterward, and that we have
        # the correct number of values.
        optimal_values_potentials_posterior = self.pac_algorithm.compute_posterior_potentials_and_pac_bound(
            samples_prior=hyperparameters, potentials_prior=potentials_prior,
            estimated_convergence_probabilities=estimated_convergence_probabilities,
            list_of_loss_functions_train=loss_functions)

        self.assertIsInstance(self.pac_algorithm.pac_bound, torch.Tensor)
        self.assertIsInstance(self.pac_algorithm.optimal_lambda, torch.Tensor)
        self.assertTrue(len(optimal_values_potentials_posterior), len(hyperparameters))

    def test_get_posterior_potentials_as_function_of_lambda(self):

        # Initialize setting.
        def natural_parameters(x):
            return torch.tensor([x, -0.5 * x**2])

        sufficient_statistics = get_sufficient_statistics(constants=torch.tensor(1.))
        self.pac_algorithm.sufficient_statistics = sufficient_statistics
        self.pac_algorithm.natural_parameters = natural_parameters
        self.pac_algorithm.n_max = 10
        number_of_functions = torch.randint(low=1, high=10, size=(1,)).item()
        number_of_hyperparameters = torch.randint(low=1, high=10, size=(1,)).item()
        loss_functions = [LossFunction(lambda x: torch.linalg.norm(x) ** 2) for _ in range(number_of_functions)]
        hyperparameters = [copy.deepcopy(self.pac_algorithm.implementation.state_dict())
                           for _ in range(number_of_hyperparameters)]
        estimated_convergence_probabilities = [torch.randn((1,)) for _ in range(number_of_hyperparameters)]
        potentials_prior = torch.rand(size=(len(hyperparameters),))

        # Check that the posterior potentials are functions that return tensors which contain as many values as we have
        # hyperparameters.
        potentials_posterior = self.pac_algorithm.get_posterior_potentials_as_function_of_lambda(
            list_of_loss_functions_train=loss_functions,
            samples_prior=hyperparameters,
            estimated_convergence_probabilities=estimated_convergence_probabilities,
            potentials_prior=potentials_prior
        )
        self.assertIsInstance(potentials_posterior, Callable)
        self.assertIsInstance(potentials_posterior(1.), torch.Tensor)
        self.assertEqual(len(hyperparameters), len(potentials_posterior(1.)))

    def test_get_upper_bound_as_function_of_lambda(self):

        def potentials(x):
            return torch.exp(x)

        def natural_parameters(x):
            return torch.tensor([x, -0.5 * x ** 2])

        self.pac_algorithm.epsilon = torch.rand(size=(1,))
        self.pac_algorithm.covering_number = torch.randint(low=1, high=100, size=(1,))
        self.pac_algorithm.natural_parameters = natural_parameters

        upper_bound = self.pac_algorithm.get_upper_bound_as_function_of_lambda(potentials=potentials)

        # Check that the return-value is a function.
        self.assertIsInstance(upper_bound, Callable)

        # Call it with a random instance to check the output.
        lamb = torch.rand(size=(1,))
        self.assertIsInstance(upper_bound(lamb), torch.Tensor)
        self.assertEqual(upper_bound(lamb),
                         -(torch.logsumexp(potentials(lamb), dim=0)
                           + torch.log(self.pac_algorithm.epsilon)
                           - torch.log(self.pac_algorithm.covering_number))
                         / (self.pac_algorithm.natural_parameters(lamb)[0]))

    def test_minimize_upper_bound_in_lambda_optimal_value_on_the_left(self):
        self.pac_algorithm.covering_number = 100

        # Upper bound minimized on the left end.
        def upper_bound(x):
            return x**2

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        best_value, best_lambda = self.pac_algorithm.minimize_upper_bound_in_lambda(upper_bound=upper_bound)
        sys.stdout = sys.__stdout__

        self.assertTrue(len(capturedOutput.getvalue()) > 0)
        self.assertEqual(best_lambda, 1e-8)
        self.assertEqual(best_value, upper_bound(1e-8))

    def test_minimize_upper_bound_in_lambda_optimal_value_on_the_right(self):
        self.pac_algorithm.covering_number = 100

        # Upper bound minimized on the right end.
        def upper_bound(x):
            return -x**2

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        best_value, best_lambda = self.pac_algorithm.minimize_upper_bound_in_lambda(upper_bound=upper_bound)
        sys.stdout = sys.__stdout__

        self.assertTrue(len(capturedOutput.getvalue()) > 0)
        self.assertEqual(best_lambda, 1e2)
        self.assertEqual(best_value, upper_bound(1e2))

    def test_minimize_upper_bound_in_lambda_optimal_value_in_between(self):
        self.pac_algorithm.covering_number = 100

        # Upper bound minimized in-between
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

        # Check that setting works.
        self.assertIsInstance(self.pac_algorithm.pac_bound, NoneType)
        new_pac_bound = torch.randn(size=(1,))
        self.pac_algorithm.set_variable__pac_bound__to(new_pac_bound)
        self.assertEqual(self.pac_algorithm.pac_bound, new_pac_bound)

        # Also check that an error is raised if the pac-bound is already set.
        with self.assertRaises(Exception):
            self.pac_algorithm.set_variable__pac_bound__to(10)

    def test_set_variable__optimal_lambda__to(self):

        # Check that setting works.
        self.assertIsInstance(self.pac_algorithm.optimal_lambda, NoneType)
        new_optimal_lambda = torch.randn(size=(1,))
        self.pac_algorithm.set_variable__optimal_lambda__to(new_optimal_lambda)
        self.assertEqual(self.pac_algorithm.optimal_lambda, new_optimal_lambda)

        # Also check that an error is raised if the optimal lambda is already set.
        with self.assertRaises(Exception):
            self.pac_algorithm.set_variable__optimal_lambda__to(10)

    def test_evaluate_convergence_risk_if_constraint_is_satisfied(self):
        # Initialize setting
        estimated_convergence_probability = torch.rand((1,))
        n_max = torch.randint(low=1, high=100, size=(1,)).item()
        self.pac_algorithm.n_max = n_max
        loss_functions = [LossFunction(function=dummy_function) for _ in range(10)]
        constraint_functions = [Constraint(function=lambda x: True) for _ in range(10)]
        convergence_risk = self.pac_algorithm.evaluate_convergence_risk(
            loss_functions=loss_functions,
            constraint_functions=constraint_functions,
            estimated_convergence_probability=estimated_convergence_probability)

        # Constraint is satisfied => Check that return-value is correct.
        self.assertIsInstance(convergence_risk, torch.Tensor)
        self.pac_algorithm.reset_state_and_iteration_counter()
        loss_at_end = compute_loss_at_end(self.pac_algorithm)
        self.assertTrue(torch.allclose(convergence_risk, loss_at_end/estimated_convergence_probability))

    def test_evaluate_convergence_risk_if_constraint_is_not_satisfied(self):
        # Initialize setting
        estimated_convergence_probability = torch.rand((1,))
        n_max = torch.randint(low=1, high=100, size=(1,)).item()
        self.pac_algorithm.n_max = n_max
        loss_functions = [LossFunction(function=dummy_function) for _ in range(10)]
        constraint_functions = [Constraint(function=lambda x: False) for _ in range(10)]
        convergence_risk = self.pac_algorithm.evaluate_convergence_risk(
            loss_functions=loss_functions,
            constraint_functions=constraint_functions,
            estimated_convergence_probability=estimated_convergence_probability)

        # Constraint is not satisfied => Should yield zero loss.
        self.assertIsInstance(convergence_risk, torch.Tensor)
        self.assertTrue(torch.allclose(convergence_risk, torch.zeros(1)))

    def test_evaluate_prior_potentials(self):
        # Initialize setting
        number_of_hyperparameters = torch.randint(low=1, high=10, size=(1,)).item()
        self.pac_algorithm.n_max = torch.randint(low=1, high=10, size=(1,)).item()
        hyperparameters = [copy.deepcopy(self.pac_algorithm.implementation.state_dict())
                           for _ in range(number_of_hyperparameters)]
        estimated_convergence_probabilities = list(torch.rand((len(hyperparameters), )))
        loss_functions = [LossFunction(function=dummy_function) for _ in range(5)]
        constraint_functions = [Constraint(function=lambda x: True) for _ in range(5)]

        # Check that the number of prior potentials is correct.
        # Note that the functions inside evaluate_prior_potentials, which compute the actual values, get tested
        # separately.
        prior_potentials = self.pac_algorithm.evaluate_prior_potentials(
            loss_functions_prior=loss_functions,
            constraint_functions_prior=constraint_functions,
            samples_prior=hyperparameters,
            estimated_convergence_probabilities=estimated_convergence_probabilities)
        self.assertIsInstance(prior_potentials, torch.Tensor)
        self.assertEqual(len(prior_potentials), len(hyperparameters))

    def test_set_maximum_likelihood(self):

        num_samples = torch.randint(low=0, high=10, size=(1,)).item()
        samples = [copy.deepcopy(self.pac_algorithm.implementation.state_dict()) for _ in range(num_samples)]
        # Change values for each sample
        for s in samples:
            s['scale'] -= torch.randn((1,)).item()
        random_numbers = torch.distributions.uniform.Uniform(0, 1).sample((num_samples,))

        # Check that the sample with the best value got selected
        self.pac_algorithm.set_hyperparameters_to_maximum_likelihood(
            state_dict_samples=samples, posterior_potentials=random_numbers)

        best = samples[torch.argmax(torch.softmax(random_numbers, dim=0))]
        self.assertEqual(self.pac_algorithm.implementation.state_dict()['scale'],
                         best['scale'])

    def test_pac_bayes_fit(self):

        def natural_parameters(x):
            return torch.tensor([x, -0.5 * x**2])

        # Instantiate setting
        sufficient_statistics = get_sufficient_statistics(constants=torch.tensor(1.))
        self.pac_algorithm.sufficient_statistics = sufficient_statistics
        self.pac_algorithm.natural_parameters = natural_parameters
        true_probability = torch.distributions.uniform.Uniform(0.9, 1.0).sample((1,)).item()
        list_of_constraints = [
            Constraint(lambda opt_algo: True)
            if torch.distributions.uniform.Uniform(0, 1).sample((1,)).item() < true_probability
            else Constraint(lambda opt_algo: False) for _ in range(200)
        ]
        parameters_estimation = {'quantile_distance': 0.05, 'quantiles': (0.01, 0.99),
                                 'probabilities': (true_probability - 0.1, 1)}
        probabilistic_constraint = ProbabilisticConstraint(list_of_constraints, parameters_estimation)
        constraint = probabilistic_constraint.create_constraint()
        self.pac_algorithm.set_constraint(constraint)
        self.pac_algorithm.n_max = 10
        self.pac_algorithm.covering_number = torch.randint(low=10, high=100, size=(1,))
        self.pac_algorithm.epsilon = torch.rand(size=(1,))
        loss_functions_prior = [ParametricLossFunction(function=dummy_function, parameter={'p': 1}) for _ in range(5)]
        loss_functions_train = [ParametricLossFunction(function=dummy_function, parameter={'p': 1}) for _ in range(5)]
        fitting_parameters = {'restart_probability': 0.5, 'length_trajectory': 1, 'n_max': 10,
                              'num_iter_update_stepsize': 5, 'factor_stepsize_update': 0.5, 'lr': 1e-4}
        number_of_samples = 2
        sampling_parameters = {'restart_probability': 0.9, 'length_trajectory': 1, 'lr': 1e-4,
                               'num_samples': number_of_samples, 'num_iter_burnin': 1}
        reduction_property, convergence_risk_constraint, empirical_second_moment = (
            instantiate_reduction_property_with(factor=1., exponent=2.))
        constraint_parameters = {'num_iter_update_constraint': 5, 'describing_property': reduction_property}
        update_parameters = {'with_print': True, 'num_iter_print_update': 10, 'bins': []}

        # Check that procedure runs through, and that
        #   1) the PAC-bound is a FloatTensor with a single entry
        #   2) the PAC-bound got stored 'inside' the algorithm
        #   3) with a created the correct amount of samples.
        pac_bound, state_dict_samples_prior = self.pac_algorithm.pac_bayes_fit(
            loss_functions_prior=loss_functions_prior, loss_functions_train=loss_functions_train,
            fitting_parameters=fitting_parameters, sampling_parameters=sampling_parameters,
            constraint_parameters=constraint_parameters, update_parameters=update_parameters)
        self.assertIsInstance(pac_bound, torch.FloatTensor)
        self.assertIsInstance(pac_bound.item(), float)
        self.assertEqual(pac_bound, self.pac_algorithm.pac_bound)
        self.assertTrue(len(state_dict_samples_prior), number_of_samples)
        self.assertTrue(self.pac_algorithm.implementation.state_dict() in state_dict_samples_prior)


class TestHelpers(unittest.TestCase):

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

    def test_compute_loss_at_end(self):
        n_max = torch.randint(low=1, high=100, size=(1,)).item()
        self.pac_algorithm.n_max = n_max
        loss_at_end = compute_loss_at_end(self.pac_algorithm)
        self.assertIsInstance(loss_at_end, float)
        self.assertEqual(self.pac_algorithm.iteration_counter, n_max)
