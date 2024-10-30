import unittest
from typing import Callable

import torch
from main import TESTING_LEVEL
from algorithms.gradient_descent import GradientDescent
from classes.Constraint.class_Constraint import Constraint
from classes.LossFunction.class_LossFunction import LossFunction
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import \
    PacBayesOptimizationAlgorithm
from natural_parameters.natural_parameters import evaluate_natural_parameters_at
from experiments.nn_training.training import (instantiate_neural_networks,
                                              create_parametric_loss_functions_from_parameters,
                                              get_algorithm_for_initialization,
                                              get_initial_state,
                                              get_constraint,
                                              get_sufficient_statistics,
                                              get_describing_property,
                                              compute_constants_for_sufficient_statistics,
                                              instantiate_algorithm_for_learning,
                                              get_update_parameters,
                                              get_fitting_parameters,
                                              get_sampling_parameters,
                                              get_pac_bayes_parameters,
                                              get_parameters_of_estimation,
                                              get_initialization_parameters,
                                              get_constraint_parameters,
                                              get_number_of_datapoints,
                                              set_up_and_train_algorithm)
from experiments.nn_training.neural_network import NeuralNetworkForStandardTraining, NeuralNetworkForLearning


class TestEvaluationNN(unittest.TestCase):

    def test_instantiate_neural_networks(self):
        nn_std, nn_learn = instantiate_neural_networks()
        self.assertIsInstance(nn_std, NeuralNetworkForStandardTraining)
        self.assertIsInstance(nn_learn, NeuralNetworkForLearning)
        self.assertEqual(nn_learn.degree, 5)

    def test_instantiate_parametric_loss_functions(self):

        def dummy_function(x, parameter):
            return parameter['scale'] * torch.linalg.norm(x)

        n_prior = torch.randint(low=1, high=10, size=(1,)).item()
        n_train = torch.randint(low=1, high=10, size=(1,)).item()
        n_test = torch.randint(low=1, high=10, size=(1,)).item()
        n_val = torch.randint(low=1, high=10, size=(1,)).item()
        parameters = {'prior': [{'scale': torch.rand(size=(1,)).item()} for _ in range(n_prior)],
                      'train': [{'scale': torch.rand(size=(1,)).item()} for _ in range(n_train)],
                      'test': [{'scale': torch.rand(size=(1,)).item()} for _ in range(n_test)],
                      'validation': [{'scale': torch.rand(size=(1,)).item()} for _ in range(n_val)]}

        loss_functions = create_parametric_loss_functions_from_parameters(template_loss_function=dummy_function,
                                                                          parameters=parameters)
        self.assertIsInstance(loss_functions, dict)
        self.assertEqual(len(loss_functions['prior']), n_prior)
        self.assertEqual(len(loss_functions['train']), n_train)
        self.assertEqual(len(loss_functions['test']), n_test)
        self.assertEqual(len(loss_functions['validation']), n_val)

        for name in ['prior', 'train', 'test', 'validation']:
            for function in loss_functions[name]:
                self.assertIsInstance(function, ParametricLossFunction)

    def test_get_initial_state(self):
        dim = torch.randint(low=1, high=100, size=(1,)).item()
        x_0 = get_initial_state(dim=dim)
        self.assertEqual(x_0.shape, torch.Size((2, dim)))
        for _ in range(3):
            # Make sure, you get same init every time.
            self.assertTrue(torch.equal(x_0, get_initial_state(dim=dim)))

    def test_get_algorithm_for_initialization(self):

        def dummy_function(x):
            return torch.linalg.norm(x)

        dim = torch.randint(low=1, high=100, size=(1,)).item()
        x_0 = get_initial_state(dim=dim)
        algo = get_algorithm_for_initialization(initial_state_for_std_algorithm=x_0[-1].reshape((1, -1)),
                                                loss_function=LossFunction(function=dummy_function))
        self.assertIsInstance(algo, OptimizationAlgorithm)
        self.assertIsInstance(algo.implementation, GradientDescent)
        self.assertEqual(algo.initial_state.shape, torch.Size((1, dim)))

    def test_get_constraint(self):

        def dummy_function(x, parameter):
            return parameter['scale'] * torch.linalg.norm(x)

        parameters = {'prior': [{'scale': torch.rand(size=(1,)).item()} for _ in range(0)],
                      'train': [{'scale': torch.rand(size=(1,)).item()} for _ in range(0)],
                      'test': [{'scale': torch.rand(size=(1,)).item()} for _ in range(0)],
                      'validation': [{'scale': torch.rand(size=(1,)).item()} for _ in range(3)]}

        loss_functions = create_parametric_loss_functions_from_parameters(template_loss_function=dummy_function,
                                                                          parameters=parameters)
        constraint = get_constraint(parameters_of_estimation={'quantile_distance': 0.1,
                                                              'quantiles': (0.05, 0.95),
                                                              'probabilities': (0.9, 1.0)},
                                    loss_functions_for_constraint=loss_functions['validation'])
        self.assertIsInstance(constraint, Constraint)

    def test_get_describing_property(self):
        reduction_property, convergence_risk_constraint, empirical_second_moment = get_describing_property()
        self.assertIsInstance(reduction_property, Callable)
        self.assertIsInstance(convergence_risk_constraint, Callable)
        self.assertIsInstance(empirical_second_moment, Callable)

    def test_compute_constants_for_sufficient_statistics(self):
        factor = 0.25
        exponent = 0.5
        initial_state = torch.tensor([2.])
        number_of_loss_functions = 10
        loss_functions = [LossFunction(function=lambda x: x**2) for _ in range(number_of_loss_functions)]
        constant = compute_constants_for_sufficient_statistics(loss_functions_for_training=loss_functions,
                                                               initial_state=initial_state)
        # Note that we have to divide by N (number of loss_functions) twice: one for the computation of the mean, the
        # other time for the factor 1/N in the PAC-bound. Here, the first 1/N is skipped, as we use N-times the same
        # function x -> x**2.
        self.assertTrue(
            torch.allclose(constant,
                           (factor * (initial_state[-1].flatten() ** 2) ** exponent) ** 2 / len(loss_functions)))

    def test_get_sufficient_statistics(self):
        sufficient_statistics = get_sufficient_statistics(template_for_loss_function=lambda x: x**2,
                                                          constants=1)
        self.assertIsInstance(sufficient_statistics, Callable)

    def test_instantiate_algorithm_for_learning(self):

        def dummy_function(x, parameter):
            return parameter['scale'] * torch.linalg.norm(x)

        n_prior = torch.randint(low=1, high=10, size=(1,)).item()
        n_train = torch.randint(low=1, high=10, size=(1,)).item()
        n_test = torch.randint(low=1, high=10, size=(1,)).item()
        n_val = torch.randint(low=1, high=10, size=(1,)).item()
        parameters = {'prior': [{'scale': torch.rand(size=(1,)).item()} for _ in range(n_prior)],
                      'train': [{'scale': torch.rand(size=(1,)).item()} for _ in range(n_train)],
                      'test': [{'scale': torch.rand(size=(1,)).item()} for _ in range(n_test)],
                      'validation': [{'scale': torch.rand(size=(1,)).item()} for _ in range(n_val)]}

        loss_functions = create_parametric_loss_functions_from_parameters(template_loss_function=dummy_function,
                                                                          parameters=parameters)
        algo = instantiate_algorithm_for_learning(loss_function_for_algorithm=lambda x: x**2,
                                                  loss_functions=loss_functions,
                                                  dimension_of_hyperparameters=10)
        self.assertIsInstance(algo, PacBayesOptimizationAlgorithm)
        self.assertIsInstance(algo.sufficient_statistics, Callable)
        self.assertIsInstance(algo.natural_parameters, Callable)
        self.assertIsInstance(algo.n_max, int)
        self.assertIsInstance(algo.epsilon.item(), float)
        self.assertIsInstance(algo.constraint, Constraint)

    @unittest.skipIf(condition=(TESTING_LEVEL != 'FULL_TEST_WITH_EXPERIMENTS'),
                     reason='Too expensive to test all the time.')
    def test_parameters_of_experiment(self):
        number_of_datapoints = get_number_of_datapoints()
        self.assertEqual(number_of_datapoints, {'prior': 250, 'train': 250, 'test': 250, 'validation': 250})

        initialization_parameters = get_initialization_parameters()
        self.assertEqual(initialization_parameters,
                         {'lr': 1e-3, 'num_iter_max': 1000, 'num_iter_print_update': 200,
                          'num_iter_update_stepsize': 200, 'with_print': True})

        maximal_number_of_iterations = torch.randint(low=1, high=100, size=(1,)).item()
        fitting_parameters = get_fitting_parameters(maximal_number_of_iterations=maximal_number_of_iterations)
        self.assertEqual(fitting_parameters,
                         {'restart_probability': 1/maximal_number_of_iterations,
                          'length_trajectory': 1,
                          'n_max': int(100e3),
                          'lr': 1e-4,
                          'num_iter_update_stepsize': int(10e3),
                          'factor_stepsize_update': 0.5})

        estimation_parameters = get_parameters_of_estimation()
        self.assertEqual(estimation_parameters,
                         {'quantile_distance': 0.075, 'quantiles': (0.01, 0.99), 'probabilities': (0.95, 1.0)})

        def dummy_statistics(x):
            return x

        pac_bayes_parameters = get_pac_bayes_parameters(sufficient_statistics=dummy_statistics)
        self.assertEqual(pac_bayes_parameters,
                         {'sufficient_statistics': dummy_statistics,
                          'natural_parameters': evaluate_natural_parameters_at,
                          'covering_number': 75000, 'epsilon': 0.05, 'n_max': 100}
                         )

        update_parameters = get_update_parameters()
        self.assertEqual(update_parameters,
                         {'num_iter_print_update': 1000,
                          'with_print': True,
                          'bins': [1e6, 1e4, 1e2, 1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10][::-1]}
                         )

        maximal_number_of_iterations = torch.randint(low=1, high=100, size=(1,)).item()
        sampling_parameters = get_sampling_parameters(maximal_number_of_iterations)
        self.assertEqual(sampling_parameters,
                         {'lr': torch.tensor(1e-6),
                          'length_trajectory': 1,
                          'with_restarting': True,
                          'restart_probability': 1 / maximal_number_of_iterations,
                          'num_samples': 100,
                          'num_iter_burnin': 0}
                         )

        maximal_number_of_iterations = torch.randint(low=1, high=100, size=(1,)).item()
        constraint_parameters = get_constraint_parameters(maximal_number_of_iterations)
        self.assertEqual(constraint_parameters,
                         {'num_iter_update_constraint': int(maximal_number_of_iterations // 4)}
                         )


    @unittest.skipIf(condition=(TESTING_LEVEL == 'FULL_TEST_WITH_EXPERIMENTS'),
                     reason='Too expensive to test all the time.')
    def test_get_parameters(self):

        number_of_datapoints = get_number_of_datapoints()
        self.assertIsInstance(number_of_datapoints, dict)

        initialization_parameters = get_initialization_parameters()
        self.assertIsInstance(initialization_parameters, dict)

        maximal_number_of_iterations = torch.randint(low=1, high=100, size=(1,)).item()
        fitting_parameters = get_fitting_parameters(maximal_number_of_iterations=maximal_number_of_iterations)
        self.assertIsInstance(fitting_parameters, dict)

        estimation_parameters = get_parameters_of_estimation()
        self.assertIsInstance(estimation_parameters, dict)

        def dummy_statistics(x):
            return x

        pac_bayes_parameters = get_pac_bayes_parameters(sufficient_statistics=dummy_statistics)
        self.assertIsInstance(pac_bayes_parameters, dict)

        update_parameters = get_update_parameters()
        self.assertIsInstance(update_parameters, dict)

        maximal_number_of_iterations = torch.randint(low=1, high=100, size=(1,)).item()
        sampling_parameters = get_sampling_parameters(maximal_number_of_iterations)
        self.assertIsInstance(sampling_parameters, dict)

        maximal_number_of_iterations = torch.randint(low=1, high=100, size=(1,)).item()
        constraint_parameters = get_constraint_parameters(maximal_number_of_iterations)
        self.assertIsInstance(constraint_parameters, dict)

    @unittest.skipIf(condition=(TESTING_LEVEL != 'FULL_TEST_WITH_EXPERIMENTS'),
                     reason='Too expensive to test all the time.')
    def test_run_nn_training_experiment(self):
        set_up_and_train_algorithm('/home/michael/Desktop/JMLR_New/Experiments')