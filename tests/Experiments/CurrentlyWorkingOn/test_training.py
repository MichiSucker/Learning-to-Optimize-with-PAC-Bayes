import unittest
from typing import Callable

import torch
from classes.Constraint.class_Constraint import Constraint
from algorithms.heavy_ball import HeavyBallWithFriction
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.LossFunction.class_LossFunction import LossFunction
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import \
    PacBayesOptimizationAlgorithm
from experiments.quadratics.algorithm import Quadratics
from experiments.quadratics.training import (get_number_of_datapoints,
                                             create_parametric_loss_functions_from_parameters,
                                             get_initial_state,
                                             get_baseline_algorithm,
                                             get_parameters_of_estimation,
                                             get_update_parameters,
                                             get_sampling_parameters,
                                             get_fitting_parameters,
                                             get_initialization_parameters,
                                             get_describing_property,
                                             get_constraint_parameters,
                                             get_pac_bayes_parameters,
                                             get_constraint,
                                             compute_constants_for_sufficient_statistics,
                                             get_sufficient_statistics,
                                             get_algorithm_for_learning,
                                             set_up_and_train_algorithm)
from experiments.quadratics.data_generation import get_data
from main import TESTING_LEVEL


class TestTraining(unittest.TestCase):

    def test_get_number_of_datapoints(self):
        number_of_datapoints = get_number_of_datapoints()
        self.assertIsInstance(number_of_datapoints, dict)
        self.assertTrue('prior' in number_of_datapoints.keys())
        self.assertTrue('train' in number_of_datapoints.keys())
        self.assertTrue('test' in number_of_datapoints.keys())
        self.assertTrue('validation' in number_of_datapoints.keys())

    def test_create_parametric_loss_functions_from_parameters(self):
        number_of_datapoints = get_number_of_datapoints()
        parameters, loss_function_of_algorithm, _, _, _ = get_data(number_of_datapoints)
        loss_functions = create_parametric_loss_functions_from_parameters(
            template_loss_function=loss_function_of_algorithm, parameters=parameters)
        self.assertIsInstance(loss_functions, dict)
        self.assertTrue('prior' in loss_functions.keys())
        self.assertTrue('train' in loss_functions.keys())
        self.assertTrue('test' in loss_functions.keys())
        self.assertTrue('validation' in loss_functions.keys())
        for name in ['prior', 'train', 'test', 'validation']:
            self.assertIsInstance(loss_functions[name], list)
            for function in loss_functions[name]:
                self.assertIsInstance(function, ParametricLossFunction)

    def test_get_initial_state(self):
        dim = torch.randint(low=1, high=100, size=(1,)).item()
        init_state = get_initial_state(dim)
        self.assertTrue(torch.equal(init_state, torch.zeros((2, dim))))

    def test_get_baseline_algorithm(self):
        number_of_datapoints = get_number_of_datapoints()
        parameters, loss_function_of_algorithm, mu_min, L_max, dim = get_data(number_of_datapoints)
        loss_functions = create_parametric_loss_functions_from_parameters(
            template_loss_function=loss_function_of_algorithm, parameters=parameters)
        std_algo = get_baseline_algorithm(
            loss_function=loss_functions['train'][0],
            smoothness_constant=mu_min,
            strong_convexity_constant=L_max,
            dim=dim)
        self.assertIsInstance(std_algo, OptimizationAlgorithm)
        self.assertIsInstance(std_algo.implementation, HeavyBallWithFriction)

    def test_get_parameters_of_estimation(self):
        parameters = get_parameters_of_estimation()
        self.assertIsInstance(parameters, dict)
        self.assertTrue('quantile_distance' in parameters.keys())
        self.assertTrue('quantiles' in parameters.keys())
        self.assertTrue('probabilities' in parameters.keys())

    def test_get_update_parameters(self):
        parameters = get_update_parameters()
        self.assertIsInstance(parameters, dict)
        self.assertTrue('num_iter_print_update' in parameters.keys())
        self.assertTrue('with_print' in parameters.keys())
        self.assertTrue('bins' in parameters.keys())

    def test_get_sampling_parameters(self):
        max_number_of_it = torch.randint(low=1, high=100, size=(1,)).item()
        parameters = get_sampling_parameters(max_number_of_it)
        self.assertIsInstance(parameters, dict)
        self.assertTrue('length_trajectory' in parameters.keys())
        self.assertTrue('lr' in parameters.keys())
        self.assertTrue('with_restarting' in parameters.keys())
        self.assertTrue('restart_probability' in parameters.keys())
        self.assertTrue('num_samples' in parameters.keys())
        self.assertTrue('num_iter_burnin' in parameters.keys())
        self.assertEqual(parameters['restart_probability'], 1/max_number_of_it)

    def test_get_fitting_parameters(self):
        max_number_of_it = torch.randint(low=1, high=100, size=(1,)).item()
        parameters = get_fitting_parameters(max_number_of_it)
        self.assertIsInstance(parameters, dict)
        self.assertTrue('length_trajectory' in parameters.keys())
        self.assertTrue('lr' in parameters.keys())
        self.assertTrue('restart_probability' in parameters.keys())
        self.assertTrue('n_max' in parameters.keys())
        self.assertTrue('num_iter_update_stepsize' in parameters.keys())
        self.assertTrue('factor_stepsize_update' in parameters.keys())
        self.assertEqual(parameters['restart_probability'], 1 / max_number_of_it)

    def test_get_initialization_parameters(self):
        parameters = get_initialization_parameters()
        self.assertIsInstance(parameters, dict)
        self.assertTrue('lr' in parameters.keys())
        self.assertTrue('num_iter_max' in parameters.keys())
        self.assertTrue('num_iter_print_update' in parameters.keys())
        self.assertTrue('num_iter_update_stepsize' in parameters.keys())
        self.assertTrue('with_print' in parameters.keys())

    def test_get_describing_property(self):
        reduction_property, convergence_risk_constraint, empirical_second_moment = get_describing_property()
        self.assertIsInstance(reduction_property, Callable)
        self.assertIsInstance(convergence_risk_constraint, Callable)
        self.assertIsInstance(empirical_second_moment, Callable)

    def test_get_constraint_parameters(self):
        number_of_training_iterations = torch.randint(low=1, high=100, size=(1,)).item()
        parameters = get_constraint_parameters(number_of_training_iterations)
        self.assertIsInstance(parameters, dict)
        self.assertTrue('describing_property' in parameters.keys())
        self.assertTrue('num_iter_update_constraint' in parameters.keys())

    def test_get_pac_bayes_parameters(self):
        parameters = get_pac_bayes_parameters(torch.tensor(1.))
        self.assertIsInstance(parameters, dict)
        self.assertTrue('sufficient_statistics' in parameters.keys())
        self.assertTrue('natural_parameters' in parameters.keys())
        self.assertTrue('covering_number' in parameters.keys())
        self.assertTrue('epsilon' in parameters.keys())
        self.assertTrue('n_max' in parameters.keys())

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

    def test_compute_constants_for_sufficient_statistics(self):
        factor = 1.
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

    def test_get_algorithm_for_learning(self):
        number_of_datapoints = get_number_of_datapoints()
        parameters, loss_function_of_algorithm, mu_min, L_max, dim = get_data(number_of_datapoints)
        loss_functions = create_parametric_loss_functions_from_parameters(
            template_loss_function=loss_function_of_algorithm, parameters=parameters)
        algo = get_algorithm_for_learning(loss_function_of_algorithm, loss_functions, dim)
        self.assertIsInstance(algo, PacBayesOptimizationAlgorithm)
        self.assertIsInstance(algo.implementation, Quadratics)

    @unittest.skipIf(condition=(TESTING_LEVEL != 'FULL_TEST_WITH_EXPERIMENTS'),
                     reason='Too expensive to test all the time.')
    def test_run_nn_training_experiment(self):
        set_up_and_train_algorithm('/home/michael/Desktop/JMLR_New/Experiments/quadratics')
