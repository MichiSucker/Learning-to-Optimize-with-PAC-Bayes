import unittest
from typing import Callable
import torch
import numpy as np
from algorithms.fista import FISTA
from classes.LossFunction.derived_classes.derived_classes.\
    subclass_NonsmoothParametricLossFunction import NonsmoothParametricLossFunction
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.Constraint.class_Constraint import Constraint
from classes.LossFunction.class_LossFunction import LossFunction
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import \
    PacBayesOptimizationAlgorithm
from experiments.lasso.algorithm import SparsityNet
from experiments.lasso.data_generation import get_data, get_dimensions
from experiments.lasso.training import (set_up_and_train_algorithm,
                                        get_number_of_datapoints,
                                        create_folder_for_storing_data,
                                        create_parametric_loss_functions_from_parameters,
                                        get_initial_states,
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
                                        save_data)


class TestTraining(unittest.TestCase):

    def setUp(self):
        self.path_of_experiment = '/home/michael/Desktop/JMLR_New/Experiments/lasso'
        self.dummy_savings_path = self.path_of_experiment + '/dummy_data/'

    def test_create_folder(self):
        # Weak test: Check that there is no error.
        path = create_folder_for_storing_data(self.path_of_experiment)
        self.assertEqual(path, self.path_of_experiment + '/data/')

    def test_get_number_of_datapoints(self):
        # Check that we specified each data set.
        number_datapoints = get_number_of_datapoints()
        self.assertIsInstance(number_datapoints, dict)
        self.assertTrue('prior' in number_datapoints.keys())
        self.assertTrue('train' in number_datapoints.keys())
        self.assertTrue('test' in number_datapoints.keys())
        self.assertTrue('validation' in number_datapoints.keys())
        self.assertTrue(len(number_datapoints.keys()) == 4)

    def test_get_parameters_of_estimation(self):
        # Check that we specified all the needed variables, and only these.
        p = get_parameters_of_estimation()
        self.assertIsInstance(p, dict)
        self.assertTrue('quantile_distance' in p.keys())
        self.assertTrue('quantiles' in p.keys())
        self.assertTrue('probabilities' in p.keys())
        self.assertTrue(len(p.keys()) == 3)

    def test_get_update_parameters(self):
        # Check that we specified all the needed variables, and only these.
        parameters = get_update_parameters()
        self.assertIsInstance(parameters, dict)
        self.assertTrue('num_iter_print_update' in parameters.keys())
        self.assertTrue('with_print' in parameters.keys())
        self.assertTrue('bins' in parameters.keys())

    def test_get_sampling_parameters(self):
        # Check that we specified all the needed variables, and only these.
        max_number_of_it = torch.randint(low=1, high=100, size=(1,)).item()
        parameters = get_sampling_parameters(max_number_of_it)
        self.assertIsInstance(parameters, dict)
        self.assertTrue('length_trajectory' in parameters.keys())
        self.assertTrue('lr' in parameters.keys())
        self.assertTrue('with_restarting' in parameters.keys())
        self.assertTrue('restart_probability' in parameters.keys())
        self.assertTrue('num_samples' in parameters.keys())
        self.assertTrue('num_iter_burnin' in parameters.keys())
        self.assertTrue(len(parameters.keys()) == 6)
        # Also check that restart_probability is calculated correctly.
        self.assertEqual(parameters['restart_probability'], 1/max_number_of_it)

    def test_get_fitting_parameters(self):
        # Check that we specified all the needed variables, and only these.
        max_number_of_it = torch.randint(low=1, high=100, size=(1,)).item()
        parameters = get_fitting_parameters(max_number_of_it)
        self.assertIsInstance(parameters, dict)
        self.assertTrue('length_trajectory' in parameters.keys())
        self.assertTrue('lr' in parameters.keys())
        self.assertTrue('restart_probability' in parameters.keys())
        self.assertTrue('n_max' in parameters.keys())
        self.assertTrue('num_iter_update_stepsize' in parameters.keys())
        self.assertTrue('factor_stepsize_update' in parameters.keys())
        self.assertTrue(len(parameters.keys()) == 6)
        # Also check that restart_probability is calculated correctly.
        self.assertEqual(parameters['restart_probability'], 1 / max_number_of_it)

    def test_get_initialization_parameters(self):
        # Check that we specified all the needed variables, and only these.
        parameters = get_initialization_parameters()
        self.assertIsInstance(parameters, dict)
        self.assertTrue('lr' in parameters.keys())
        self.assertTrue('num_iter_max' in parameters.keys())
        self.assertTrue('num_iter_print_update' in parameters.keys())
        self.assertTrue('num_iter_update_stepsize' in parameters.keys())
        self.assertTrue('with_print' in parameters.keys())
        self.assertTrue(len(parameters.keys()) == 5)

    def test_get_describing_property(self):
        # Check that we get three callable functions.
        reduction_property, convergence_risk_constraint, empirical_second_moment = get_describing_property()
        self.assertIsInstance(reduction_property, Callable)
        self.assertIsInstance(convergence_risk_constraint, Callable)
        self.assertIsInstance(empirical_second_moment, Callable)

    def test_get_constraint_parameters(self):
        # Check that we specified all the needed variables, and only these.
        number_of_training_iterations = torch.randint(low=1, high=100, size=(1,)).item()
        parameters = get_constraint_parameters(number_of_training_iterations)
        self.assertIsInstance(parameters, dict)
        self.assertTrue('describing_property' in parameters.keys())
        self.assertTrue('num_iter_update_constraint' in parameters.keys())
        self.assertTrue(len(parameters.keys()) == 2)

    def test_get_pac_bayes_parameters(self):
        # Check that we specified all the needed variables, and only these.
        parameters = get_pac_bayes_parameters(lambda x: None)
        self.assertIsInstance(parameters, dict)
        self.assertTrue('sufficient_statistics' in parameters.keys())
        self.assertTrue('natural_parameters' in parameters.keys())
        self.assertTrue('covering_number' in parameters.keys())
        self.assertTrue('epsilon' in parameters.keys())
        self.assertTrue('n_max' in parameters.keys())
        self.assertTrue(len(parameters.keys()) == 5)

    def test_get_constraint(self):

        # Initialize setting
        def dummy_function(x, parameter):
            return parameter['scale'] * torch.linalg.norm(x)

        parameters = {'prior': [{'scale': torch.rand(size=(1,)).item()} for _ in range(0)],
                      'train': [{'scale': torch.rand(size=(1,)).item()} for _ in range(0)],
                      'test': [{'scale': torch.rand(size=(1,)).item()} for _ in range(0)],
                      'validation': [{'scale': torch.rand(size=(1,)).item()} for _ in range(3)]}

        loss_functions = create_parametric_loss_functions_from_parameters(template_loss_function=dummy_function,
                                                                          smooth_part=dummy_function,
                                                                          nonsmooth_part=dummy_function,
                                                                          parameters=parameters)
        # Check that we get and Constraint-object indeed.
        constraint = get_constraint(parameters_of_estimation={'quantile_distance': 0.1,
                                                              'quantiles': (0.05, 0.95),
                                                              'probabilities': (0.9, 1.0)},
                                    loss_functions_for_constraint=loss_functions['validation'])
        self.assertIsInstance(constraint, Constraint)

    def test_compute_constants_for_sufficient_statistics(self):
        # Initialize setting
        factor = 1.
        exponent = 0.6
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
        # Check that we get a Callable.
        sufficient_statistics = get_sufficient_statistics(constants=torch.tensor(1.))
        self.assertIsInstance(sufficient_statistics, Callable)
        # And call it.
        number_of_datapoints = get_number_of_datapoints()
        parameters, loss_function_of_algorithm, smooth_part, nonsmooth_part, smoothness_parameter = get_data(
            number_of_datapoints)
        loss_functions = create_parametric_loss_functions_from_parameters(
            template_loss_function=loss_function_of_algorithm, smooth_part=smooth_part, nonsmooth_part=nonsmooth_part,
            parameters=parameters)
        initial_state, _ = get_initial_states()
        baseline = get_baseline_algorithm(smoothness_parameter=smoothness_parameter, initial_state=initial_state,
                                          loss_function=loss_functions['test'][0])
        baseline.n_max = 3
        x = sufficient_statistics(baseline, loss_functions['test'][0], probability=1.)
        self.assertIsInstance(x, torch.Tensor)

    def test_get_initial_states(self):
        # Check that initial states are specified correctly, and that they do match-up correctly.
        dim_b, dim_x = get_dimensions()
        x_0_fista, x_0_learned = get_initial_states()
        self.assertIsInstance(x_0_fista, torch.Tensor)
        self.assertIsInstance(x_0_learned, torch.Tensor)
        self.assertTrue(torch.equal(x_0_fista[1:], x_0_learned))
        self.assertEqual(x_0_fista.shape, torch.Size((3, dim_x)))
        self.assertEqual(x_0_learned.shape, torch.Size((2, dim_x)))

    def test_get_algorithm_for_learning(self):
        # Check that we get an PacBayesOptimizationAlgorithm with the correct implementation.
        number_of_datapoints = get_number_of_datapoints()
        parameters, loss_function_of_algorithm, smooth_part, nonsmooth_part, smoothness_parameter = get_data(
            number_of_datapoints)
        loss_functions = create_parametric_loss_functions_from_parameters(
            template_loss_function=loss_function_of_algorithm, smooth_part=smooth_part, nonsmooth_part=nonsmooth_part,
            parameters=parameters)
        algo = get_algorithm_for_learning(loss_functions=loss_functions, smoothness_parameter=smoothness_parameter)
        self.assertIsInstance(algo, PacBayesOptimizationAlgorithm)
        self.assertIsInstance(algo.implementation, SparsityNet)

    def test_create_parametric_loss_functions_from_parameters(self):
        # Check that parameters are turned into the correct number of NonsmoothParametricLossFunctions.
        number_of_datapoints = get_number_of_datapoints()
        parameters, loss_function_of_algorithm, smooth_part, nonsmooth_part, smoothness_parameter = get_data(
            number_of_datapoints)
        loss_functions = create_parametric_loss_functions_from_parameters(
            template_loss_function=loss_function_of_algorithm, smooth_part=smooth_part, nonsmooth_part=nonsmooth_part,
            parameters=parameters)
        self.assertIsInstance(loss_functions, dict)
        self.assertTrue('prior' in loss_functions.keys())
        self.assertTrue('train' in loss_functions.keys())
        self.assertTrue('test' in loss_functions.keys())
        self.assertTrue('validation' in loss_functions.keys())
        for name in ['prior', 'train', 'test', 'validation']:
            self.assertIsInstance(loss_functions[name], list)
            self.assertEqual(len(loss_functions[name]), number_of_datapoints[name])
            for function in loss_functions[name]:
                self.assertIsInstance(function, NonsmoothParametricLossFunction)

    def test_get_baseline_algorithm(self):
        # Check that we get an OptimizationAlgorithm with the correct implementation.
        number_of_datapoints = get_number_of_datapoints()
        parameters, loss_function_of_algorithm, smooth_part, nonsmooth_part, smoothness_parameter = get_data(
            number_of_datapoints)
        loss_functions = create_parametric_loss_functions_from_parameters(
            template_loss_function=loss_function_of_algorithm, smooth_part=smooth_part, nonsmooth_part=nonsmooth_part,
            parameters=parameters)
        initial_state, _ = get_initial_states()
        baseline = get_baseline_algorithm(smoothness_parameter=smoothness_parameter, initial_state=initial_state,
                                          loss_function=loss_functions['test'][0])
        self.assertIsInstance(baseline, OptimizationAlgorithm)
        self.assertIsInstance(baseline.implementation, FISTA)

    @unittest.skip('Too expensive to test all the time.')
    def test_run_nn_training_experiment(self):
        set_up_and_train_algorithm(self.path_of_experiment)

    def test_save_data(self):
        save_data(savings_path=self.dummy_savings_path,
                  smoothness_parameter=np.empty(1),
                  pac_bound=np.empty(1),
                  initialization_learned_algorithm=np.empty(1),
                  initialization_baseline_algorithm=np.empty(1),
                  number_of_iterations=0,
                  parameters={},
                  samples_prior=[],
                  best_sample={})
