import unittest
import torch
from typing import Callable
import numpy as np

from algorithms.gradient_descent import GradientDescent
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.OptimizationAlgorithm.derived_classes.derived_classes.\
    subclass_PacBayesOptimizationAlgorithm import PacBayesOptimizationAlgorithm
from classes.Constraint.class_Constraint import Constraint
from classes.LossFunction.class_LossFunction import LossFunction
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from experiments.mnist.neural_network import NeuralNetworkForStandardTraining, NeuralNetworkForLearning
from experiments.mnist.training import (get_number_of_datapoints,
                                        get_initialization_parameters,
                                        get_fitting_parameters,
                                        get_sampling_parameters,
                                        get_update_parameters,
                                        get_parameters_of_estimation,
                                        get_constraint_parameters,
                                        get_pac_bayes_parameters,
                                        instantiate_neural_networks,
                                        create_parametric_loss_functions_from_parameters,
                                        get_algorithm_for_initialization,
                                        get_initial_state,
                                        get_describing_property,
                                        get_constraint,
                                        compute_constants_for_sufficient_statistics,
                                        get_sufficient_statistics,
                                        instantiate_algorithm_for_learning,
                                        create_folder_for_storing_data,
                                        save_data,
                                        set_up_and_train_algorithm)


class TestTraining(unittest.TestCase):

    def test_get_number_of_datapoints(self):
        # Check that each data set is specified
        number_of_datapoints = get_number_of_datapoints()
        self.assertTrue('prior' in number_of_datapoints.keys())
        self.assertTrue('train' in number_of_datapoints.keys())
        self.assertTrue('test' in number_of_datapoints.keys())
        self.assertTrue('validation' in number_of_datapoints.keys())
        self.assertTrue(len(number_of_datapoints.keys()) == 4)

    def test_get_initialization_parameters(self):
        # Check that each data set is specified, and only those.
        initialization_parameters = get_initialization_parameters()
        self.assertTrue('lr' in initialization_parameters.keys())
        self.assertTrue('num_iter_max' in initialization_parameters.keys())
        self.assertTrue('num_iter_print_update' in initialization_parameters.keys())
        self.assertTrue('num_iter_update_stepsize' in initialization_parameters.keys())
        self.assertTrue('with_print' in initialization_parameters.keys())
        self.assertTrue(len(initialization_parameters.keys()) == 5)

    def test_get_fitting_parameters(self):
        # Check that each data set is specified, and only those.
        # Also check that restart_probability is specified correctly.
        maximal_number_of_iterations = torch.randint(low=1, high=100, size=(1,)).item()
        fitting_parameters = get_fitting_parameters(maximal_number_of_iterations=maximal_number_of_iterations)
        self.assertTrue('restart_probability' in fitting_parameters.keys())
        self.assertTrue('length_trajectory' in fitting_parameters.keys())
        self.assertTrue('n_max' in fitting_parameters.keys())
        self.assertTrue('lr' in fitting_parameters.keys())
        self.assertTrue('num_iter_update_stepsize' in fitting_parameters.keys())
        self.assertTrue('factor_stepsize_update' in fitting_parameters.keys())
        self.assertTrue(len(fitting_parameters.keys()) == 6)
        self.assertTrue(fitting_parameters['restart_probability']
                        == fitting_parameters['length_trajectory']/maximal_number_of_iterations)

    def test_get_sampling_parameters(self):
        # Check that each data set is specified, and only those.
        # Also check that restart_probability is specified correctly.
        maximal_number_of_iterations = torch.randint(low=1, high=100, size=(1,)).item()
        sampling_parameters = get_sampling_parameters(maximal_number_of_iterations)
        self.assertTrue('restart_probability' in sampling_parameters.keys())
        self.assertTrue('length_trajectory' in sampling_parameters.keys())
        self.assertTrue('num_samples' in sampling_parameters.keys())
        self.assertTrue('lr' in sampling_parameters.keys())
        self.assertTrue('with_restarting' in sampling_parameters.keys())
        self.assertTrue('num_iter_burnin' in sampling_parameters.keys())
        self.assertTrue(len(sampling_parameters.keys()) == 6)
        self.assertTrue(sampling_parameters['restart_probability']
                        == sampling_parameters['length_trajectory']/maximal_number_of_iterations)

    def test_get_update_parameters(self):
        # Check that each data set is specified, and only those.
        update_parameters = get_update_parameters()
        self.assertTrue('num_iter_print_update' in update_parameters.keys())
        self.assertTrue('with_print' in update_parameters.keys())
        self.assertTrue('bins' in update_parameters.keys())
        self.assertTrue(len(update_parameters.keys()) == 3)

    def test_get_parameters_of_estimation(self):
        # Check that each data set is specified, and only those.
        estimation_parameters = get_parameters_of_estimation()
        self.assertTrue('quantile_distance' in estimation_parameters.keys())
        self.assertTrue('quantiles' in estimation_parameters.keys())
        self.assertTrue('probabilities' in estimation_parameters.keys())
        self.assertTrue(len(estimation_parameters.keys()) == 3)

    def test_get_constraint_parameters(self):
        # Check that each data set is specified, and only those.
        maximal_number_of_iterations = torch.randint(low=1, high=100, size=(1,)).item()
        constraint_parameters = get_constraint_parameters(maximal_number_of_iterations)
        self.assertTrue('describing_property' in constraint_parameters.keys())
        self.assertTrue('num_iter_update_constraint' in constraint_parameters.keys())
        self.assertTrue(len(constraint_parameters.keys()) == 2)

    def test_get_pac_bayes_parameters(self):

        def dummy_statistics(x):
            return x

        # Check that each data set is specified, and only those.
        pac_bayes_parameters = get_pac_bayes_parameters(sufficient_statistics=dummy_statistics)
        self.assertTrue('sufficient_statistics' in pac_bayes_parameters.keys())
        self.assertTrue('natural_parameters' in pac_bayes_parameters.keys())
        self.assertTrue('covering_number' in pac_bayes_parameters.keys())
        self.assertTrue('epsilon' in pac_bayes_parameters.keys())
        self.assertTrue('n_max' in pac_bayes_parameters.keys())
        self.assertTrue(len(pac_bayes_parameters.keys()) == 5)

    def test_instantiate_neural_networks(self):
        net_std, net_learn = instantiate_neural_networks()
        self.assertIsInstance(net_std, NeuralNetworkForStandardTraining)
        self.assertIsInstance(net_learn, NeuralNetworkForLearning)

    def test_create_parametric_loss_functions(self):

        # Initialize setting.
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

        # Check that we get a dictionary with four entries, where each entry corresponds to one data set and is a list
        # of ParametricLossFunctions.
        loss_functions = create_parametric_loss_functions_from_parameters(template_loss_function=dummy_function,
                                                                          parameters=parameters)
        self.assertIsInstance(loss_functions, dict)
        self.assertEqual(len(loss_functions['prior']), n_prior)
        self.assertEqual(len(loss_functions['train']), n_train)
        self.assertEqual(len(loss_functions['test']), n_test)
        self.assertEqual(len(loss_functions['validation']), n_val)
        self.assertTrue(len(loss_functions.keys()) == 4)

        for name in ['prior', 'train', 'test', 'validation']:
            for function in loss_functions[name]:
                self.assertIsInstance(function, ParametricLossFunction)

    def test_get_algorithm_for_initialization(self):

        def dummy_function(x):
            return torch.linalg.norm(x)

        # Check that we initialize with GradientDescent.
        dim = torch.randint(low=1, high=100, size=(1,)).item()
        x_0 = get_initial_state(dim=dim)
        algo = get_algorithm_for_initialization(initial_state_for_std_algorithm=x_0[-1].reshape((1, -1)),
                                                loss_function=LossFunction(function=dummy_function))
        self.assertIsInstance(algo, OptimizationAlgorithm)
        self.assertIsInstance(algo.implementation, GradientDescent)
        self.assertEqual(algo.initial_state.shape, torch.Size((1, dim)))

    def test_get_initial_state(self):
        dim = torch.randint(low=1, high=100, size=(1,)).item()
        x_0 = get_initial_state(dim=dim)
        self.assertEqual(x_0.shape, torch.Size((2, dim)))

    def test_get_describing_property(self):
        # Check that you get three functions.
        reduction_property, convergence_risk_constraint, empirical_second_moment = get_describing_property()
        self.assertIsInstance(reduction_property, Callable)
        self.assertIsInstance(convergence_risk_constraint, Callable)
        self.assertIsInstance(empirical_second_moment, Callable)

    def test_get_constraint(self):

        def dummy_function(x, parameter):
            return parameter['scale'] * torch.linalg.norm(x)

        parameters = {'prior': [{'scale': torch.rand(size=(1,)).item()} for _ in range(0)],
                      'train': [{'scale': torch.rand(size=(1,)).item()} for _ in range(0)],
                      'test': [{'scale': torch.rand(size=(1,)).item()} for _ in range(0)],
                      'validation': [{'scale': torch.rand(size=(1,)).item()} for _ in range(3)]}

        # Check that you get a Constraint-object.
        loss_functions = create_parametric_loss_functions_from_parameters(template_loss_function=dummy_function,
                                                                          parameters=parameters)
        constraint = get_constraint(parameters_of_estimation={'quantile_distance': 0.1,
                                                              'quantiles': (0.05, 0.95),
                                                              'probabilities': (0.9, 1.0)},
                                    loss_functions_for_constraint=loss_functions['validation'])
        self.assertIsInstance(constraint, Constraint)

    def test_compute_constants_for_sufficient_statistics(self):
        factor = 0.5
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
        # Check that you get a callable.
        sufficient_statistics = get_sufficient_statistics(constants=torch.tensor(1.))
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
        algo = instantiate_algorithm_for_learning(loss_functions=loss_functions,
                                                  dimension_of_hyperparameters=10)
        # Check that we get a PacBayesOptimizationAlgorithm, and the needed parameters are set.
        self.assertIsInstance(algo, PacBayesOptimizationAlgorithm)
        self.assertIsInstance(algo.sufficient_statistics, Callable)
        self.assertIsInstance(algo.natural_parameters, Callable)
        self.assertIsInstance(algo.n_max, int)
        self.assertIsInstance(algo.epsilon.item(), float)
        self.assertIsInstance(algo.constraint, Constraint)

    def test_create_folder(self):
        path = '/home/michael/Desktop/JMLR_New/Experiments/mnist/'
        create_folder_for_storing_data(path)

    def test_save_data(self):
        dummy_savings_path = '/home/michael/Desktop/JMLR_New/Experiments/mnist/dummy_data/'
        create_folder_for_storing_data(dummy_savings_path)
        save_data(savings_path=dummy_savings_path,
                  pac_bound=np.empty(1),
                  initial_state=np.empty(1),
                  number_of_iterations=0,
                  parameters={},
                  samples_prior=[],
                  best_sample={})

    @unittest.skip(reason='Too expensive to test all the time.')
    def test_run_nn_training_experiment(self):
        set_up_and_train_algorithm('/home/michael/Desktop/JMLR_New/Experiments/mnist/')
