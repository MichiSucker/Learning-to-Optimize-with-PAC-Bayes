import unittest
from typing import Callable
import torch
import numpy as np
from algorithms.nesterov_accelerated_gradient_descent import NesterovAcceleratedGradient
from classes.Constraint.class_Constraint import Constraint
from classes.LossFunction.class_LossFunction import LossFunction
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import \
    PacBayesOptimizationAlgorithm
from experiments.image_processing.algorithm import ConvNet
from experiments.image_processing.data_generation import get_data
from experiments.image_processing.training import (set_up_and_train_algorithm,
                                                   get_baseline_algorithm,
                                                   get_initial_states,
                                                   get_dimension_of_optimization_variable,
                                                   get_image_height_and_width,
                                                   create_folder_for_storing_data,
                                                   get_number_of_datapoints,
                                                   get_parameters_of_estimation,
                                                   get_update_parameters,
                                                   get_sampling_parameters,
                                                   get_fitting_parameters,
                                                   get_initialization_parameters,
                                                   get_describing_property,
                                                   get_constraint_parameters,
                                                   get_pac_bayes_parameters,
                                                   create_parametric_loss_functions_from_parameters,
                                                   get_constraint,
                                                   compute_constants_for_sufficient_statistics,
                                                   get_sufficient_statistics,
                                                   get_algorithm_for_learning,
                                                   save_data)


class TestTraining(unittest.TestCase):

    def setUp(self):
        self.path_of_experiment = '/home/michael/Desktop/JMLR_New/Experiments/image_processing'
        self.image_path = '/home/michael/Desktop/Experiments/Images/'
        self.dummy_savings_path = self.path_of_experiment + '/dummy_data/'

    def test_create_folder(self):
        create_folder_for_storing_data(path_of_experiment=self.path_of_experiment)

    def test_get_number_of_datapoints(self):
        # Check that we have exactly four data sets.
        num_data = get_number_of_datapoints()
        self.assertIsInstance(num_data, dict)
        self.assertTrue('prior' in num_data.keys())
        self.assertTrue('train' in num_data.keys())
        self.assertTrue('test' in num_data.keys())
        self.assertTrue('validation' in num_data.keys())
        self.assertTrue(len(num_data.keys()) == 4)

    def test_parameters_of_estimation(self):
        # Check that all the needed parameters are given, and only those.
        par = get_parameters_of_estimation()
        self.assertIsInstance(par, dict)
        self.assertTrue('quantile_distance' in par.keys())
        self.assertTrue('quantiles' in par.keys())
        self.assertTrue('probabilities' in par.keys())
        self.assertTrue(len(par.keys()) == 3)

    def test_update_parameters(self):
        # Check that all the needed parameters are given, and only those.
        par = get_update_parameters()
        self.assertIsInstance(par, dict)
        self.assertTrue('num_iter_print_update' in par.keys())
        self.assertTrue('with_print' in par.keys())
        self.assertTrue('bins' in par.keys())
        self.assertTrue(len(par.keys()) == 3)

    def test_sampling_parameters(self):
        # Check that all the needed parameters are given, and only those.
        max_number = torch.randint(low=1, high=100, size=(1,)).item()
        par = get_sampling_parameters(maximal_number_of_iterations=max_number)
        self.assertIsInstance(par, dict)
        self.assertTrue('lr' in par.keys())
        self.assertTrue('length_trajectory' in par.keys())
        self.assertTrue('with_restarting' in par.keys())
        self.assertTrue('restart_probability' in par.keys())
        self.assertTrue('num_samples' in par.keys())
        self.assertTrue('num_iter_burnin' in par.keys())
        self.assertTrue(len(par.keys()) == 6)
        # Check that restart_probability is set correctly.
        self.assertTrue(par['restart_probability'] == 1/max_number)

    def test_fitting_parameters(self):
        # Check that all the needed parameters are given, and only those.
        max_number = torch.randint(low=1, high=100, size=(1,)).item()
        par = get_fitting_parameters(maximal_number_of_iterations=max_number)
        self.assertIsInstance(par, dict)
        self.assertTrue('lr' in par.keys())
        self.assertTrue('length_trajectory' in par.keys())
        self.assertTrue('restart_probability' in par.keys())
        self.assertTrue('n_max' in par.keys())
        self.assertTrue('num_iter_update_stepsize' in par.keys())
        self.assertTrue('factor_stepsize_update' in par.keys())
        self.assertTrue(len(par.keys()) == 6)
        # Check that restart_probability is set correctly.
        self.assertTrue(par['restart_probability'] == 1/max_number)

    def test_init_parameters(self):
        # Check that all the needed parameters are given, and only those.
        par = get_initialization_parameters()
        self.assertIsInstance(par, dict)
        self.assertTrue('lr' in par.keys())
        self.assertTrue('num_iter_max' in par.keys())
        self.assertTrue('num_iter_print_update' in par.keys())
        self.assertTrue('num_iter_update_stepsize' in par.keys())
        self.assertTrue('with_print' in par.keys())
        self.assertTrue(len(par.keys()) == 5)

    def test_constraint_parameters(self):
        # Check that all the needed parameters are given, and only those.
        max_number = torch.randint(low=1, high=100, size=(1,)).item()
        par = get_constraint_parameters(number_of_training_iterations=max_number)
        self.assertIsInstance(par, dict)
        self.assertTrue('describing_property' in par.keys())
        self.assertTrue('num_iter_update_constraint' in par.keys())
        self.assertTrue(len(par.keys()) == 2)

    def test_pac_parameters(self):
        # Check that all the needed parameters are given, and only those.
        par = get_pac_bayes_parameters(sufficient_statistics=lambda x: None)  # Dummy
        self.assertIsInstance(par, dict)
        self.assertTrue('sufficient_statistics' in par.keys())
        self.assertTrue('natural_parameters' in par.keys())
        self.assertTrue('covering_number' in par.keys())
        self.assertTrue('epsilon' in par.keys())
        self.assertTrue('n_max' in par.keys())
        self.assertTrue(len(par.keys()) == 5)

    def test_describing_property(self):
        # We need to have three functions in the end.
        reduction_property, convergence_risk_constraint, empirical_second_moment = get_describing_property()
        self.assertIsInstance(reduction_property, Callable)
        self.assertIsInstance(convergence_risk_constraint, Callable)
        self.assertIsInstance(empirical_second_moment, Callable)

    def test_get_dimension(self):
        # Check that dimension gets calculated correctly based on images.
        h, w = get_image_height_and_width()
        dim = get_dimension_of_optimization_variable()
        self.assertIsInstance(dim, int)
        self.assertEqual(dim, h * w)

    def test_initial_states(self):
        # We need two initial states, as Nesterov needs one dimension more.
        init_base, init_lear = get_initial_states()
        self.assertIsInstance(init_base, torch.Tensor)
        self.assertIsInstance(init_lear, torch.Tensor)
        self.assertTrue(torch.equal(init_base[1:], init_lear))

    def test_get_baseline_algorithm(self):

        def dummy_function(x):
            return torch.linalg.norm(x) ** 2

        baseline = get_baseline_algorithm(loss_function_of_algorithm=dummy_function, smoothness_parameter=1)
        self.assertIsInstance(baseline, OptimizationAlgorithm)
        self.assertIsInstance(baseline.implementation, NesterovAcceleratedGradient)

    def test_get_constraint(self):

        def dummy_function(x, parameter):
            return parameter['scale'] * torch.linalg.norm(x)

        parameters = {'prior': [{'scale': torch.rand(size=(1,)).item()} for _ in range(0)],
                      'train': [{'scale': torch.rand(size=(1,)).item()} for _ in range(0)],
                      'test': [{'scale': torch.rand(size=(1,)).item()} for _ in range(0)],
                      'validation': [{'scale': torch.rand(size=(1,)).item()} for _ in range(3)]}

        loss_functions = create_parametric_loss_functions_from_parameters(template_loss_function=dummy_function,
                                                                          parameters=parameters)
        # Check that it indeed creates a Constraint-object.
        constraint = get_constraint(loss_functions_for_constraint=loss_functions['validation'])
        self.assertIsInstance(constraint, Constraint)

    def test_get_sufficient_statistics(self):

        def dummy_function(x):
            return torch.linalg.norm(x) ** 2

        # Check that we get a callable, and actually call it.
        sufficient_statistics = get_sufficient_statistics(constants=torch.tensor(1))
        self.assertIsInstance(sufficient_statistics, Callable)
        baseline = get_baseline_algorithm(loss_function_of_algorithm=dummy_function, smoothness_parameter=1)
        baseline.n_max = 1
        loss_function = LossFunction(dummy_function)
        sufficient_statistics(optimization_algorithm=baseline, loss_function=loss_function,
                              probability=torch.tensor(0.))

    def test_compute_constants_for_sufficient_statistics(self):
        factor = 0.2
        exponent = 1.
        number_of_loss_functions = 10
        loss_functions = [LossFunction(function=lambda x: (torch.linalg.norm(x)+2)**2)
                          for _ in range(number_of_loss_functions)]
        constant = compute_constants_for_sufficient_statistics(loss_functions_for_training=loss_functions)
        # Note that we have to divide by N (number of loss_functions) twice: one for the computation of the mean, the
        # other time for the factor 1/N in the PAC-bound. Here, the first 1/N is skipped, as we use N-times the same
        # function x -> (||x|| + 2)**2. Since the initial state is zero, the norm vanishes, and it remains 2**2 = 4.
        self.assertTrue(torch.allclose(constant, torch.tensor(factor * 4 ** exponent) ** 2 / len(loss_functions)))

    @unittest.skip("Tests whole training. Typically, takes too long.")
    def test_get_algorithm_for_learning(self):
        number_data = {'prior': torch.randint(low=1, high=10, size=(1,)).item(),
                       'train': torch.randint(low=1, high=10, size=(1,)).item(),
                       'test': torch.randint(low=1, high=10, size=(1,)).item(),
                       'validation': torch.randint(low=1, high=10, size=(1,)).item()}
        parameters, loss_function_of_algorithm, smoothness = get_data(
            number_of_datapoints_per_dataset=number_data, path_to_images=self.image_path, device='cpu')
        loss_functions = create_parametric_loss_functions_from_parameters(
            template_loss_function=loss_function_of_algorithm, parameters=parameters)
        algo = get_algorithm_for_learning(loss_functions=loss_functions)
        self.assertIsInstance(algo, PacBayesOptimizationAlgorithm)
        self.assertIsInstance(algo.implementation, ConvNet)

    @unittest.skip("Tests whole training. Typically, takes too long.")
    def test_create_loss_functions_from_parameters(self):
        number_data = {'prior': torch.randint(low=1, high=10, size=(1,)).item(),
                       'train': torch.randint(low=1, high=10, size=(1,)).item(),
                       'test': torch.randint(low=1, high=10, size=(1,)).item(),
                       'validation': torch.randint(low=1, high=10, size=(1,)).item()}
        parameters, loss_function_of_algorithm, smoothness = get_data(
            number_of_datapoints_per_dataset=number_data, path_to_images=self.image_path, device='cpu')
        loss_functions = create_parametric_loss_functions_from_parameters(
            template_loss_function=loss_function_of_algorithm, parameters=parameters)
        self.assertIsInstance(loss_functions, dict)
        self.assertTrue(len(loss_functions.keys()) == len(number_data.keys()))
        for name in number_data.keys():
            self.assertEqual(len(loss_functions[name]), number_data[name])

    def test_save_data(self):
        # Dummy data saved into dummy path
        save_data(savings_path=self.dummy_savings_path,
                  smoothness_parameter=0.,
                  pac_bound=np.array(0.),
                  initialization_baseline_algorithm=np.empty(1),
                  initialization_learned_algorithm=np.empty(1),
                  number_of_iterations=10,
                  parameters={},
                  best_sample={},
                  samples_prior=[])

    @unittest.skip("Tests whole training. Typically, takes too long.")
    def test_training(self):
        set_up_and_train_algorithm(path_to_images=self.image_path, path_of_experiment=self.path_of_experiment)
