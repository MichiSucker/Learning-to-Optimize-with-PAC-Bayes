import unittest
from typing import Callable

import torch.nn
import torch

from experiments.nn_training.data_generation import (get_loss_of_neural_network,
                                                     get_distribution_of_datapoints,
                                                     get_distribution_of_coefficients,
                                                     get_powers_of_polynomials,
                                                     get_observations_for_x_values,
                                                     get_coefficients,
                                                     get_ground_truth_values,
                                                     get_y_values, create_parameter,
                                                     check_and_extract_number_of_datapoints,
                                                     get_loss_of_algorithm,
                                                     get_parameters,
                                                     get_data)


class TestDataGeneration(unittest.TestCase):

    def test_get_criterion(self):
        c = get_loss_of_neural_network()
        x, y = torch.rand(size=(100,)), torch.rand(size=(100,))
        self.assertEqual(c(x, y), torch.nn.MSELoss()(x, y))

    def test_get_distribution_of_datapoints(self):
        d = get_distribution_of_datapoints()
        self.assertIsInstance(d, torch.distributions.uniform.Uniform)
        self.assertEqual(d.low, -2)
        self.assertEqual(d.high, 2)

    def test_get_distribution_of_coefficients(self):
        d = get_distribution_of_coefficients()
        self.assertIsInstance(d, torch.distributions.uniform.Uniform)
        self.assertEqual(d.low, -5)
        self.assertEqual(d.high, 5)

    def test_get_powers_of_polynomials(self):
        powers = get_powers_of_polynomials()
        self.assertTrue(torch.equal(powers, torch.arange(6)))
        self.assertTrue(torch.max(powers) == 5)
        self.assertTrue(torch.min(powers) == 0)

    def test_get_observations_for_x_values(self):
        number_of_samples = torch.randint(low=1, high=100, size=(1,)).item()
        d = get_distribution_of_datapoints()
        xes = get_observations_for_x_values(number_of_samples=number_of_samples, distribution_x_values=d)
        self.assertEqual(xes.shape, torch.Size((number_of_samples, 1)))

    def test_get_coefficients(self):
        d = get_distribution_of_coefficients()
        powers = get_powers_of_polynomials()
        c = get_coefficients(get_distribution_of_coefficients(), maximal_degree=torch.max(powers))
        self.assertEqual(len(c), len(powers))

    def test_get_ground_truth_values(self):
        number_of_samples = torch.randint(low=1, high=100, size=(1,)).item()
        x_values = get_observations_for_x_values(number_of_samples, get_distribution_of_datapoints())
        powers = get_powers_of_polynomials()
        coefficients = get_coefficients(get_distribution_of_coefficients(), maximal_degree=torch.max(powers))
        gt_values = get_ground_truth_values(x_values=x_values, coefficients=coefficients, powers=powers)
        self.assertEqual(gt_values.shape, torch.Size((number_of_samples, 1)))
        for i, x in enumerate(x_values):
            self.assertEqual(torch.sum(torch.stack([coefficients[k] * x ** powers[k] for k in range(len(powers))])),
                             gt_values[i])

    def test_get_y_values(self):
        number_of_samples = torch.randint(low=100, high=250, size=(1,)).item()
        x_values = get_observations_for_x_values(number_of_samples, get_distribution_of_datapoints())
        powers = get_powers_of_polynomials()
        coefficients = get_coefficients(get_distribution_of_coefficients(), maximal_degree=torch.max(powers))
        gt_values = get_ground_truth_values(x_values=x_values, coefficients=coefficients, powers=powers)
        y_values = get_y_values(gt_values)
        self.assertEqual(gt_values.shape, y_values.shape)
        self.assertTrue(torch.mean(gt_values - y_values) < 1)   # This could be made more precise.

    def test_get_parameter(self):
        number_of_samples = torch.randint(low=100, high=250, size=(1,)).item()
        x_values = get_observations_for_x_values(number_of_samples, get_distribution_of_datapoints())
        powers = get_powers_of_polynomials()
        coefficients = get_coefficients(get_distribution_of_coefficients(), maximal_degree=torch.max(powers))
        gt_values = get_ground_truth_values(x_values=x_values, coefficients=coefficients, powers=powers)
        y_values = get_y_values(gt_values)
        p = create_parameter(x_values=x_values, y_values=y_values, ground_truth_values=gt_values,
                             coefficients=coefficients)
        self.assertIsInstance(p, dict)
        self.assertTrue('x_values' in list(p.keys()))
        self.assertTrue('y_values' in list(p.keys()))
        self.assertTrue('ground_truth_values' in list(p.keys()))
        self.assertTrue('coefficients' in list(p.keys()))
        self.assertTrue('optimal_loss' in list(p.keys()))

    def test_check_and_extract_number_of_datapoints(self):
        with self.assertRaises(ValueError):
            check_and_extract_number_of_datapoints({})
        with self.assertRaises(ValueError):
            check_and_extract_number_of_datapoints({'prior': 1})
        with self.assertRaises(ValueError):
            check_and_extract_number_of_datapoints({'prior': 1, 'train': 1})
        with self.assertRaises(ValueError):
            check_and_extract_number_of_datapoints({'prior': 1, 'train': 1, 'test': 1})
        number_data = {'prior': torch.randint(low=1, high=100, size=(1,)).item(),
             'train': torch.randint(low=1, high=100, size=(1,)).item(),
             'test': torch.randint(low=1, high=100, size=(1,)).item(),
             'validation': torch.randint(low=1, high=100, size=(1,)).item()}
        n_prior, n_train, n_test, n_val = check_and_extract_number_of_datapoints(number_data)
        self.assertEqual(n_prior, number_data['prior'])
        self.assertEqual(n_train, number_data['train'])
        self.assertEqual(n_test, number_data['test'])
        self.assertEqual(n_val, number_data['validation'])

    def test_get_loss_of_algorithm(self):
        criterion = get_loss_of_neural_network()

        def dummy_neural_network(x):
            return torch.tensor(1.)

        loss = get_loss_of_algorithm(dummy_neural_network, criterion)
        self.assertIsInstance(loss, Callable)

    def test_get_parameters(self):
        number_data = {'prior': torch.randint(low=1, high=100, size=(1,)).item(),
                       'train': torch.randint(low=1, high=100, size=(1,)).item(),
                       'test': torch.randint(low=1, high=100, size=(1,)).item(),
                       'validation': torch.randint(low=1, high=100, size=(1,)).item()}
        parameters = get_parameters(number_data)
        self.assertIsInstance(parameters, dict)
        self.assertTrue('prior' in list(parameters.keys()))
        self.assertTrue('train' in list(parameters.keys()))
        self.assertTrue('test' in list(parameters.keys()))
        self.assertTrue('validation' in list(parameters.keys()))

        for key in parameters.keys():
            self.assertIsInstance(parameters[key], list)
            for p in parameters[key]:
                self.assertIsInstance(p, dict)
                self.assertTrue('x_values' in list(p.keys()))
                self.assertTrue('y_values' in list(p.keys()))
                self.assertTrue('ground_truth_values' in list(p.keys()))
                self.assertTrue('coefficients' in list(p.keys()))
                self.assertTrue('optimal_loss' in list(p.keys()))

    def test_get_data(self):

        def dummy_neural_network(x):
            return torch.tensor(1.)

        number_data = {'prior': torch.randint(low=1, high=100, size=(1,)).item(),
                       'train': torch.randint(low=1, high=100, size=(1,)).item(),
                       'test': torch.randint(low=1, high=100, size=(1,)).item(),
                       'validation': torch.randint(low=1, high=100, size=(1,)).item()}
        loss, criterion, parameters = get_data(neural_network=dummy_neural_network,
                                               number_of_datapoints_per_dataset=number_data)
        self.assertIsInstance(loss, Callable)
        self.assertIsInstance(criterion, Callable)
        x, y = torch.rand(size=(100,)), torch.rand(size=(100,))
        self.assertEqual(criterion(x, y), torch.nn.MSELoss()(x, y))
        self.assertIsInstance(parameters, dict)
        self.assertTrue('prior' in list(parameters.keys()))
        self.assertTrue('train' in list(parameters.keys()))
        self.assertTrue('test' in list(parameters.keys()))
        self.assertTrue('validation' in list(parameters.keys()))

        for key in parameters.keys():
            self.assertIsInstance(parameters[key], list)
            for p in parameters[key]:
                self.assertIsInstance(p, dict)
                self.assertTrue('x_values' in list(p.keys()))
                self.assertTrue('y_values' in list(p.keys()))
                self.assertTrue('ground_truth_values' in list(p.keys()))
                self.assertTrue('coefficients' in list(p.keys()))
                self.assertTrue('optimal_loss' in list(p.keys()))
