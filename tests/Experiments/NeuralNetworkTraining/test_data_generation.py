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
        # Check that loss of neural network is MSE
        c = get_loss_of_neural_network()
        x, y = torch.rand(size=(100,)), torch.rand(size=(100,))
        self.assertEqual(c(x, y), torch.nn.MSELoss()(x, y))

    def test_get_distribution_of_datapoints(self):
        # Check that data points are sampled uniformly
        d = get_distribution_of_datapoints()
        self.assertIsInstance(d, torch.distributions.uniform.Uniform)
        self.assertEqual(d.low, -2)
        self.assertEqual(d.high, 2)

    def test_get_distribution_of_coefficients(self):
        # Check that coefficients are sampled uniformly
        d = get_distribution_of_coefficients()
        self.assertIsInstance(d, torch.distributions.uniform.Uniform)
        self.assertEqual(d.low, -5)
        self.assertEqual(d.high, 5)

    def test_get_powers_of_polynomials(self):
        # Check that powers of polynomials are specified correctly.
        powers = get_powers_of_polynomials()
        self.assertTrue(torch.equal(powers, torch.arange(6)))
        self.assertTrue(torch.max(powers) == 5)
        self.assertTrue(torch.min(powers) == 0)

    def test_get_observations_for_x_values(self):
        # Check that you get an sorted tensor of x-values.
        number_of_samples = torch.randint(low=1, high=100, size=(1,)).item()
        d = get_distribution_of_datapoints()
        xes = get_observations_for_x_values(number_of_samples=number_of_samples, distribution_x_values=d)
        self.assertEqual(xes.shape, torch.Size((number_of_samples, 1)))
        self.assertTrue(torch.equal(xes, torch.sort(xes)[0]))

    def test_get_coefficients(self):
        # Check that you get a number for each power of the polynomial
        powers = get_powers_of_polynomials()
        c = get_coefficients(get_distribution_of_coefficients(), maximal_degree=torch.max(powers).item())
        self.assertEqual(len(c), len(powers))

    def test_get_ground_truth_values(self):
        # Check that ground-truth values are computed correctly.
        number_of_samples = torch.randint(low=1, high=100, size=(1,)).item()
        x_values = get_observations_for_x_values(number_of_samples, get_distribution_of_datapoints())
        powers = get_powers_of_polynomials()
        coefficients = get_coefficients(get_distribution_of_coefficients(), maximal_degree=torch.max(powers).item())
        gt_values = get_ground_truth_values(x_values=x_values, coefficients=coefficients, powers=powers)
        self.assertEqual(gt_values.shape, torch.Size((number_of_samples, 1)))
        for i, x in enumerate(x_values):
            self.assertEqual(torch.sum(torch.stack([coefficients[k] * x ** powers[k] for k in range(len(powers))])),
                             gt_values[i])

    def test_get_y_values(self):
        # Check that y-values are ground-truth + standard-normal noise.
        number_of_samples = torch.randint(low=100, high=250, size=(1,)).item()
        x_values = get_observations_for_x_values(number_of_samples, get_distribution_of_datapoints())
        powers = get_powers_of_polynomials()
        coefficients = get_coefficients(get_distribution_of_coefficients(), maximal_degree=torch.max(powers).item())
        gt_values = get_ground_truth_values(x_values=x_values, coefficients=coefficients, powers=powers)
        y_values = get_y_values(gt_values)
        self.assertEqual(gt_values.shape, y_values.shape)
        self.assertTrue(torch.mean(gt_values - y_values) < 1)   # This could be made more precise.

    def test_create_parameter(self):
        # Initialize
        number_of_samples = torch.randint(low=100, high=250, size=(1,)).item()
        x_values = get_observations_for_x_values(number_of_samples, get_distribution_of_datapoints())
        powers = get_powers_of_polynomials()
        coefficients = get_coefficients(get_distribution_of_coefficients(), maximal_degree=torch.max(powers).item())
        gt_values = get_ground_truth_values(x_values=x_values, coefficients=coefficients, powers=powers)
        y_values = get_y_values(gt_values)

        # Check that the parameter has the needed keys.
        p = create_parameter(x_values=x_values, y_values=y_values, ground_truth_values=gt_values,
                             coefficients=coefficients)
        self.assertIsInstance(p, dict)
        self.assertTrue('x_values' in list(p.keys()))
        self.assertTrue('y_values' in list(p.keys()))
        self.assertTrue('ground_truth_values' in list(p.keys()))
        self.assertTrue('coefficients' in list(p.keys()))
        self.assertTrue('optimal_loss' in list(p.keys()))
        self.assertTrue(len(p.keys()) == 5)

    def test_check_and_extract_number_of_datapoints(self):
        # Check that it raises an error if at least one of the data sets is not specified.
        # And check that the extracted numbers are correct.
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
        # This is a weak test: Just check that you get a callable function.
        criterion = get_loss_of_neural_network()

        def dummy_neural_network(x):
            return torch.tensor(1.)

        loss = get_loss_of_algorithm(dummy_neural_network, criterion)
        self.assertIsInstance(loss, Callable)

    def test_get_parameters(self):
        # We only have to check that we get a dictionary, and for each entry in the dictionary we have a list of
        # parameters.
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

        # Check that we get a loss-function for the algorithm, a loss-function for the neural network, and a dictionary
        # of data sets.
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
