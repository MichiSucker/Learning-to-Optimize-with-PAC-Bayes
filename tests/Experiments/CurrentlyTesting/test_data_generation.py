import unittest
from typing import Callable

import torch

from experiments.lasso.data_generation import (get_loss_function_of_algorithm,
                                               get_dimensions,
                                               get_distribution_of_right_hand_side,
                                               get_matrix_for_smooth_part,
                                               calculate_smoothness_parameter,
                                               get_distribution_of_regularization_parameter,
                                               check_and_extract_number_of_datapoints,
                                               get_parameters,
                                               create_parameter,
                                               get_data)


class TestDataGeneration(unittest.TestCase):

    def test_get_dimensions(self):
        # Check that first return value is dimension of right-hand side.
        x, y = get_dimensions()
        self.assertIsInstance(x, int)
        self.assertIsInstance(y, int)
        self.assertTrue(x < y)

    def test_get_distribution_of_right_hand_side(self):
        # Check that right-hand side gets sampled from a Normal distribution.
        dist = get_distribution_of_right_hand_side()
        self.assertIsInstance(dist, torch.distributions.multivariate_normal.MultivariateNormal)

    def test_get_distribution_of_regularization_parameter(self):
        # Check that regularization parameter gets sampled uniformly.
        dist = get_distribution_of_regularization_parameter()
        self.assertIsInstance(dist, torch.distributions.uniform.Uniform)

    def test_get_matrix_for_smooth_part(self):
        # Check that the matrix for the smooth part only has small enough values.
        A = get_matrix_for_smooth_part()
        dim_b, dim_x = get_dimensions()
        self.assertIsInstance(A, torch.Tensor)
        self.assertEqual(A.shape, torch.Size((dim_b, dim_x)))
        self.assertTrue(torch.max(torch.abs(A)) <= 0.5)

    def test_get_smoothness_parameter(self):
        # Check that the smoothness parameter is the largest eigenvalue of A.T @ A.
        # Note that this one is always >= 0.
        A = get_matrix_for_smooth_part()
        eigenvalues = torch.linalg.eigvalsh(A.T @ A)
        p = calculate_smoothness_parameter(A)
        self.assertIsInstance(p, torch.Tensor)
        self.assertEqual(p, eigenvalues[-1])

    def test_get_loss_function_of_algorithm(self):

        # Check that we get three callables.
        # And check that they add-up correctly.
        # Note that this automatically also checks calling the function, and that the return value is a tensor.
        f, g, h = get_loss_function_of_algorithm()
        self.assertIsInstance(f, Callable)
        self.assertIsInstance(g, Callable)
        self.assertIsInstance(h, Callable)

        dim = torch.randint(low=1, high=10, size=(1,)).item()
        A = torch.randn((dim, dim))
        b = torch.randn((dim,))
        mu = torch.rand((1,))
        parameter = {'A': A, 'b': b, 'mu': mu}
        x = torch.randn((dim, ))
        self.assertTrue(torch.allclose(f(x, parameter), g(x, parameter) + h(x, parameter)))

    def test_check_and_extract_number_of_datapoints(self):
        # Check that we raise an error if not all data sets are specified.
        # And check that the extraction works correctly.
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

    def test_create_parameter(self):
        # Check that a parameter is a dictionary that has the keys 'A', 'b', and 'mu'.
        # and that they get stored correctly.
        A = get_matrix_for_smooth_part()
        rhs = get_distribution_of_right_hand_side().sample((1,))
        mu = get_distribution_of_regularization_parameter().sample((1,))
        p = create_parameter(matrix=A, right_hand_side=rhs, regularization_parameter=mu)
        self.assertIsInstance(p, dict)
        self.assertTrue('A' in p.keys())
        self.assertTrue('b' in p.keys())
        self.assertTrue('mu' in p.keys())
        self.assertEqual(len(p.keys()), 3)
        self.assertTrue(torch.equal(p['A'], A))
        self.assertTrue(torch.equal(p['b'], rhs))
        self.assertTrue(torch.equal(p['mu'], mu))

    def test_get_parameters(self):
        # Check that parameters is a dictionary with the keys specified below.
        # And check that each entry is a list with the correct number of data points.
        number_data = {'prior': torch.randint(low=1, high=10, size=(1,)).item(),
                       'train': torch.randint(low=1, high=10, size=(1,)).item(),
                       'test': torch.randint(low=1, high=10, size=(1,)).item(),
                       'validation': torch.randint(low=1, high=10, size=(1,)).item()}
        A = get_matrix_for_smooth_part()
        parameters = get_parameters(matrix=A, number_of_datapoints_per_dataset=number_data)
        self.assertIsInstance(parameters, dict)
        self.assertTrue('prior' in parameters.keys())
        self.assertTrue('train' in parameters.keys())
        self.assertTrue('test' in parameters.keys())
        self.assertTrue('validation' in parameters.keys())
        self.assertEqual(len(parameters.keys()), 4)
        for name in parameters.keys():
            self.assertIsInstance(parameters[name], list)
            self.assertEqual(len(parameters[name]), number_data[name])

    def test_get_data(self):
        # Check that data-creation runs through, and returns the parameters, the loss-function and its decomposition,
        # and the smoothness parameter.
        numb_data = {'prior': torch.randint(low=1, high=100, size=(1,)).item(),
                     'train': torch.randint(low=1, high=100, size=(1,)).item(),
                     'test': torch.randint(low=1, high=100, size=(1,)).item(),
                     'validation': torch.randint(low=1, high=100, size=(1,)).item()}
        parameters, loss_function_of_algorithm, smooth_part, nonsmooth_part, smoothness_parameter = get_data(numb_data)
        self.assertIsInstance(parameters, dict)
        self.assertIsInstance(loss_function_of_algorithm, Callable)
        self.assertIsInstance(smooth_part, Callable)
        self.assertIsInstance(nonsmooth_part, Callable)
        self.assertIsInstance(smoothness_parameter.item(), float)
