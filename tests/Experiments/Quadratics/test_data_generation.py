import unittest
from typing import Callable

import torch

from experiments.quadratics.data_generation import (get_distribution_of_strong_convexity_parameter,
                                                    get_distribution_of_smoothness_parameter,
                                                    get_distribution_of_right_hand_side,
                                                    check_and_extract_number_of_datapoints,
                                                    create_parameter,
                                                    get_loss_function_of_algorithm,
                                                    get_parameters,
                                                    get_data,
                                                    get_values_of_diagonal)


class TestDataGeneration(unittest.TestCase):

    def test_get_distribution_of_strong_convexity_parameter(self):
        distribution, mu_min = get_distribution_of_strong_convexity_parameter()
        self.assertIsInstance(distribution, torch.distributions.uniform.Uniform)
        self.assertIsInstance(mu_min, torch.Tensor)
        self.assertTrue(torch.equal(mu_min, torch.tensor(1e-3)))
        self.assertTrue(torch.eq(distribution.low, torch.tensor(1e-3)))
        self.assertTrue(torch.eq(distribution.high, torch.tensor(5e-3)))

    def test_get_distribution_of_smoothness_parameter(self):
        distribution, L_max = get_distribution_of_smoothness_parameter()
        self.assertIsInstance(distribution, torch.distributions.uniform.Uniform)
        self.assertIsInstance(L_max, torch.Tensor)
        self.assertTrue(torch.equal(L_max, torch.tensor(5e2)))
        self.assertTrue(torch.eq(distribution.low, torch.tensor(1e2)))
        self.assertTrue(torch.eq(distribution.high, torch.tensor(5e2)))

    def test_get_distribution_of_right_hand_side(self):
        distribution, dim = get_distribution_of_right_hand_side()
        self.assertIsInstance(distribution, torch.distributions.multivariate_normal.MultivariateNormal)
        self.assertEqual(distribution.covariance_matrix.shape, torch.Size((200, 200)))
        self.assertEqual(dim, 200)

    def test_get_values_of_diagonal(self):
        min_value = torch.randint(low=1, high=10, size=(1,)).item()
        max_value = torch.randint(low=11, high=100, size=(1,)).item()
        number_of_values = torch.randint(low=2, high=100, size=(1,)).item()
        values_diagonal = get_values_of_diagonal(
            min_value=min_value, max_value=max_value, number_of_values=number_of_values
        )
        self.assertTrue(torch.equal(values_diagonal, torch.linspace(min_value, max_value, number_of_values)))

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

    def test_create_parameter(self):
        dim = torch.randint(low=1, high=100, size=(1,)).item()
        diagonal = torch.randn((dim,))
        rhs = torch.randn((dim,))
        parameter = create_parameter(diagonal=diagonal, right_hand_side=rhs)
        self.assertIsInstance(parameter, dict)
        self.assertTrue('A' in list(parameter.keys()))
        self.assertTrue('b' in list(parameter.keys()))
        self.assertTrue('optimal_loss' in list(parameter.keys()))

    def test_get_loss_function_of_algorithm(self):
        loss_function = get_loss_function_of_algorithm()
        self.assertIsInstance(loss_function, Callable)

    def test_get_parameters(self):
        number_data = {'prior': torch.randint(low=1, high=100, size=(1,)).item(),
                       'train': torch.randint(low=1, high=100, size=(1,)).item(),
                       'test': torch.randint(low=1, high=100, size=(1,)).item(),
                       'validation': torch.randint(low=1, high=100, size=(1,)).item()}
        parameters, mu_min, L_max, dim = get_parameters(number_of_datapoints_per_dataset=number_data)
        self.assertIsInstance(parameters, dict)
        self.assertIsInstance(mu_min, torch.Tensor)
        self.assertIsInstance(L_max, torch.Tensor)
        self.assertTrue(mu_min.item() < L_max.item())
        self.assertTrue(torch.equal(mu_min, torch.tensor(1e-3)))
        self.assertTrue(torch.equal(L_max, torch.tensor(5e2)))
        self.assertEqual(dim, 200)
        self.assertTrue('prior' in list(parameters.keys()))
        self.assertTrue('train' in list(parameters.keys()))
        self.assertTrue('test' in list(parameters.keys()))
        self.assertTrue('validation' in list(parameters.keys()))
        for name in ['prior', 'train', 'test', 'validation']:
            self.assertEqual(len(parameters[name]), number_data[name])
            for parameter in parameters[name]:
                self.assertIsInstance(parameters, dict)
                self.assertTrue('A' in list(parameter.keys()))
                self.assertTrue('b' in list(parameter.keys()))
                self.assertTrue('optimal_loss' in list(parameter.keys()))


def test_quadratics_with_variable_curvature_without_rand_perm(self):
    number_data = {'prior': torch.randint(low=1, high=100, size=(1,)).item(),
                   'train': torch.randint(low=1, high=100, size=(1,)).item(),
                   'test': torch.randint(low=1, high=100, size=(1,)).item(),
                   'validation': torch.randint(low=1, high=100, size=(1,)).item()}
    parameters, loss_function, mu_min, L_max, dim = get_data(number_data)
    self.assertIsInstance(parameters, dict)
    self.assertIsInstance(loss_function, Callable)
    self.assertIsInstance(mu_min, torch.Tensor)
    self.assertIsInstance(L_max, torch.Tensor)
    self.assertIsInstance(dim, int)
