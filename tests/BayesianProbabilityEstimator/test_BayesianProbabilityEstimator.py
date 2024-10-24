import unittest
import torch
from scipy.stats import beta
from classes.Constraint.class_BayesianProbabilityEstimator import (BayesianProbabilityEstimator,
                                                                   sample_and_evaluate_random_constraint,
                                                                   update_parameters_and_uncertainty,
                                                                   estimation_should_be_stopped)


class TestProbabilisticConstraint(unittest.TestCase):

    def setUp(self):
        self.list_of_constraints = []
        self.parameters_estimation = {'quantile_distance': 0.05,
                                      'quantiles': (0.01, 0.99),
                                      'probabilities': (0.85, 0.95)}
        self.probabilistic_constraint = BayesianProbabilityEstimator(list_of_constraints=self.list_of_constraints,
                                                                     parameters_of_estimation=self.parameters_estimation)

    def test_creation(self):
        self.assertIsInstance(self.probabilistic_constraint, BayesianProbabilityEstimator)

    def test_get_parameters_of_estimation(self):

        self.assertIsInstance(self.probabilistic_constraint.get_parameters_of_estimation(), dict)

        parameters_estimation = self.probabilistic_constraint.get_parameters_of_estimation()
        self.assertTrue('quantile_distance' in list(parameters_estimation.keys()))
        self.assertTrue('quantiles' in list(parameters_estimation.keys()))
        self.assertTrue('probabilities' in list(parameters_estimation.keys()))

    def test_get_quantile_distance(self):
        self.assertIsInstance(self.probabilistic_constraint.get_quantile_distance(), float)

    def test_get_quantiles(self):
        self.assertIsInstance(self.probabilistic_constraint.get_quantiles(), tuple)
        quantiles = self.probabilistic_constraint.get_quantiles()
        self.assertTrue(quantiles[0] <= quantiles[1])

    def test_get_probabilities(self):
        self.assertIsInstance(self.probabilistic_constraint.get_probabilities(), tuple)
        probabilities = self.probabilistic_constraint.get_probabilities()
        self.assertTrue(probabilities[0] <= probabilities[1])

    def test_set_parameters_of_estimation(self):
        new_parameters = {}
        with self.assertRaises(ValueError):
            self.probabilistic_constraint.set_parameters_of_estimation(new_parameters)
        new_parameters['quantile_distance'] = 1.1

        with self.assertRaises(ValueError):
            self.probabilistic_constraint.set_parameters_of_estimation(new_parameters)
        new_parameters['quantiles'] = (2, 1)

        with self.assertRaises(ValueError):
            self.probabilistic_constraint.set_parameters_of_estimation(new_parameters)
        new_parameters['probabilities'] = (9, 7)

        with self.assertRaises(ValueError):
            self.probabilistic_constraint.set_parameters_of_estimation(new_parameters)
        new_parameters['quantile_distance'] = 0.1

        with self.assertRaises(ValueError):
            self.probabilistic_constraint.set_parameters_of_estimation(new_parameters)
        new_parameters['quantiles'] = (1, 2)

        with self.assertRaises(ValueError):
            self.probabilistic_constraint.set_parameters_of_estimation(new_parameters)
        new_parameters['quantiles'] = (0.1, 0.2)

        with self.assertRaises(ValueError):
            self.probabilistic_constraint.set_parameters_of_estimation(new_parameters)
        new_parameters['probabilities'] = (7, 9)

        with self.assertRaises(ValueError):
            self.probabilistic_constraint.set_parameters_of_estimation(new_parameters)
        new_parameters['probabilities'] = (0.7, 0.9)

        self.probabilistic_constraint.set_parameters_of_estimation(new_parameters)
        self.assertTrue(self.probabilistic_constraint.get_parameters_of_estimation() == new_parameters)

    def test_set_quantile_distance(self):
        with self.assertRaises(ValueError):
            self.probabilistic_constraint.set_quantile_distance(1.1)
        random_number = torch.rand(1).item()
        self.probabilistic_constraint.set_quantile_distance(random_number)
        self.assertTrue(self.probabilistic_constraint.get_parameters_of_estimation()['quantile_distance']
                        == random_number)

    def test_set_quantiles(self):
        with self.assertRaises(ValueError):
            self.probabilistic_constraint.set_quantiles((1.1, 0.9))
        self.probabilistic_constraint.set_quantiles((0.011, 0.932))
        self.assertTrue(self.probabilistic_constraint.get_parameters_of_estimation()['quantiles'] == (0.011, 0.932))

    def test_set_probabilities(self):
        with self.assertRaises(ValueError):
            self.probabilistic_constraint.set_probabilities((1.1, 0.9))
        self.probabilistic_constraint.set_probabilities((0.11, 0.32))
        self.assertTrue(self.probabilistic_constraint.get_parameters_of_estimation()['probabilities'] == (0.11, 0.32))

    def test_sample_and_evaluate_random_constraint(self):
        with self.assertRaises(ValueError):
            sample_and_evaluate_random_constraint(point=torch.Tensor([1.]), list_of_constraints=[])

        list_of_constraints = [lambda x: True]
        self.assertIsInstance(sample_and_evaluate_random_constraint(point=torch.Tensor([1.]),
                                                                    list_of_constraints=list_of_constraints),
                              int)

    def test_update_believe_and_bounds(self):
        a, b = 1, 1
        prior = beta(a=a, b=b)
        lower_quantile, upper_quantile = 0.1, 0.9
        current_upper_quantile, current_lower_quantile = prior.ppf(upper_quantile), prior.ppf(lower_quantile)
        initial_quantile_distance = current_upper_quantile - current_lower_quantile
        p = 0.75
        for _ in range(100):
            result = int(torch.rand(1) <= p)
            a_new, b_new, current_upper_quantile, current_lower_quantile = update_parameters_and_uncertainty(
                result=result, a=a, b=b, upper_quantile=upper_quantile, lower_quantile=lower_quantile)
            self.assertTrue((a < a_new) or (b < b_new))

            a, b = a_new, b_new
        self.assertTrue((current_upper_quantile - current_lower_quantile) < initial_quantile_distance)

    def test_estimation_should_be_stopped(self):
        self.assertTrue(estimation_should_be_stopped(current_upper_quantile=0.3, current_lower_quantile=0.2,
                                                     current_posterior_mean=0.25, desired_upper_probability=0.85,
                                                     desired_lower_probability=0.75, desired_quantile_distance=0.2))

        self.assertTrue(estimation_should_be_stopped(current_upper_quantile=0.85, current_lower_quantile=0.75,
                                                     current_posterior_mean=0.8, desired_upper_probability=0.3,
                                                     desired_lower_probability=0.2, desired_quantile_distance=0.2))

        self.assertFalse(estimation_should_be_stopped(current_upper_quantile=0.85, current_lower_quantile=0.75,
                                                      current_posterior_mean=0.8, desired_upper_probability=0.95,
                                                      desired_lower_probability=0.85, desired_quantile_distance=0.2))

        self.assertFalse(estimation_should_be_stopped(current_upper_quantile=0.3, current_lower_quantile=0.2,
                                                      current_posterior_mean=0.25, desired_upper_probability=0.2,
                                                      desired_lower_probability=0.15, desired_quantile_distance=0.1))

        self.assertFalse(estimation_should_be_stopped(current_upper_quantile=0.6, current_lower_quantile=0.1,
                                                      current_posterior_mean=0.45, desired_upper_probability=1.0,
                                                      desired_lower_probability=0.95, desired_quantile_distance=0.05))

    def test_get_list_of_constraints(self):
        self.assertEqual(self.list_of_constraints, self.probabilistic_constraint.get_list_of_constraints())

    def test_set_list_of_constraints(self):
        true_probability = torch.rand((1,)).item()
        new_list_of_constraints = [(lambda x: True) if torch.rand((1,)).item() <= true_probability else (lambda x: True)
                                   for _ in range(10)]
        self.assertNotEqual(self.probabilistic_constraint.get_list_of_constraints(), new_list_of_constraints)
        self.probabilistic_constraint.set_list_of_constraints(new_list_of_constraints)
        self.assertEqual(self.probabilistic_constraint.get_list_of_constraints(), new_list_of_constraints)

    @unittest.skip("Skip 'test_estimate_probability' because it takes long.")
    def test_estimate_probability(self):
        dummy_point = torch.tensor([1.])
        true_probability = torch.distributions.uniform.Uniform(0.1, 0.9).sample((1,)).item()
        new_list_of_constraints = [(lambda x: True)
                                   if torch.rand((1,)).item() <= true_probability else (lambda x: False)
                                   for _ in range(1000)]
        self.probabilistic_constraint.set_list_of_constraints(new_list_of_constraints)
        self.probabilistic_constraint.set_probabilities((true_probability-0.1, true_probability+0.1))
        self.probabilistic_constraint.set_quantiles((0.025, 0.975))
        quantile_distance_to_test = 0.05
        self.probabilistic_constraint.set_quantile_distance(quantile_distance_to_test)
        posterior_mean, current_lower_quantile, current_upper_quantile, n_iterates = (
            self.probabilistic_constraint.estimate_probability(dummy_point))

        self.assertTrue(current_upper_quantile - current_lower_quantile < quantile_distance_to_test)
        self.assertTrue(true_probability-0.1 <= posterior_mean <= true_probability+0.1)
