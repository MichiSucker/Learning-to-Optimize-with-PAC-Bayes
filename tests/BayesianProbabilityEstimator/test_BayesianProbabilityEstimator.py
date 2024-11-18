import unittest
import torch
from scipy.stats import beta
from classes.Constraint.class_BayesianProbabilityEstimator import (BayesianProbabilityEstimator,
                                                                   sample_and_evaluate_random_constraint,
                                                                   update_parameters_and_uncertainty,
                                                                   estimation_should_be_stopped,
                                                                   check_quantiles,
                                                                   check_quantile_distance,
                                                                   check_probabilities)


class TestBayesianProbabilityEstimator(unittest.TestCase):

    def setUp(self):
        self.list_of_constraints = []
        self.parameters_estimation = {'quantile_distance': 0.05,
                                      'quantiles': (0.01, 0.99),
                                      'probabilities': (0.85, 0.95)}
        self.probabilistic_constraint = BayesianProbabilityEstimator(
            list_of_constraints=self.list_of_constraints, parameters_of_estimation=self.parameters_estimation)

    def test_creation(self):
        self.assertIsInstance(self.probabilistic_constraint, BayesianProbabilityEstimator)

    def test_get_parameters_of_estimation(self):
        parameters_estimation = self.probabilistic_constraint.get_parameters_of_estimation()
        self.assertIsInstance(parameters_estimation, dict)
        self.assertTrue('quantile_distance' in list(parameters_estimation.keys()))
        self.assertTrue('quantiles' in list(parameters_estimation.keys()))
        self.assertTrue('probabilities' in list(parameters_estimation.keys()))
        self.assertEqual(len(parameters_estimation.keys()), 3)

    def test_get_list_of_constraints(self):
        self.assertEqual(self.list_of_constraints, self.probabilistic_constraint.get_list_of_constraints())

    def test_set_list_of_constraints(self):
        true_probability = torch.rand((1,)).item()
        new_list_of_constraints = [(lambda x: True) if torch.rand((1,)).item() <= true_probability else (lambda x: True)
                                   for _ in range(10)]
        self.assertNotEqual(self.probabilistic_constraint.get_list_of_constraints(), new_list_of_constraints)
        self.probabilistic_constraint.set_list_of_constraints(new_list_of_constraints)
        self.assertEqual(self.probabilistic_constraint.get_list_of_constraints(), new_list_of_constraints)

    def test_set_parameters_of_estimation(self):
        new_parameters = {}
        # Cannot set empty dict as new parameters:
        # It has to have the fields 'quantile_distance', 'quantiles', 'probabilities'.
        # Furthermore, the corresponding values have to have specific values.
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
        # Quantile distance is a number between 0 and 1.
        with self.assertRaises(ValueError):
            self.probabilistic_constraint.set_quantile_distance(1.1)
        random_number = torch.rand(1).item()
        self.probabilistic_constraint.set_quantile_distance(random_number)
        self.assertTrue(self.probabilistic_constraint.get_parameters_of_estimation()['quantile_distance']
                        == random_number)

    def test_get_quantile_distance(self):
        self.assertIsInstance(self.probabilistic_constraint.get_quantile_distance(), float)

    def test_get_quantiles(self):
        self.assertEqual(self.parameters_estimation['quantiles'], self.probabilistic_constraint.get_quantiles())

    def test_get_probabilities(self):
        self.assertEqual(self.parameters_estimation['probabilities'], self.probabilistic_constraint.get_probabilities())

    def test_set_quantiles(self):
        # Quantiles have to lie in (0,1)
        with self.assertRaises(ValueError):
            self.probabilistic_constraint.set_quantiles((1.1, 0.9))
        self.probabilistic_constraint.set_quantiles((0.011, 0.932))
        self.assertTrue(self.probabilistic_constraint.get_parameters_of_estimation()['quantiles'] == (0.011, 0.932))

    def test_set_probabilities(self):
        # Probabilities have to lie in (0,1)
        with self.assertRaises(ValueError):
            self.probabilistic_constraint.set_probabilities((1.1, 0.9))
        self.probabilistic_constraint.set_probabilities((0.11, 0.32))
        self.assertTrue(self.probabilistic_constraint.get_parameters_of_estimation()['probabilities'] == (0.11, 0.32))

    # @unittest.skipIf(condition=(TESTING_LEVEL == 'SKIP_EXPENSIVE_TESTS'),
    #                  reason='Too expensive to test all the time.')
    def test_estimate_probability(self):
        # If the true probability does lie within the desired range, the algorithm should be able to provide a
        # reasonable estimate for it
        true_probability = torch.distributions.uniform.Uniform(0.1, 0.9).sample((1,)).item()
        new_list_of_constraints = [(lambda x: True)
                                   if torch.rand((1,)).item() <= true_probability else (lambda x: False)
                                   for _ in range(1000)]
        self.probabilistic_constraint.set_list_of_constraints(new_list_of_constraints)
        # True probability actually does lie within the desired range
        self.probabilistic_constraint.set_probabilities((true_probability-0.1, true_probability+0.1))
        # Most of the mass (95%) do lie within a distance of 0.05
        self.probabilistic_constraint.set_quantiles((0.025, 0.975))
        quantile_distance_to_test = 0.05
        self.probabilistic_constraint.set_quantile_distance(quantile_distance_to_test)
        # Note that here, we can just call the method as we want, because the constraints just evaluate to 'True' or
        # 'False' directly
        posterior_mean, current_lower_quantile, current_upper_quantile, n_iterates = (
            self.probabilistic_constraint.estimate_probability('some_input'))
        # The algorithm should have contracted strong enough, and the posterior mean should lie within the
        # specified range. Note that the lower test can actually fail, but this is unlikely.
        self.assertTrue(current_upper_quantile - current_lower_quantile < quantile_distance_to_test)
        self.assertTrue(true_probability-0.1 <= posterior_mean <= true_probability+0.1)


class TestHelpers(unittest.TestCase):

    def test_check_quantile_distance(self):
        # Check that quantile distance lies in (0,1)
        self.assertFalse(check_quantile_distance(1.1))
        self.assertTrue(check_quantile_distance(0.1))

    def test_check_quantiles(self):
        # Check that quantiles are ordered correctly and that they lie in the [0,1].
        self.assertFalse(check_quantiles((1.0, 0.1)))
        self.assertFalse(check_quantiles((1.1, 1.2)))
        self.assertFalse(check_quantiles((0.1, 1.1)))
        self.assertTrue(check_quantiles((0.1, 0.5)))

    def test_check_probabilities(self):
        # Check that probabilities are ordered correctly and that they lie in the [0,1].
        self.assertFalse(check_probabilities((1.0, 0.1)))
        self.assertFalse(check_probabilities((1.1, 1.2)))
        self.assertFalse(check_probabilities((0.1, 1.1)))
        self.assertTrue(check_probabilities((0.1, 0.5)))

    def test_sample_and_evaluate_random_constraint(self):
        # Raise an error if there are no constraints to sample from.
        with self.assertRaises(ValueError):
            sample_and_evaluate_random_constraint(input_to_constraint=torch.Tensor([1.]), list_of_constraints=[])

        list_of_constraints = [lambda x: True]
        self.assertIsInstance(sample_and_evaluate_random_constraint(input_to_constraint=torch.Tensor([1.]),
                                                                    list_of_constraints=list_of_constraints),
                              int)

    def test_update_believe_and_bounds(self):
        a, b = 1, 1
        prior = beta(a=a, b=b)
        lower_quantile, upper_quantile = 0.1, 0.9
        current_upper_quantile, current_lower_quantile = prior.ppf(upper_quantile), prior.ppf(lower_quantile)
        initial_quantile_distance = current_upper_quantile - current_lower_quantile
        p = 0.75
        for _ in range(150):
            result = int(torch.rand(1) <= p)
            a_new, b_new, current_upper_quantile, current_lower_quantile = update_parameters_and_uncertainty(
                result=result, a=a, b=b, upper_quantile=upper_quantile, lower_quantile=lower_quantile)
            # Check that at least one of the two values gets updated each time
            self.assertTrue((a < a_new) or (b < b_new))
            a, b = a_new, b_new
        # Quantile distance after estimation should be smaller than at the beginning.
        # Note that this can actually fail. However, this is very unlikely.
        self.assertTrue((current_upper_quantile - current_lower_quantile) < initial_quantile_distance)
        # Posterior mean should be close to true value of p. Again, this can fail, but it is unlikely.
        self.assertTrue(abs(p - (a/(a + b))) < 0.1)

    def test_estimation_should_be_stopped(self):
        # Stop estimation, if estimate is quite tight already, but far away from the desired interval
        self.assertTrue(estimation_should_be_stopped(current_upper_quantile=0.3, current_lower_quantile=0.2,
                                                     current_posterior_mean=0.25, desired_upper_probability=0.85,
                                                     desired_lower_probability=0.75, desired_quantile_distance=0.2))
        self.assertTrue(estimation_should_be_stopped(current_upper_quantile=0.85, current_lower_quantile=0.75,
                                                     current_posterior_mean=0.8, desired_upper_probability=0.3,
                                                     desired_lower_probability=0.2, desired_quantile_distance=0.2))

        # Do not stop estimation, if estimate is quite tight, but close to desired interval.
        self.assertFalse(estimation_should_be_stopped(current_upper_quantile=0.85, current_lower_quantile=0.75,
                                                      current_posterior_mean=0.8, desired_upper_probability=0.95,
                                                      desired_lower_probability=0.85, desired_quantile_distance=0.2))
        self.assertFalse(estimation_should_be_stopped(current_upper_quantile=0.3, current_lower_quantile=0.2,
                                                      current_posterior_mean=0.25, desired_upper_probability=0.2,
                                                      desired_lower_probability=0.15, desired_quantile_distance=0.1))

        # Do not stop estimation, if estimate is not tight yet.
        self.assertFalse(estimation_should_be_stopped(current_upper_quantile=0.6, current_lower_quantile=0.1,
                                                      current_posterior_mean=0.45, desired_upper_probability=1.0,
                                                      desired_lower_probability=0.95, desired_quantile_distance=0.05))
