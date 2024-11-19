import unittest
from typing import Callable
import torch
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import \
    PacBayesOptimizationAlgorithm
from classes.LossFunction.class_LossFunction import LossFunction
from exponential_family.describing_property.reduction_property import (instantiate_reduction_property_with,
                                                                       compute_loss_at_beginning_and_end,
                                                                       store_current_loss_function_state_and_iteration_counter,
                                                                       reset_loss_function_state_and_iteration_counter)
from algorithms.dummy import Dummy


class TestReductionProperty(unittest.TestCase):

    def setUp(self):

        def dummy_function(x):
            return 0.5 * torch.linalg.norm(x) ** 2

        self.dummy_function = dummy_function
        dim = torch.randint(low=1, high=1000, size=(1,)).item()
        length_state = 1  # Take one, because it has to be compatible with Dummy()
        self.initial_state = torch.randn(size=(length_state, dim))
        self.loss_function = LossFunction(function=dummy_function)
        self.n_max = torch.randint(low=2, high=25, size=(1,)).item()
        self.pac_parameters = {'sufficient_statistics': None,
                               'natural_parameters': None,
                               'covering_number': None,
                               'epsilon': None,
                               'n_max': self.n_max}
        self.pac_algorithm = PacBayesOptimizationAlgorithm(implementation=Dummy(),
                                                           initial_state=self.initial_state,
                                                           loss_function=self.loss_function,
                                                           pac_parameters=self.pac_parameters)

    def test_store_current_loss_function_state_and_iteration_counter(self):
        current_state = torch.rand(size=self.initial_state.size())
        iteration_counter = torch.randint(low=10, high=20, size=(1,)).item()
        self.pac_algorithm.set_current_state(current_state)
        self.pac_algorithm.set_iteration_counter(n=iteration_counter)

        # Check that we get the current loss-function, the current state, and the current iteration counter.
        current_loss_function, current_state, current_iteration_counter = (
            store_current_loss_function_state_and_iteration_counter(self.pac_algorithm))
        self.assertEqual(self.loss_function, current_loss_function)
        self.assertTrue(torch.equal(self.pac_algorithm.current_state, current_state))
        self.assertEqual(self.pac_algorithm.iteration_counter, current_iteration_counter)

    def test_reset_loss_function_state_and_iteration_counter(self):
        state = torch.rand(size=self.initial_state.size())
        iteration_counter = torch.randint(low=10, high=20, size=(1,)).item()
        loss_function = LossFunction(self.dummy_function)

        # Check that resetting works correctly.
        reset_loss_function_state_and_iteration_counter(optimization_algorithm=self.pac_algorithm,
                                                        loss_function=loss_function,
                                                        state=state,
                                                        iteration_counter=iteration_counter)
        self.assertEqual(self.pac_algorithm.loss_function, loss_function)
        self.assertTrue(torch.equal(self.pac_algorithm.current_state, state))
        self.assertEqual(self.pac_algorithm.iteration_counter, iteration_counter)

    def test_compute_loss_at_beginning_and_end(self):
        loss_at_beginning, loss_at_end = compute_loss_at_beginning_and_end(optimization_algorithm=self.pac_algorithm)
        self.assertEqual(loss_at_beginning, self.pac_algorithm.loss_function(self.pac_algorithm.initial_state[-1]))
        self.assertEqual(loss_at_end, self.pac_algorithm.loss_function(self.pac_algorithm.current_state[-1]))
        self.assertEqual(self.pac_algorithm.iteration_counter, self.n_max)

    def test_empirical_second_moment(self):

        # Instantiate reduction property with random values.
        factor = torch.rand((1,)).item()
        exponent = (torch.randint(low=0, high=3, size=(1,)) + torch.rand((1,))).item()
        reduction_property, convergence_risk_constraint, empirical_second_moment = (
            instantiate_reduction_property_with(factor=factor, exponent=exponent))

        # Check computation of empirical second moment.
        list_of_loss_functions = [lambda x: torch.tensor(i) for i in range(10)]
        point = torch.rand((1,))
        self.assertIsInstance(empirical_second_moment(list_of_loss_functions=list_of_loss_functions, point=point),
                              torch.Tensor)
        self.assertEqual(empirical_second_moment(list_of_loss_functions=list_of_loss_functions, point=point),
                         torch.mean(torch.stack([(factor * loss_function(point) ** exponent) ** 2
                                                 for loss_function in list_of_loss_functions]))
                         )

    def test_reduction_property(self):

        # Instantiate reduction property with random values.
        factor = torch.rand((1,)).item()
        exponent = (torch.randint(low=0, high=3, size=(1,)) + torch.rand((1,))).item()
        reduction_property, convergence_risk_constraint, empirical_second_moment = (
            instantiate_reduction_property_with(factor=factor, exponent=exponent))
        self.assertIsInstance(reduction_property, Callable)
        self.assertIsInstance(convergence_risk_constraint, Callable)
        self.assertIsInstance(empirical_second_moment, Callable)

        # Check that it evaluates correctly.
        loss_at_beginning = torch.rand((1,))
        self.assertIsInstance(convergence_risk_constraint(1, 1), bool)
        self.assertTrue(convergence_risk_constraint(loss_at_beginning=loss_at_beginning,
                                                    loss_at_end=factor*loss_at_beginning**exponent))
        self.assertFalse(convergence_risk_constraint(loss_at_beginning=loss_at_beginning,
                                                     loss_at_end=2 * factor * loss_at_beginning ** exponent))

    def test_instantiate_reduction_property_with(self):

        # Instantiate reduction property with random values.
        factor = torch.rand((1,)).item()
        exponent = (torch.randint(low=0, high=3, size=(1,)) + torch.rand((1,))).item()
        reduction_property, convergence_risk_constraint, empirical_second_moment = (
            instantiate_reduction_property_with(factor=factor, exponent=exponent))

        # Check that it yields a boolean-value, and that the algorithm gets reset correctly.
        current_state = torch.rand(size=self.initial_state.size())
        iteration_counter = torch.randint(low=10, high=20, size=(1,)).item()
        self.pac_algorithm.set_current_state(current_state)
        self.pac_algorithm.set_iteration_counter(n=iteration_counter)
        loss_function_to_test = LossFunction(function=self.dummy_function)
        self.assertNotEqual(loss_function_to_test, self.loss_function)

        # Perform test.
        self.assertIsInstance(reduction_property(loss_function_to_test, self.pac_algorithm), bool)
        self.assertEqual(self.pac_algorithm.loss_function, self.loss_function)
        self.assertTrue(torch.equal(self.pac_algorithm.current_state, current_state))
        self.assertEqual(self.pac_algorithm.iteration_counter, iteration_counter)
