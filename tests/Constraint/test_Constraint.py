import unittest
from classes.Constraint.class_Constraint import Constraint, create_list_of_constraints_from_functions
import torch
from algorithms.dummy import Dummy
from classes.LossFunction.class_LossFunction import LossFunction
from classes.OptimizationAlgorithm.derived_classes.subclass_ParametricOptimizationAlgorithm import (
    ParametricOptimizationAlgorithm)


class TestConstraint(unittest.TestCase):

    def setUp(self):

        def dummy_constraint(optimization_algorithm):
            if torch.all(optimization_algorithm.current_state > 0):
                return True
            else:
                return False

        def dummy_function(x):
            return 0.5 * torch.linalg.norm(x) ** 2

        self.dummy_constraint = dummy_constraint
        self.dummy_function = dummy_function
        self.constraint = Constraint(dummy_constraint)

    def test_creation(self):
        self.assertIsInstance(self.constraint, Constraint)

    def test_call_constraint(self):
        dim = torch.randint(low=1, high=1000, size=(1,)).item()
        length_state = 1
        initial_state = torch.randn(size=(length_state, dim))
        loss_function = LossFunction(function=self.dummy_function)
        optimization_algorithm = ParametricOptimizationAlgorithm(implementation=Dummy(),
                                                                 initial_state=initial_state,
                                                                 loss_function=loss_function)
        self.assertIsInstance(self.constraint(optimization_algorithm), bool)
        optimization_algorithm.set_current_state(torch.ones(initial_state.shape))
        self.assertTrue(self.constraint(optimization_algorithm))
        optimization_algorithm.set_current_state(torch.zeros(initial_state.shape))
        self.assertFalse(self.constraint(optimization_algorithm))


class TestHelper(unittest.TestCase):

    def setUp(self):

        def dummy_function(x):
            return 0.5 * torch.linalg.norm(x) ** 2
        self.dummy_function = dummy_function

    def test_create_list_of_constraints_from_functions(self):

        def describing_property(function, optimization_algorithm):
            return function(optimization_algorithm.current_iterate) >= 0

        list_of_functions = [lambda x: torch.linalg.norm(x)**2, lambda y: -torch.linalg.norm(y)**2]
        list_of_constraints = create_list_of_constraints_from_functions(describing_property=describing_property,
                                                                        list_of_functions=list_of_functions)
        self.assertEqual(len(list_of_constraints), len(list_of_functions))
        for constraint in list_of_constraints:
            self.assertIsInstance(constraint, Constraint)

        self.dim = torch.randint(low=1, high=1000, size=(1,)).item()
        self.length_state = 1
        self.initial_state = torch.randn(size=(self.length_state, self.dim))
        self.current_state = self.initial_state.clone()
        self.loss_function = LossFunction(function=self.dummy_function)
        parametric_optimization_algorithm = ParametricOptimizationAlgorithm(implementation=Dummy(),
                                                                            initial_state=self.initial_state,
                                                                            loss_function=self.loss_function)

        self.assertTrue(list_of_constraints[0](parametric_optimization_algorithm))
        self.assertFalse(list_of_constraints[1](parametric_optimization_algorithm))