import unittest
from types import NoneType
import copy
from classes.OptimizationAlgorithm.derived_classes.subclass_ParametricOptimizationAlgorithm import (ConstraintChecker,
                                                                                                    check_constraint)
import torch
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.Constraint.class_Constraint import Constraint
from classes.LossFunction.class_LossFunction import LossFunction
from algorithms.dummy import Dummy


def dummy_function(x):
    return 0.5 * torch.linalg.norm(x) ** 2


class TestConstraintChecker(unittest.TestCase):

    def setUp(self):
        self.check_constraint_every = 10
        self.there_is_a_constraint = True
        self.constraint_checker = ConstraintChecker(check_constraint_every=self.check_constraint_every,
                                                    there_is_a_constraint=self.there_is_a_constraint)

    def test_creation(self):
        self.assertIsInstance(self.constraint_checker, ConstraintChecker)
        self.assertFalse(self.constraint_checker.found_point_inside_constraint)
        self.assertTrue(isinstance(self.constraint_checker.point_inside_constraint, NoneType))

    def test_should_check_constraint(self):
        i = torch.randint(low=1, high=9, size=(1,)).item() * self.check_constraint_every
        self.assertTrue(self.constraint_checker.should_check_constraint(i))
        i = i-1
        self.assertFalse(self.constraint_checker.should_check_constraint(i))
        self.assertFalse(self.constraint_checker.should_check_constraint(0))

        self.constraint_checker.set_there_is_a_constraint(False)
        i = torch.randint(low=1, high=9, size=(1,)).item() * self.check_constraint_every
        self.assertFalse(self.constraint_checker.should_check_constraint(i))

    def test_set_there_is_a_constraint(self):
        self.assertTrue(self.constraint_checker.there_is_a_constraint)
        self.constraint_checker.set_there_is_a_constraint(False)
        self.assertFalse(self.constraint_checker.there_is_a_constraint)
        self.constraint_checker.set_there_is_a_constraint(True)
        self.assertTrue(self.constraint_checker.there_is_a_constraint)

    def test_set_check_constraint_every(self):
        self.assertEqual(self.constraint_checker.check_constraint_every, self.check_constraint_every)
        random_number = torch.randint(low=1, high=9, size=(1,)).item()
        self.constraint_checker.set_check_constraint_every(new_number=random_number)
        self.assertEqual(self.constraint_checker.check_constraint_every, random_number)

    def test_check_constraint(self):
        constraint_always_true = Constraint(function=lambda x: True)
        dummy_algorithm = OptimizationAlgorithm(implementation=Dummy(),
                                                initial_state=torch.randn((3, 2)),
                                                loss_function=LossFunction(function=dummy_function),
                                                constraint=constraint_always_true)
        self.assertTrue(check_constraint(optimization_algorithm=dummy_algorithm))

    def test_update_point_inside_constraint_or_reject(self):
        # Test case 1: Constraint is true
        constraint_always_true = Constraint(function=lambda x: True)
        dummy_algorithm = OptimizationAlgorithm(implementation=Dummy(),
                                                initial_state=torch.randn((3, 2)),
                                                loss_function=LossFunction(function=dummy_function),
                                                constraint=constraint_always_true)
        old_hyperparameters = copy.deepcopy(dummy_algorithm.implementation.state_dict())
        dummy_algorithm.implementation.state_dict()['scale'] += 0.1
        self.constraint_checker.update_point_inside_constraint_or_reject(dummy_algorithm)
        self.assertNotEqual(old_hyperparameters, dummy_algorithm.implementation.state_dict())
        self.assertEqual(self.constraint_checker.point_inside_constraint, dummy_algorithm.implementation.state_dict())
        self.assertTrue(self.constraint_checker.found_point_inside_constraint)

        # Test case 2: Constraint is false and one already found a point inside the constraint
        constraint_always_false = Constraint(function=lambda x: False)
        dummy_algorithm = OptimizationAlgorithm(implementation=Dummy(),
                                                initial_state=torch.randn((3, 2)),
                                                loss_function=LossFunction(function=dummy_function),
                                                constraint=constraint_always_false)
        old_hyperparameters = copy.deepcopy(dummy_algorithm.implementation.state_dict())
        self.constraint_checker.point_inside_constraint = old_hyperparameters
        self.constraint_checker.found_point_inside_constraint = True
        dummy_algorithm.implementation.state_dict()['scale'] += 0.1
        self.assertNotEqual(old_hyperparameters, dummy_algorithm.implementation.state_dict())
        new_hyperparameters = copy.deepcopy(dummy_algorithm.implementation.state_dict())
        self.constraint_checker.update_point_inside_constraint_or_reject(dummy_algorithm)
        self.assertEqual(old_hyperparameters, dummy_algorithm.implementation.state_dict())
        self.assertNotEqual(dummy_algorithm.implementation.state_dict(), new_hyperparameters)
        self.assertTrue(self.constraint_checker.found_point_inside_constraint)

        # Test case 3: Constraint is false and one does not have found point inside constraint already
        old_hyperparameters = copy.deepcopy(dummy_algorithm.implementation.state_dict())
        self.constraint_checker.point_inside_constraint = None
        self.constraint_checker.found_point_inside_constraint = False
        dummy_algorithm.implementation.state_dict()['scale'] += 0.1
        new_hyperparameters = copy.deepcopy(dummy_algorithm.implementation.state_dict())
        self.assertNotEqual(old_hyperparameters, new_hyperparameters)
        self.constraint_checker.update_point_inside_constraint_or_reject(dummy_algorithm)
        self.assertEqual(new_hyperparameters, dummy_algorithm.implementation.state_dict())
        self.assertFalse(self.constraint_checker.found_point_inside_constraint)
        self.assertTrue(isinstance(self.constraint_checker.point_inside_constraint, NoneType))

    def test_final_check(self):
        # Test Case 1: Constraint is satisfied
        constraint_always_true = Constraint(function=lambda x: True)
        dummy_algorithm = OptimizationAlgorithm(implementation=Dummy(),
                                                initial_state=torch.randn((3, 2)),
                                                loss_function=LossFunction(function=dummy_function),
                                                constraint=constraint_always_true)
        hyperparameters = copy.deepcopy(dummy_algorithm.implementation.state_dict())
        self.constraint_checker.final_check(dummy_algorithm)
        self.assertEqual(dummy_algorithm.implementation.state_dict(), hyperparameters)

        # Test Case 2: Constraint is not satisfied, but we have found a point inside the constraint before
        constraint_always_false = Constraint(function=lambda x: False)
        dummy_algorithm = OptimizationAlgorithm(implementation=Dummy(),
                                                initial_state=torch.randn((3, 2)),
                                                loss_function=LossFunction(function=dummy_function),
                                                constraint=constraint_always_false)
        old_hyperparameters = copy.deepcopy(dummy_algorithm.implementation.state_dict())
        self.constraint_checker.point_inside_constraint = old_hyperparameters
        self.constraint_checker.found_point_inside_constraint = True
        dummy_algorithm.implementation.state_dict()['scale'] += 0.1
        current_hyperparameters = copy.deepcopy(dummy_algorithm.implementation.state_dict())
        self.assertNotEqual(old_hyperparameters, current_hyperparameters)
        self.constraint_checker.final_check(dummy_algorithm)
        self.assertEqual(dummy_algorithm.implementation.state_dict(), old_hyperparameters)

        # Test case 3: Constraint is not satisfied, and we have not found any point that does satisfy the constraint
        self.constraint_checker.found_point_inside_constraint = False
        with self.assertRaises(Exception):
            self.constraint_checker.final_check(dummy_algorithm)


