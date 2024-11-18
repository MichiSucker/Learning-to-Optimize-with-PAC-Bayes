import torch
import io
import sys
import copy
import unittest
from algorithms.dummy import Dummy
from classes.Constraint.class_Constraint import Constraint
from classes.LossFunction.class_LossFunction import LossFunction
from classes.OptimizationAlgorithm.derived_classes.subclass_ParametricOptimizationAlgorithm import (
    ParametricOptimizationAlgorithm, losses_are_invalid)
from classes.Helpers.class_ConstraintChecker import ConstraintChecker
from classes.Helpers.class_TrainingAssistant import TrainingAssistant
from classes.Helpers.class_TrajectoryRandomizer import TrajectoryRandomizer


class TestFitOfParametricOptimizationAlgorithm(unittest.TestCase):

    def setUp(self):
        def dummy_function(x):
            return 0.5 * torch.linalg.norm(x) ** 2

        self.dummy_function = dummy_function
        self.dim = torch.randint(low=1, high=1000, size=(1,)).item()
        self.length_state = 1  # Take one, because it has to be compatible with Dummy()
        self.initial_state = torch.randn(size=(self.length_state, self.dim))
        self.current_state = self.initial_state.clone()
        self.loss_function = LossFunction(function=dummy_function)
        self.optimization_algorithm = ParametricOptimizationAlgorithm(implementation=Dummy(),
                                                                      initial_state=self.initial_state,
                                                                      loss_function=self.loss_function)

    def test_creation(self):
        self.assertIsInstance(self.optimization_algorithm, ParametricOptimizationAlgorithm)

    def test_fit(self):
        # This is a weak test: We only check whether the hyperparameters have been changed.
        # (we do not really know more here; only during evaluation do we see whether training was successful).

        # Initialize setting
        fitting_parameters = {'restart_probability': 0.5, 'length_trajectory': 1, 'n_max': 100,
                              'num_iter_update_stepsize': 5, 'factor_stepsize_update': 0.5, 'lr': 1e-4}
        constraint_parameters = {'num_iter_update_constraint': 5}
        update_parameters = {'with_print': True, 'num_iter_print_update': 10, 'bins': []}
        loss_functions = [LossFunction(self.dummy_function) for _ in range(10)]
        constraint = Constraint(function=lambda x: True)
        old_hyperparameters = copy.deepcopy(self.optimization_algorithm.implementation.state_dict())
        self.optimization_algorithm.set_constraint(constraint)

        # Check that an output was created.
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        self.optimization_algorithm.fit(loss_functions=loss_functions,
                                        fitting_parameters=fitting_parameters,
                                        constraint_parameters=constraint_parameters,
                                        update_parameters=update_parameters)
        sys.stdout = sys.__stdout__
        self.assertTrue(len(capturedOutput.getvalue()) > 0)

        # Check that the hyperparameters got changed, and that the internal states got reset correctly.
        self.assertNotEqual(old_hyperparameters, self.optimization_algorithm.implementation.state_dict())
        self.assertTrue(torch.equal(self.optimization_algorithm.current_state,
                                    self.optimization_algorithm.initial_state))
        self.assertEqual(self.optimization_algorithm.iteration_counter, 0)

        # Also check that, typically, there was no nan or inf.
        self.assertFalse(torch.isnan(self.optimization_algorithm.implementation.state_dict()['scale']) or
                         torch.isinf(self.optimization_algorithm.implementation.state_dict()['scale']))

    def test_initialize_helpers_for_training(self):
        # Initialize setting
        fitting_parameters = {'restart_probability': 0.5, 'length_trajectory': 1, 'n_max': 100,
                              'num_iter_update_stepsize': 5, 'factor_stepsize_update': 0.5, 'lr': 1e-4}
        constraint_parameters = {'num_iter_update_constraint': 5}
        update_parameters = {'with_print': True, 'num_iter_print_update': 10, 'bins': []}

        # Get helpers
        optimizer, training_assistant, trajectory_randomizer, constraint_checker = (
            self.optimization_algorithm.initialize_helpers_for_training(
                fitting_parameters=fitting_parameters,
                constraint_parameters=constraint_parameters,
                update_parameters=update_parameters))
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertIsInstance(training_assistant, TrainingAssistant)
        self.assertIsInstance(trajectory_randomizer, TrajectoryRandomizer)
        self.assertIsInstance(constraint_checker, ConstraintChecker)

    def test_update_hyperparameters(self):
        # Note that this is a weak test: We only check whether the hyperparameters did change.

        # Initialize setting.
        trajectory_randomizer = TrajectoryRandomizer(should_restart=True, restart_probability=1.,
                                                     length_partial_trajectory=1)
        training_assistant = TrainingAssistant(
            printing_enabled=True,
            print_update_every=10,
            maximal_number_of_iterations=100,
            update_stepsize_every=10,
            factor_update_stepsize=0.5
        )
        loss_functions = [LossFunction(self.dummy_function) for _ in range(10)]
        old_hyperparameters = [p.clone() for p in self.optimization_algorithm.implementation.parameters()
                               if p.requires_grad]
        optimizer = torch.optim.Adam(self.optimization_algorithm.implementation.parameters(), lr=1e-4)

        # Perform update and make sure that the hyperparameter do change through that.
        self.optimization_algorithm.update_hyperparameters(optimizer=optimizer,
                                                           trajectory_randomizer=trajectory_randomizer,
                                                           loss_functions=loss_functions,
                                                           training_assistant=training_assistant)
        new_hyperparameters = [p.clone() for p in self.optimization_algorithm.implementation.parameters()
                               if p.requires_grad]
        self.assertNotEqual(old_hyperparameters, new_hyperparameters)

    def test_restart_algorithm_with_new_loss(self):
        # Initialize setting.
        restart_probability = 0.65
        trajectory_randomizer = TrajectoryRandomizer(should_restart=True,
                                                     restart_probability=restart_probability,
                                                     length_partial_trajectory=1)
        self.optimization_algorithm.set_iteration_counter(10)
        loss_functions = [LossFunction(self.dummy_function) for _ in range(10)]
        old_loss_function = self.optimization_algorithm.loss_function
        self.optimization_algorithm.set_current_state(torch.randn(size=self.optimization_algorithm.initial_state.shape))

        # Check where to start next 'cycle'. Here, the algorithm should be reset to its initial state, and it should
        # have a new loss-function.
        self.optimization_algorithm.determine_next_starting_point(trajectory_randomizer, loss_functions=loss_functions)
        self.assertFalse(trajectory_randomizer.should_restart)
        self.assertEqual(self.optimization_algorithm.iteration_counter, 0)
        self.assertTrue(torch.equal(self.optimization_algorithm.current_state,
                                    self.optimization_algorithm.initial_state))
        self.assertNotEqual(old_loss_function, self.optimization_algorithm.loss_function)
        self.assertTrue(self.optimization_algorithm.loss_function in loss_functions)

    def test_continue_from_current_state(self):
        # Initialize setting.
        restart_probability = 0.65
        trajectory_randomizer = TrajectoryRandomizer(should_restart=True,
                                                     restart_probability=restart_probability,
                                                     length_partial_trajectory=1)
        loss_functions = [LossFunction(self.dummy_function) for _ in range(10)]
        trajectory_randomizer.set_variable__should_restart__to(False)
        current_loss_function = self.optimization_algorithm.loss_function
        current_state = self.optimization_algorithm.current_state.clone()
        self.optimization_algorithm.current_state.requires_grad = True
        self.optimization_algorithm.set_iteration_counter(10)

        # Check where to start next 'cycle'. Here, the algorithm should continue from the current state with the same
        # loss-function.
        self.optimization_algorithm.determine_next_starting_point(trajectory_randomizer, loss_functions=loss_functions)
        self.assertFalse(self.optimization_algorithm.current_state.requires_grad)
        self.assertEqual(self.optimization_algorithm.iteration_counter, 10)
        self.assertTrue(torch.equal(self.optimization_algorithm.current_state, current_state))
        self.assertEqual(current_loss_function, self.optimization_algorithm.loss_function)

    def test_restart_with_new_loss(self):
        # Initialize setting.
        self.optimization_algorithm.set_iteration_counter(10)
        loss_functions = [LossFunction(self.dummy_function) for _ in range(10)]
        old_loss_function = self.optimization_algorithm.loss_function

        # Check that algorithm gets reset correctly.
        self.optimization_algorithm.restart_with_new_loss(loss_functions)
        self.assertEqual(self.optimization_algorithm.iteration_counter, 0)
        self.assertNotEqual(old_loss_function, self.optimization_algorithm.loss_function)
        self.assertTrue(self.optimization_algorithm.loss_function in loss_functions)

    def test_detach_current_state_from_computational_graph(self):
        # Initialize setting.
        self.optimization_algorithm.current_state.requires_grad = True
        current_state = self.optimization_algorithm.current_state.clone()
        self.assertTrue(self.optimization_algorithm.current_state.requires_grad)

        # Check that current state gets detached from computational graph.
        self.optimization_algorithm.detach_current_state_from_computational_graph()
        self.assertFalse(self.optimization_algorithm.current_state.requires_grad)
        self.assertTrue(torch.equal(current_state, self.optimization_algorithm.current_state))

    def test_compute_ratio_of_losses(self):
        # Initialize setting.
        predicted_iterates = [torch.tensor(1.), torch.tensor(2.), torch.tensor(3.), torch.tensor(4.), torch.tensor(5.)]
        self.optimization_algorithm.set_loss_function(lambda x: x)

        # Check that losses do take the correct ratios here.
        ratio_of_losses = self.optimization_algorithm.compute_ratio_of_losses(predicted_iterates)
        self.assertTrue(len(ratio_of_losses) == len(predicted_iterates) - 1)
        self.assertEqual(ratio_of_losses, [2./1., 3./2., 4./3., 5./4.])


class TestHelper(unittest.TestCase):

    def test_losses_are_invalid(self):
        self.assertFalse(losses_are_invalid([1., 2., 3.]))
        self.assertTrue(losses_are_invalid([]))
        self.assertTrue(losses_are_invalid([1., None]))
        self.assertTrue(losses_are_invalid([1., torch.inf]))
