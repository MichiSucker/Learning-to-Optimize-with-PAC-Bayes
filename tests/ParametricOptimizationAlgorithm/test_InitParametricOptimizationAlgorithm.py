import unittest
import io
import sys
from classes.OptimizationAlgorithm.derived_classes.subclass_ParametricOptimizationAlgorithm import (
    ParametricOptimizationAlgorithm, compute_initialization_loss)
from classes.Helpers.class_TrajectoryRandomizer import TrajectoryRandomizer
from classes.Helpers.class_InitializationAssistant import InitializationAssistant
import torch
from algorithms.dummy import Dummy, NonTrainableDummy
from classes.LossFunction.class_LossFunction import LossFunction
import copy


class TestInitParametricOptimizationAlgorithm(unittest.TestCase):

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

    def test_initialize_with_other_algorithm(self):
        other_algorithm = ParametricOptimizationAlgorithm(implementation=NonTrainableDummy(),
                                                          initial_state=self.initial_state,
                                                          loss_function=self.loss_function)
        loss_functions = [LossFunction(self.dummy_function) for _ in range(10)]
        parameters_init = {
            'with_print': True, 'num_iter_max': 100, 'num_iter_update_stepsize': 30, 'num_iter_print_update': 10,
            'lr': 1e-4
        }
        # Make step-size small enough; otherwise the algorithm will directly explode.
        old_hyperparameters = copy.deepcopy(self.optimization_algorithm.implementation.state_dict())

        # Check that there is an output sometimes.
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        self.optimization_algorithm.initialize_with_other_algorithm(other_algorithm=other_algorithm,
                                                                    loss_functions=loss_functions,
                                                                    parameters_of_initialization=parameters_init)
        sys.stdout = sys.__stdout__
        self.assertTrue(len(capturedOutput.getvalue()) > 0)

        # Check that hyperparameters got updated, but the state of the algorithm got reset.
        # Further, make sure that, typically, there is no nan or inf.
        self.assertNotEqual(old_hyperparameters, self.optimization_algorithm.implementation.state_dict())
        self.assertTrue(torch.equal(self.optimization_algorithm.current_state,
                                    self.optimization_algorithm.initial_state))
        self.assertEqual(self.optimization_algorithm.iteration_counter, 0)
        self.assertFalse(torch.isnan(self.optimization_algorithm.implementation.state_dict()['scale']) or
                         torch.isinf(self.optimization_algorithm.implementation.state_dict()['scale']))

    def test_initialize_helpers_for_initialization(self):
        parameters_init = {'with_print': True, 'num_iter_max': 100, 'lr': 1e-4,
                           'num_iter_update_stepsize': 10, 'num_iter_print_update': 10}
        optimizer, initialization_assistant, trajectory_randomizer = (
            self.optimization_algorithm.initialize_helpers_for_initialization(parameters=parameters_init))
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertIsInstance(trajectory_randomizer, TrajectoryRandomizer)
        self.assertIsInstance(initialization_assistant, InitializationAssistant)

    def test_update_initialization_of_hyperparameters(self):
        # Note that this is a weak test! We only check whether the hyperparameters did change.
        trajectory_randomizer = TrajectoryRandomizer(should_restart=True, restart_probability=1.,
                                                     length_partial_trajectory=1)
        initialization_assistant = InitializationAssistant(printing_enabled=True,
                                                           maximal_number_of_iterations=100,
                                                           update_stepsize_every=10,
                                                           print_update_every=10,
                                                           factor_update_stepsize=0.5)
        other_algorithm = copy.deepcopy(self.optimization_algorithm)
        loss_functions = [LossFunction(self.dummy_function) for _ in range(10)]

        # Change hyperparameters of other algorithm. Then, adjust hyperparameters of optimization_algorithm
        # by following other_algorithm.
        other_algorithm.implementation.state_dict()['scale'] -= 0.5
        old_hyperparameters = [p.clone() for p in self.optimization_algorithm.implementation.parameters()
                               if p.requires_grad]
        optimizer = torch.optim.Adam(self.optimization_algorithm.implementation.parameters(), lr=1e-4)
        self.optimization_algorithm.update_initialization_of_hyperparameters(
            optimizer=optimizer,
            other_algorithm=other_algorithm,
            trajectory_randomizer=trajectory_randomizer,
            loss_functions=loss_functions,
            initialization_assistant=initialization_assistant
        )
        new_hyperparameters = [p.clone() for p in self.optimization_algorithm.implementation.parameters()
                               if p.requires_grad]
        self.assertNotEqual(old_hyperparameters, new_hyperparameters)

    def test_restart_both_algorithms(self):
        # Initialization
        restart_probability = 0.65
        trajectory_randomizer = TrajectoryRandomizer(should_restart=True,
                                                     restart_probability=restart_probability,
                                                     length_partial_trajectory=1)
        other_algorithm = copy.deepcopy(self.optimization_algorithm)
        self.optimization_algorithm.set_iteration_counter(10)
        other_algorithm.set_iteration_counter(10)
        loss_functions = [LossFunction(self.dummy_function) for _ in range(10)]
        old_loss_function = self.optimization_algorithm.loss_function
        self.optimization_algorithm.set_current_state(torch.randn(size=self.optimization_algorithm.initial_state.shape))
        other_algorithm.set_current_state(torch.randn(size=self.optimization_algorithm.initial_state.shape))

        # Make sure that the states are not the same.
        self.assertFalse(torch.equal(self.optimization_algorithm.current_state, other_algorithm.current_state))
        self.optimization_algorithm.determine_next_starting_point_for_both_algorithms(
            trajectory_randomizer=trajectory_randomizer, other_algorithm=other_algorithm, loss_functions=loss_functions)

        # Actual tests for behavior.
        self.assertFalse(trajectory_randomizer.should_restart)
        self.assertEqual(self.optimization_algorithm.iteration_counter, 0)
        self.assertEqual(other_algorithm.iteration_counter, 0)
        self.assertTrue(torch.equal(self.optimization_algorithm.current_state,
                                    self.optimization_algorithm.initial_state))
        self.assertTrue(torch.equal(other_algorithm.current_state, self.optimization_algorithm.current_state))
        self.assertNotEqual(old_loss_function, self.optimization_algorithm.loss_function)
        self.assertTrue(self.optimization_algorithm.loss_function in loss_functions)
        self.assertEqual(self.optimization_algorithm.loss_function, other_algorithm.loss_function)

    def test_do_not_restart_the_algorithms(self):
        # Initialization
        restart_probability = 0.65
        trajectory_randomizer = TrajectoryRandomizer(should_restart=True,
                                                     restart_probability=restart_probability,
                                                     length_partial_trajectory=1)
        other_algorithm = copy.deepcopy(self.optimization_algorithm)
        self.optimization_algorithm.set_iteration_counter(10)
        other_algorithm.set_iteration_counter(10)
        loss_functions = [LossFunction(self.dummy_function) for _ in range(10)]
        self.optimization_algorithm.set_current_state(torch.randn(size=self.optimization_algorithm.initial_state.shape))
        other_algorithm.set_current_state(torch.randn(size=self.optimization_algorithm.initial_state.shape))

        # Make sure that the algorithm does not get restarted.
        trajectory_randomizer.set_variable__should_restart__to(False)
        current_loss_function = self.optimization_algorithm.loss_function
        current_state = self.optimization_algorithm.current_state.clone()
        self.optimization_algorithm.set_iteration_counter(10)
        other_algorithm = copy.deepcopy(self.optimization_algorithm)
        self.optimization_algorithm.current_state.requires_grad = True
        self.optimization_algorithm.determine_next_starting_point_for_both_algorithms(
            trajectory_randomizer=trajectory_randomizer, other_algorithm=other_algorithm, loss_functions=loss_functions)

        # Actual tests for behavior.
        self.assertFalse(self.optimization_algorithm.current_state.requires_grad)
        self.assertEqual(self.optimization_algorithm.iteration_counter, 10)
        self.assertTrue(torch.equal(self.optimization_algorithm.current_state, current_state))
        self.assertEqual(current_loss_function, self.optimization_algorithm.loss_function)


class TestHelper(unittest.TestCase):

    def test_compute_initialization_loss(self):
        # Only compute loss if the same number of iterations got performed. Otherwise, there was an error.
        with self.assertRaises(ValueError):
            iterates_1 = [torch.randn(size=(3,)) for _ in range(3)]
            iterates_2 = [torch.randn(size=(3,)) for _ in range(2)]
            compute_initialization_loss(iterates_1, iterates_2)

        # Check that a loss got computed.
        iterates_1 = [torch.randn(size=(3,)) for _ in range(3)]
        iterates_2 = [torch.randn(size=(3,)) for _ in range(3)]
        loss = compute_initialization_loss(iterates_1, iterates_2)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss >= 0)
