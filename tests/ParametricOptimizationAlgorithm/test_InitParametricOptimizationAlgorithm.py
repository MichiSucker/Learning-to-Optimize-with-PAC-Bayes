import unittest
from classes.OptimizationAlgorithm.derived_classes.subclass_ParametricOptimizationAlgorithm import (
    ParametricOptimizationAlgorithm, compute_initialization_loss, TrajectoryRandomizer, InitializationAssistant)
import torch
from algorithms.dummy import Dummy
from classes.LossFunction.class_LossFunction import LossFunction
import copy

def dummy_function(x):
    return 0.5 * torch.linalg.norm(x) ** 2


class TestInitParametricOptimizationAlgorithm(unittest.TestCase):

    def setUp(self):
        self.dim = torch.randint(low=1, high=1000, size=(1,)).item()
        self.length_state = 1  # Take one, because it has to be compatible with Dummy()
        self.initial_state = torch.randn(size=(self.length_state, self.dim))
        self.current_state = self.initial_state.clone()
        self.loss_function = LossFunction(function=dummy_function)
        self.optimization_algorithm = ParametricOptimizationAlgorithm(implementation=Dummy(),
                                                                      initial_state=self.initial_state,
                                                                      loss_function=self.loss_function)

    def test_compute_initialization_loss(self):
        with self.assertRaises(ValueError):
            iterates_1 = [torch.randn(size=(3,)) for _ in range(3)]
            iterates_2 = [torch.randn(size=(3,)) for _ in range(2)]
            compute_initialization_loss(iterates_1, iterates_2)
        iterates_1 = [torch.randn(size=(3,)) for _ in range(3)]
        iterates_2 = [torch.randn(size=(3,)) for _ in range(3)]
        loss = compute_initialization_loss(iterates_1, iterates_2)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss >= 0)

    def test_determine_next_starting_point_for_two_algorithms(self):
        restart_probability = 0.65
        trajectory_randomizer = TrajectoryRandomizer(should_restart=True,
                                                     restart_probability=restart_probability,
                                                     length_partial_trajectory=1)
        other_algorithm = copy.deepcopy(self.optimization_algorithm)
        self.optimization_algorithm.set_iteration_counter(10)
        other_algorithm.set_iteration_counter(10)
        loss_functions = [LossFunction(dummy_function) for _ in range(10)]
        old_loss_function = self.optimization_algorithm.loss_function
        self.optimization_algorithm.set_current_state(torch.randn(size=self.optimization_algorithm.initial_state.shape))
        other_algorithm.set_current_state(torch.randn(size=self.optimization_algorithm.initial_state.shape))
        self.assertFalse(torch.equal(self.optimization_algorithm.current_state, other_algorithm.current_state))
        self.optimization_algorithm.determine_next_starting_point_for_both_algorithms(
            trajectory_randomizer=trajectory_randomizer, other_algorithm=other_algorithm, loss_functions=loss_functions)
        self.assertFalse(trajectory_randomizer.should_restart)
        self.assertEqual(self.optimization_algorithm.iteration_counter, 0)
        self.assertEqual(other_algorithm.iteration_counter, 0)
        self.assertTrue(torch.equal(self.optimization_algorithm.current_state,
                                    self.optimization_algorithm.initial_state))
        self.assertTrue(torch.equal(other_algorithm.current_state, self.optimization_algorithm.current_state))
        self.assertNotEqual(old_loss_function, self.optimization_algorithm.loss_function)
        self.assertTrue(self.optimization_algorithm.loss_function in loss_functions)
        self.assertEqual(self.optimization_algorithm.loss_function, other_algorithm.loss_function)

        trajectory_randomizer.set_should_restart(False)
        current_loss_function = self.optimization_algorithm.loss_function
        current_state = self.optimization_algorithm.current_state.clone()
        self.optimization_algorithm.set_iteration_counter(10)
        other_algorithm = copy.deepcopy(self.optimization_algorithm)
        self.optimization_algorithm.current_state.requires_grad = True
        self.optimization_algorithm.determine_next_starting_point_for_both_algorithms(
            trajectory_randomizer=trajectory_randomizer, other_algorithm=other_algorithm, loss_functions=loss_functions)
        self.assertFalse(self.optimization_algorithm.current_state.requires_grad)
        self.assertEqual(self.optimization_algorithm.iteration_counter, 10)
        self.assertTrue(torch.equal(self.optimization_algorithm.current_state, current_state))
        self.assertEqual(current_loss_function, self.optimization_algorithm.loss_function)

    @unittest.skip("Skip 'test_update_initialization_of_hyperparameters' because it takes long.")
    def test_update_initialization_of_hyperparameters(self):
        # Note that this is a weak test! We only check whether the hyperparameters did change.
        trajectory_randomizer = TrajectoryRandomizer(should_restart=True, restart_probability=1.,
                                                     length_partial_trajectory=1)
        initialization_assistant = InitializationAssistant(
            printing_enabled=True,
            maximal_number_of_iterations=100,
            update_stepsize_every=10,
            print_update_every=10,
            factor_update_stepsize=0.5
        )
        other_algorithm = copy.deepcopy(self.optimization_algorithm)
        other_algorithm.implementation.state_dict()['scale'] -= 0.5
        loss_functions = [LossFunction(dummy_function) for _ in range(10)]
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

    @unittest.skip("Skip 'test_initialize_helpers_for_initialization' because it takes long.")
    def test_initialize_helpers_for_initialization(self):
        parameters = {'with_print': True, 'num_iter_max': 100, 'lr': 1e-4,
                      'num_iter_update_stepsize': 10, 'num_iter_print_update': 10}
        optimizer, initialization_assistant, trajectory_randomizer = (
            self.optimization_algorithm.initialize_helpers_for_initialization(parameters=parameters))
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertIsInstance(trajectory_randomizer, TrajectoryRandomizer)
        self.assertIsInstance(initialization_assistant, InitializationAssistant)
