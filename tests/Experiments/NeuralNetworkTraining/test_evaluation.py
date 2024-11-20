import unittest
import torch
import torch.nn as nn
import numpy as np
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from experiments.nn_training.evaluation import (compute_losses,
                                                compute_ground_truth_loss,
                                                compute_losses_over_iterations_for_learned_algorithm,
                                                does_satisfy_constraint,
                                                compute_losses_over_iterations_for_adam,
                                                set_up_evaluation_assistant,
                                                load_data, EvaluationAssistant,
                                                time_problem_for_adam,
                                                time_problem_for_learned_algorithm,
                                                compute_times,
                                                evaluate_algorithm,
                                                save_data,
                                                create_folder_for_storing_data)


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.data_path = '/home/michael/Desktop/JMLR_New/Experiments/neural_network_training/data/'
        self.dummy_savings_path = '/home/michael/Desktop/JMLR_New/Experiments/neural_network_training/dummy_data/'

    def test_load_data(self):
        # Just check that it does not throw an error => all variables are found.
        load_data(self.data_path)

    def test_create_folder(self):
        # Just check that it does not throw an error.
        create_folder_for_storing_data(self.data_path)

    def test_compute_ground_truth_loss(self):
        # Check that the ground-truth loss is computed with MSE.
        criterion = nn.MSELoss()
        parameter = {'ground_truth_values': torch.rand((10,)), 'y_values': torch.rand((10,))}
        gt_loss = compute_ground_truth_loss(loss_of_neural_network=criterion, parameter=parameter)
        self.assertEqual(gt_loss, criterion(parameter['ground_truth_values'], parameter['y_values']))

    def test_compute_losses_of_learned_algorithm(self):
        # Check that we get N+1 values, where N is the number of iterations for testing.
        # +1 because we want the initial loss.
        eval_assist, neural_network = set_up_evaluation_assistant(loading_path=self.data_path)
        learned_algorithm = eval_assist.set_up_learned_algorithm(
            arguments_of_implementation_class=eval_assist.implementation_arguments)
        losses = compute_losses_over_iterations_for_learned_algorithm(learned_algorithm=learned_algorithm,
                                                                      evaluation_assistant=eval_assist,
                                                                      parameter=eval_assist.test_set[0])
        self.assertTrue(len(losses), eval_assist.number_of_iterations_for_testing + 1)
        self.assertEqual(losses[0], eval_assist.loss_of_algorithm(eval_assist.initial_state[-1],
                                                                  eval_assist.test_set[0]))

    def test_does_satisfy_constraint(self):

        def dummy_constraint(loss_at_beginning, loss_at_end):
            return loss_at_end < loss_at_beginning

        self.assertTrue(does_satisfy_constraint(convergence_risk_constraint=dummy_constraint,
                                                loss_at_beginning=10, loss_at_end=1))
        self.assertFalse(does_satisfy_constraint(convergence_risk_constraint=dummy_constraint,
                                                 loss_at_beginning=1, loss_at_end=10))

    def test_compute_losses_of_adam(self):
        # Basically the same test as for the learned algorithm.
        eval_assist, neural_network = set_up_evaluation_assistant(loading_path=self.data_path)
        neural_network.load_parameters_from_tensor(eval_assist.initial_state[-1].clone())
        losses_adam = compute_losses_over_iterations_for_adam(neural_network, evaluation_assistant=eval_assist,
                                                              parameter=eval_assist.test_set[0])
        self.assertEqual(len(losses_adam), eval_assist.number_of_iterations_for_testing + 1)
        self.assertEqual(losses_adam[0], eval_assist.loss_of_algorithm(eval_assist.initial_state[-1],
                                                                       eval_assist.test_set[0]))

    def test_compute_losses(self):
        # Check that we compute the same amount of values for both algorithms, that we have the correct number of
        # ground-truth losses, and that percentage lies in [0,1].
        eval_assist, neural_network = set_up_evaluation_assistant(loading_path=self.data_path)
        eval_assist.test_set = eval_assist.test_set[0:2]
        learned_algorithm = eval_assist.set_up_learned_algorithm(
            arguments_of_implementation_class=eval_assist.implementation_arguments)

        losses_adam, losses_of_learned_algorithm, ground_truth_losses, percentage = (
            compute_losses(evaluation_assistant=eval_assist,
                           learned_algorithm=learned_algorithm,
                           neural_network_for_standard_training=neural_network))
        self.assertEqual(losses_adam.shape, losses_of_learned_algorithm.shape)
        self.assertEqual(len(losses_adam), len(ground_truth_losses))
        self.assertTrue(0 <= percentage <= 1)

    def test_time_problem_for_learned_algorithm(self):
        # Check that the elapsed time is a positive float.
        eval_assist, neural_network = set_up_evaluation_assistant(loading_path=self.data_path)
        eval_assist.test_set = eval_assist.test_set[0:2]
        learned_algorithm = eval_assist.set_up_learned_algorithm(
            arguments_of_implementation_class=eval_assist.implementation_arguments)
        loss_function = ParametricLossFunction(function=eval_assist.loss_of_algorithm,
                                               parameter=eval_assist.test_set[0])
        time = time_problem_for_learned_algorithm(learned_algorithm=learned_algorithm, loss_function=loss_function,
                                                  maximal_number_of_iterations=5, optimal_loss=torch.tensor(0.),
                                                  level_of_accuracy=1.)
        self.assertIsInstance(time, float)
        self.assertTrue(time > 0)

    def test_time_problem_for_adam(self):
        # Basically the same test as for the learned algorithm.
        eval_assist, neural_network = set_up_evaluation_assistant(loading_path=self.data_path)
        eval_assist.test_set = eval_assist.test_set[0:2]
        time = time_problem_for_adam(
            neural_network=neural_network, loss_of_neural_network=eval_assist.loss_of_neural_network,
            maximal_number_of_iterations=5, parameter=eval_assist.test_set[0], optimal_loss=torch.tensor(0.),
            level_of_accuracy=1., lr_adam=torch.tensor(0.008))
        self.assertIsInstance(time, float)
        self.assertTrue(time > 0)

    def test_compute_times(self):
        eval_assist, neural_network = set_up_evaluation_assistant(loading_path=self.data_path)
        eval_assist.test_set = eval_assist.test_set[0:2]
        learned_algorithm = eval_assist.set_up_learned_algorithm(
            arguments_of_implementation_class=eval_assist.implementation_arguments)
        times_learned, times_adam = compute_times(
            learned_algorithm=learned_algorithm, neural_network_for_standard_training=neural_network,
            evaluation_assistant=eval_assist,
            ground_truth_losses=torch.zeros(len(eval_assist.test_set)),
            stop_procedure_after_at_most=5)

        self.assertIsInstance(times_learned, dict)
        self.assertIsInstance(times_adam, dict)
        self.assertEqual(len(times_adam.keys()), 3)
        self.assertEqual(len(times_learned.keys()), 3)
        for key in times_adam.keys():
            self.assertEqual(len(times_adam[key]), len(eval_assist.test_set) + 1)
        for key in times_learned.keys():
            self.assertEqual(len(times_learned[key]), len(eval_assist.test_set) + 1)

    def test_set_up_evaluation_assistant(self):
        eval_assist, _ = set_up_evaluation_assistant(loading_path=self.data_path)
        self.assertIsInstance(eval_assist, EvaluationAssistant)

    def test_save_data(self):
        # This is a weak test.
        save_data(savings_path=self.dummy_savings_path,
                  times_of_learned_algorithm={},
                  times_of_adam={},
                  losses_of_learned_algorithm=np.empty(1),
                  losses_of_adam=np.empty(1),
                  ground_truth_losses=np.empty(1),
                  percentage_constrained_satisfied=1.)

    @unittest.skip('Too expensive to test all the time.')
    def test_evaluate_algorithm(self):
        path_of_experiment = '/home/michael/Desktop/JMLR_New/Experiments/neural_network_training/'
        loading_path = '/home/michael/Desktop/JMLR_New/Experiments/neural_network_training/data_after_training/'
        evaluate_algorithm(loading_path=loading_path, path_of_experiment=path_of_experiment)





