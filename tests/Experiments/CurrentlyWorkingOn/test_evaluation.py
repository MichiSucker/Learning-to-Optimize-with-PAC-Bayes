import unittest
import torch
import torch.nn as nn
from main import TESTING_LEVEL
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
                                                evaluate_algorithm)


class TestEvaluation(unittest.TestCase):

    def test_compute_ground_truth_loss(self):
        criterion = nn.MSELoss()
        parameter = {'ground_truth_values': torch.rand((10,)), 'y_values': torch.rand((10,))}
        gt_loss = compute_ground_truth_loss(loss_of_neural_network=criterion, parameter=parameter)
        self.assertEqual(gt_loss, criterion(parameter['ground_truth_values'], parameter['y_values']))

    def test_compute_losses_of_learned_algorithm(self):

        path = '/home/michael/Desktop/JMLR_New/Experiments/neural_network_training/data_after_training/'
        eval_assist, neural_network = set_up_evaluation_assistant(loading_path=path)
        eval_assist.test_set = eval_assist.test_set[0:2]
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
        path = '/home/michael/Desktop/JMLR_New/Experiments/neural_network_training/data_after_training/'
        eval_assist, neural_network = set_up_evaluation_assistant(loading_path=path)
        losses_adam = compute_losses_over_iterations_for_adam(neural_network, evaluation_assistant=eval_assist,
                                                              parameter=eval_assist.test_set[0])
        self.assertEqual(len(losses_adam), eval_assist.number_of_iterations_for_testing + 1)

    def test_set_up_evaluation_assistant(self):
        path = '/home/michael/Desktop/JMLR_New/Experiments/neural_network_training/data_after_training/'
        eval_assist, _ = set_up_evaluation_assistant(loading_path=path)
        self.assertIsInstance(eval_assist, EvaluationAssistant)

    def test_load_data(self):
        path = '/home/michael/Desktop/JMLR_New/Experiments/neural_network_training/data_after_training/'
        load_data(path)

    def test_compute_losses(self):
        path = '/home/michael/Desktop/JMLR_New/Experiments/neural_network_training/data_after_training/'
        eval_assist, neural_network = set_up_evaluation_assistant(loading_path=path)
        eval_assist.test_set = eval_assist.test_set[0:2]
        learned_algorithm = eval_assist.set_up_learned_algorithm(
            arguments_of_implementation_class=eval_assist.implementation_arguments)

        losses_adam, losses_of_learned_algorithm, ground_truth_losses, percentage = (
            compute_losses(evaluation_assistant=eval_assist,
                           learned_algorithm=learned_algorithm,
                           neural_network_for_standard_training=neural_network))
        self.assertEqual(len(losses_adam), len(losses_of_learned_algorithm))
        self.assertEqual(len(losses_adam), len(ground_truth_losses))
        self.assertTrue(0 <= percentage <= 1)

    def test_time_problem_for_learned_algorithm(self):
        path = '/home/michael/Desktop/JMLR_New/Experiments/neural_network_training/data_after_training/'
        eval_assist, neural_network = set_up_evaluation_assistant(loading_path=path)
        eval_assist.test_set = eval_assist.test_set[0:2]
        learned_algorithm = eval_assist.set_up_learned_algorithm(
            arguments_of_implementation_class=eval_assist.implementation_arguments)
        loss_function = ParametricLossFunction(function=eval_assist.loss_of_algorithm,
                                               parameter=eval_assist.test_set[0])
        time = time_problem_for_learned_algorithm(learned_algorithm=learned_algorithm, loss_function=loss_function,
                                                  maximal_number_of_iterations=5, optimal_loss=torch.tensor(0.),
                                                  level_of_accuracy=1.)
        self.assertIsInstance(time, float)

    def test_time_problem_for_adam(self):
        path = '/home/michael/Desktop/JMLR_New/Experiments/neural_network_training/data_after_training/'
        eval_assist, neural_network = set_up_evaluation_assistant(loading_path=path)
        eval_assist.test_set = eval_assist.test_set[0:2]
        time = time_problem_for_adam(
            neural_network=neural_network, loss_of_neural_network=eval_assist.loss_of_neural_network,
            maximal_number_of_iterations=5, parameter=eval_assist.test_set[0], optimal_loss=torch.tensor(0.),
            level_of_accuracy=1.)
        self.assertIsInstance(time, float)

    def compute_times(self):
        path = '/home/michael/Desktop/JMLR_New/Experiments/neural_network_training/data_after_training/'
        eval_assist, neural_network = set_up_evaluation_assistant(loading_path=path)
        eval_assist.test_set = eval_assist.test_set[0:2]
        learned_algorithm = eval_assist.set_up_learned_algorithm(
            arguments_of_implementation_class=eval_assist.implementation_arguments)
        times_learned, times_adam = compute_times(
            learned_algorithm=learned_algorithm, neural_network_for_standard_training=neural_network,
            evaluation_assistant=eval_assist, ground_truth_losses=[torch.tensor(0.) for _ in eval_assist.test_set],
            stop_procedure_after_at_most=5)

        self.assertIsInstance(times_learned, dict)
        self.assertIsInstance(times_adam, dict)
        self.assertEqual(len(times_adam.keys()), 3)
        self.assertEqual(len(times_learned.keys()), 3)
        for key in times_adam.keys():
            self.assertEqual(len(times_adam[key]), len(eval_assist.test_set) + 1)
        for key in times_learned.keys():
            self.assertEqual(len(times_learned[key]), len(eval_assist.test_set) + 1)

    @unittest.skipIf(condition=(TESTING_LEVEL != 'FULL_TEST_WITH_EXPERIMENTS'),
                     reason='Too expensive to test all the time.')
    def test_evaluate_algorithm(self):
        path_of_experiment = '/home/michael/Desktop/JMLR_New/Experiments/neural_network_training/'
        loading_path = '/home/michael/Desktop/JMLR_New/Experiments/neural_network_training/data_after_training/'
        evaluate_algorithm(loading_path=loading_path, path_of_experiment=path_of_experiment)





