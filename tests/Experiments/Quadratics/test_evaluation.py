import unittest
import torch
from experiments.quadratics.algorithm import Quadratics
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from experiments.quadratics.evaluation import (load_data,
                                               create_folder_for_storing_data,
                                               save_data,
                                               compute_losses_over_iterations,
                                               set_up_evaluation_assistant,
                                               EvaluationAssistant,
                                               does_satisfy_constraint,
                                               compute_losses,
                                               time_problem,
                                               compute_times,
                                               evaluate_algorithm)
from experiments.quadratics.training import get_baseline_algorithm
from main import TESTING_LEVEL


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.path_to_experiment = '/home/michael/Desktop/JMLR_New/Experiments/quadratics/'
        self.dummy_savings_path = self.path_to_experiment + 'dummy_data/'
        self.loading_path = self.path_to_experiment + 'data/'

    def test_load_data(self):
        load_data(self.loading_path)

    def test_create_folder(self):
        create_folder_for_storing_data(self.path_to_experiment)

    def test_save_data(self):
        save_data(self.dummy_savings_path,
                  times_of_learned_algorithm=[],
                  losses_of_learned_algorithm=[],
                  times_of_baseline_algorithm=[],
                  losses_of_baseline_algorithm=[],
                  ground_truth_losses=[],
                  percentage_constrained_satisfied=[])

    def test_set_up_evaluation_assistant(self):
        eval_assist = set_up_evaluation_assistant(loading_path=self.loading_path)
        self.assertIsInstance(eval_assist, EvaluationAssistant)
        self.assertIsInstance(eval_assist.implementation_class(eval_assist.implementation_arguments), Quadratics)

    def test_does_satisfy_constraint(self):

        def dummy_constraint(loss_at_beginning, loss_at_end):
            return loss_at_end < loss_at_beginning

        self.assertTrue(does_satisfy_constraint(convergence_risk_constraint=dummy_constraint,
                                                loss_at_beginning=10, loss_at_end=1))
        self.assertFalse(does_satisfy_constraint(convergence_risk_constraint=dummy_constraint,
                                                 loss_at_beginning=1, loss_at_end=10))

    def test_compute_losses_over_iterations(self):
        eval_assist = set_up_evaluation_assistant(loading_path=self.loading_path)
        eval_assist.test_set = eval_assist.test_set[0:2]
        algorithm = eval_assist.set_up_learned_algorithm(
            arguments_of_implementation_class=eval_assist.implementation_arguments)
        losses = compute_losses_over_iterations(algorithm=algorithm,
                                                evaluation_assistant=eval_assist,
                                                parameter=eval_assist.test_set[0])
        self.assertTrue(len(losses), eval_assist.number_of_iterations_for_testing + 1)
        self.assertEqual(losses[0], eval_assist.loss_of_algorithm(eval_assist.initial_state[-1],
                                                                  eval_assist.test_set[0]))

    def test_compute_losses(self):
        eval_assist = set_up_evaluation_assistant(loading_path=self.loading_path)
        eval_assist.test_set = eval_assist.test_set[0:2]
        learned_algorithm = eval_assist.set_up_learned_algorithm(
            arguments_of_implementation_class=eval_assist.implementation_arguments)
        baseline_algorithm = get_baseline_algorithm(loss_function=learned_algorithm.loss_function,
                                                    smoothness_constant=eval_assist.smoothness_parameter,
                                                    strong_convexity_constant=eval_assist.strong_convexity_parameter,
                                                    dim=eval_assist.dim)
        losses_of_baseline_algorithm, losses_of_learned_algorithm, percentage = (
            compute_losses(evaluation_assistant=eval_assist,
                           learned_algorithm=learned_algorithm,
                           baseline_algorithm=baseline_algorithm))
        self.assertEqual(len(losses_of_baseline_algorithm), len(losses_of_learned_algorithm))
        self.assertTrue(0 <= percentage <= 1)

    def test_time_problem(self):
        eval_assist = set_up_evaluation_assistant(loading_path=self.loading_path)
        eval_assist.test_set = eval_assist.test_set[0:2]
        learned_algorithm = eval_assist.set_up_learned_algorithm(
            arguments_of_implementation_class=eval_assist.implementation_arguments)
        loss_function = ParametricLossFunction(function=eval_assist.loss_of_algorithm,
                                               parameter=eval_assist.test_set[0])
        time = time_problem(algorithm=learned_algorithm, loss_function=loss_function,
                            maximal_number_of_iterations=5, optimal_loss=torch.tensor(0.), level_of_accuracy=1.)
        self.assertIsInstance(time, float)

    def test_compute_times(self):

        eval_assist = set_up_evaluation_assistant(loading_path=self.loading_path)
        eval_assist.test_set = eval_assist.test_set[0:2]
        learned_algorithm = eval_assist.set_up_learned_algorithm(
            arguments_of_implementation_class=eval_assist.implementation_arguments)
        baseline_algorithm = get_baseline_algorithm(loss_function=learned_algorithm.loss_function,
                                                    smoothness_constant=eval_assist.smoothness_parameter,
                                                    strong_convexity_constant=eval_assist.strong_convexity_parameter,
                                                    dim=eval_assist.dim)
        times_learned, times_adam = compute_times(
            learned_algorithm=learned_algorithm, baseline_algorithm=baseline_algorithm,
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
        evaluate_algorithm(loading_path=self.loading_path, path_of_experiment=self.path_to_experiment)
