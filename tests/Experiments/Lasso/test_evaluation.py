import unittest
import torch

from algorithms.fista import FISTA
from algorithms.gradient_descent import GradientDescent
from classes.LossFunction.class_LossFunction import LossFunction
from classes.LossFunction.derived_classes.derived_classes.\
    subclass_NonsmoothParametricLossFunction import NonsmoothParametricLossFunction
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from experiments.lasso.evaluation import EvaluationAssistant
from experiments.lasso.algorithm import SparsityNet
from experiments.lasso.training import get_baseline_algorithm
from experiments.lasso.evaluation import (evaluate_algorithm,
                                          load_data,
                                          create_folder_for_storing_data,
                                          set_up_evaluation_assistant,
                                          save_data,
                                          does_satisfy_constraint,
                                          compute_losses_over_iterations,
                                          compute_losses,
                                          time_problem,
                                          compute_times,
                                          approximate_optimal_loss,
                                          set_up_algorithms)


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.path_to_experiment = '/home/michael/Desktop/JMLR_New/Experiments/lasso/'
        self.dummy_savings_path = self.path_to_experiment + 'dummy_data/'
        self.loading_path = self.path_to_experiment + 'data/'

    def test_load_data(self):
        load_data(loading_path=self.loading_path)

    def test_create_folder(self):
        create_folder_for_storing_data(path_of_experiment=self.path_to_experiment)

    def test_save_data(self):
        save_data(self.dummy_savings_path,
                  times_of_learned_algorithm=[],
                  losses_of_learned_algorithm=[],
                  times_of_baseline_algorithm=[],
                  losses_of_baseline_algorithm=[],
                  percentage_constrained_satisfied=[],
                  approximate_optimal_losses=[],
                  number_of_iterations_for_approximation=1)

    def test_set_up_evaluation_assistant(self):
        eval_assist = set_up_evaluation_assistant(loading_path=self.loading_path)
        self.assertIsInstance(eval_assist, EvaluationAssistant)
        self.assertIsInstance(eval_assist.implementation_class(eval_assist.implementation_arguments), SparsityNet)

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
        self.assertEqual(losses[0], eval_assist.loss_of_algorithm(eval_assist.initial_state_learned_algorithm[-1],
                                                                  eval_assist.test_set[0]))

    def test_compute_losses(self):
        eval_assist = set_up_evaluation_assistant(loading_path=self.loading_path)
        eval_assist.test_set = eval_assist.test_set[0:2]
        learned_algorithm = eval_assist.set_up_learned_algorithm(
            arguments_of_implementation_class=eval_assist.implementation_arguments)
        baseline_algorithm = get_baseline_algorithm(smoothness_parameter=eval_assist.smoothness_parameter,
                                                    initial_state=eval_assist.initial_state_baseline_algorithm,
                                                    loss_function=learned_algorithm.loss_function)
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
        loss_function = NonsmoothParametricLossFunction(function=eval_assist.loss_of_algorithm,
                                                        smooth_part=eval_assist.smooth_part,
                                                        nonsmooth_part=eval_assist.nonsmooth_part,
                                                        parameter=eval_assist.test_set[0])
        time = time_problem(algorithm=learned_algorithm, loss_function=loss_function,
                            maximal_number_of_iterations=5, optimal_loss=torch.tensor(0.), level_of_accuracy=1.)
        self.assertIsInstance(time, float)

    def test_compute_times(self):

        eval_assist = set_up_evaluation_assistant(loading_path=self.loading_path)
        eval_assist.test_set = eval_assist.test_set[0:2]
        learned_algorithm = eval_assist.set_up_learned_algorithm(
            arguments_of_implementation_class=eval_assist.implementation_arguments)
        baseline_algorithm = get_baseline_algorithm(smoothness_parameter=eval_assist.smoothness_parameter,
                                                    initial_state=eval_assist.initial_state_baseline_algorithm,
                                                    loss_function=learned_algorithm.loss_function)
        times_learned, times_adam = compute_times(learned_algorithm=learned_algorithm,
                                                  baseline_algorithm=baseline_algorithm,
                                                  evaluation_assistant=eval_assist,
                                                  optimal_losses=[torch.tensor(0.) for _ in eval_assist.test_set],
                                                  stop_procedure_after_at_most=5)

        self.assertIsInstance(times_learned, dict)
        self.assertIsInstance(times_adam, dict)
        self.assertEqual(len(times_adam.keys()), 3)
        self.assertEqual(len(times_learned.keys()), 3)
        for key in times_adam.keys():
            self.assertEqual(len(times_adam[key]), len(eval_assist.test_set) + 1)
        for key in times_learned.keys():
            self.assertEqual(len(times_learned[key]), len(eval_assist.test_set) + 1)

    def test_approximate_solution(self):

        def dummy_function(x, parameter):
            return parameter['scale'] * torch.linalg.norm(x) ** 2

        test_set = [{'scale': 0.1}, {'scale': 0.5}, {'scale': 0.25}]

        eval_assist = set_up_evaluation_assistant(loading_path=self.loading_path)
        eval_assist.loss_of_algorithm = dummy_function
        eval_assist.test_set = test_set
        number_of_iterations = 10
        eval_assist.number_of_iterations_for_approximation = number_of_iterations

        baseline_algorithm = OptimizationAlgorithm(
            implementation=GradientDescent(alpha=torch.tensor(0.1)),
            initial_state=eval_assist.initial_state_baseline_algorithm[-1].reshape((1, -1)),
            loss_function=LossFunction(dummy_function))
        optimal_losses = approximate_optimal_loss(baseline_algorithm=baseline_algorithm,
                                                  evaluation_assistant=eval_assist)
        self.assertEqual(len(optimal_losses), len(test_set))
        for i, p in enumerate(test_set):
            baseline_algorithm.reset_state_and_iteration_counter()
            baseline_algorithm.set_loss_function(ParametricLossFunction(function=dummy_function, parameter=p))
            for _ in range(number_of_iterations):
                baseline_algorithm.perform_step()
            self.assertEqual(baseline_algorithm.iteration_counter, number_of_iterations)
            self.assertEqual(optimal_losses[i], baseline_algorithm.evaluate_loss_function_at_current_iterate().item())

    def test_set_up_algorithms(self):
        eval_assist = set_up_evaluation_assistant(loading_path=self.loading_path)
        learned, std = set_up_algorithms(eval_assist)
        self.assertIsInstance(learned, OptimizationAlgorithm)
        self.assertIsInstance(learned.implementation, SparsityNet)
        self.assertIsInstance(std, OptimizationAlgorithm)
        self.assertIsInstance(std.implementation, FISTA)

    # @unittest.skipIf(condition=(TESTING_LEVEL != 'FULL_TEST_WITH_EXPERIMENTS'),
    #                  reason='Too expensive to test all the time.')
    def test_evaluate_algorithm(self):
        evaluate_algorithm(loading_path=self.loading_path, path_of_experiment=self.path_to_experiment)
