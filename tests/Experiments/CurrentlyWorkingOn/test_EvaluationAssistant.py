import copy
import unittest

from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from experiments.lasso.evaluation import EvaluationAssistant
from algorithms.dummy import Dummy
import torch


class TestEvaluationAssistant(unittest.TestCase):

    def setUp(self):

        def dummy_loss(x, p):
            return p['scale'] * torch.lingalg.norm(x) ** 2

        self.test_set = [{'scale': 0.1}]
        self.number_of_iterations_during_training = 10
        self.number_of_iterations_for_testing = 20
        self.initial_state = torch.rand(size=(1, 20))
        self.optimal_hyperparameters = copy.deepcopy(Dummy().state_dict())
        self.eval_assist = EvaluationAssistant(
            test_set=self.test_set, loss_of_algorithm=dummy_loss, smooth_part=dummy_loss, nonsmooth_part=dummy_loss,
            initial_state_learned_algorithm=self.initial_state,
            number_of_iterations_during_training=self.number_of_iterations_during_training,
            optimal_hyperparameters=self.optimal_hyperparameters, implementation_class=Dummy)

    def test_creation(self):
        self.assertIsInstance(self.eval_assist, EvaluationAssistant)

    def test_set_up_learned_algorithm(self):
        self.eval_assist.implementation_class = Dummy
        learned_algo = self.eval_assist.set_up_learned_algorithm(arguments_of_implementation_class=None)
        self.assertIsInstance(learned_algo, OptimizationAlgorithm)
        with self.assertRaises(Exception):
            self.eval_assist.set_up_learned_algorithm(arguments_of_implementation_class=1)