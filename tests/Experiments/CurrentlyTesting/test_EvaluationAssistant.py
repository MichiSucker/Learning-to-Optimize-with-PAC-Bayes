import unittest
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from experiments.mnist.evaluation import EvaluationAssistant
from experiments.mnist.neural_network import NeuralNetworkForStandardTraining
from experiments.mnist.algorithm import MnistOptimizer
import torch
import copy


class TestEvaluationAssistant(unittest.TestCase):

    def setUp(self):

        def dummy_loss(x, p):
            return p['scale'] * torch.lingalg.norm(x) ** 2

        self.test_set = [{'scale': 0.1}]
        self.number_of_iterations_during_training = 10
        self.number_of_iterations_for_testing = 20
        self.initial_state = torch.rand(size=(1, 20))
        dim = NeuralNetworkForStandardTraining().get_dimension_of_hyperparameters()
        self.optimal_hyperparameters = copy.deepcopy(MnistOptimizer(dim).state_dict())
        self.eval_assist = EvaluationAssistant(
            test_set=self.test_set, number_of_iterations_during_training=self.number_of_iterations_during_training,
            number_of_iterations_for_testing=self.number_of_iterations_for_testing, loss_of_algorithm=dummy_loss,
            initial_state=self.initial_state, optimal_hyperparameters=self.optimal_hyperparameters,
            dimension_of_hyperparameters=dim
        )

    def test_creation(self):
        self.assertIsInstance(self.eval_assist, EvaluationAssistant)

    def test_set_up_learned_algorithm(self):
        learned_algo = self.eval_assist.set_up_learned_algorithm()
        self.assertIsInstance(learned_algo, OptimizationAlgorithm)
        with self.assertRaises(Exception):
            self.eval_assist.set_up_learned_algorithm(1)
