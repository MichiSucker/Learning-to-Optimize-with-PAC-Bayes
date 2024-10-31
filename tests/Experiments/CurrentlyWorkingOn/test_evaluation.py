import unittest
import torch
import torch.nn as nn

from algorithms.gradient_descent import GradientDescent
from experiments.nn_training.neural_network import NeuralNetworkForStandardTraining, NeuralNetworkForLearning
from experiments.nn_training.data_generation import get_data
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from experiments.nn_training.evaluation import (compute_losses,
                                                compute_ground_truth_loss,
                                                compute_losses_over_iterations_for_learned_algorithm,
                                                does_satisfy_constraint,
                                                compute_losses_over_iterations_for_adam)


class TestEvaluation(unittest.TestCase):

    def test_compute_ground_truth_loss(self):
        criterion = nn.MSELoss()
        parameter = {'ground_truth_values': torch.rand((10,)), 'y_values': torch.rand((10,))}
        gt_loss = compute_ground_truth_loss(loss_of_neural_network=criterion, parameter=parameter)
        self.assertEqual(gt_loss, criterion(parameter['ground_truth_values'], parameter['y_values']))

    def test_compute_losses_of_learned_algorithm(self):

        def dummy_function(x, parameter):
            return parameter['scale'] * torch.linalg.norm(x) ** 2

        dim = torch.randint(low=2, high=100, size=(1,)).item()
        initial_state = 2 + torch.randn(size=(dim, 1))
        algorithm = OptimizationAlgorithm(implementation=GradientDescent(alpha=torch.tensor(1e-1)),
                                          initial_state=initial_state,
                                          loss_function=ParametricLossFunction(function=dummy_function,
                                                                               parameter={'scale': 0.5}))
        number_of_iterations = torch.randint(low=2, high=15, size=(1,)).item()
        parameter = {'scale': 0.5}
        loss_function = ParametricLossFunction(dummy_function, parameter)
        losses = compute_losses_over_iterations_for_learned_algorithm(learned_algorithm=algorithm,
                                                                      loss_of_algorithm=dummy_function,
                                                                      parameter=parameter,
                                                                      number_of_iterations=number_of_iterations)
        self.assertTrue(len(losses), number_of_iterations+1)
        self.assertEqual(losses[0], loss_function(initial_state[-1]))

    def test_does_satisfy_constraint(self):

        def dummy_constraint(loss_at_beginning, loss_at_end):
            return loss_at_end < loss_at_beginning

        self.assertTrue(does_satisfy_constraint(convergence_risk_constraint=dummy_constraint,
                                                loss_at_beginning=10, loss_at_end=1))
        self.assertFalse(does_satisfy_constraint(convergence_risk_constraint=dummy_constraint,
                                                 loss_at_beginning=1, loss_at_end=10))

    def test_compute_losses_of_adam(self):
        criterion = nn.MSELoss()
        neural_network = NeuralNetworkForStandardTraining(degree=5)
        parameter = {'ground_truth': torch.rand((10,1)), 'x_values': torch.rand((10,1)), 'y_values': torch.rand((10,1))}
        losses_adam = compute_losses_over_iterations_for_adam(neural_network, loss_of_neural_network=criterion,
                                                              parameter=parameter, number_of_iterations=20)
        self.assertEqual(len(losses_adam), 21)

    def test_compute_losses(self):

        neural_network = NeuralNetworkForStandardTraining(degree=5)
        neural_network_for_learning = NeuralNetworkForLearning(degree=5,
                                                               shape_parameters=neural_network.get_shape_parameters())
        number_of_parameters = torch.randint(low=1, high=10, size=(1,)).item()
        loss_of_algorithm, loss_of_neural_network, parameters = get_data(
            neural_network_for_learning,
            number_of_datapoints_per_dataset={'prior': 0, 'train': 0, 'test': number_of_parameters, 'validation': 0})
        dim = neural_network.get_dimension_of_hyperparameters()
        initial_state = torch.randn(size=(1, dim))
        learned_algorithm = OptimizationAlgorithm(implementation=GradientDescent(alpha=torch.tensor(1e-1)),
                                                  initial_state=initial_state,
                                                  loss_function=ParametricLossFunction(function=loss_of_algorithm,
                                                                                       parameter=parameters['test'][0]))

        learned_algorithm.evaluate_loss_function_at_current_iterate()
        learned_algorithm.n_max = 10

        losses_adam, losses_of_learned_algorithm, ground_truth_losses, percentage = (
            compute_losses(parameters_to_test=parameters['test'],
                           learned_algorithm=learned_algorithm,
                           neural_network=neural_network,
                           loss_of_neural_network=loss_of_neural_network, loss_of_algorithm=loss_of_algorithm))
        self.assertEqual(len(losses_adam), len(losses_of_learned_algorithm))
        self.assertEqual(len(losses_adam), len(ground_truth_losses))
        self.assertTrue(0 <= percentage <= 1)



