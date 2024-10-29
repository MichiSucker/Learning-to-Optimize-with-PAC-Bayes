import unittest
import torch
from experiments.nn_training.neural_network import (polynomial_features,
                                                    NeuralNetworkForStandardTraining,
                                                    NeuralNetworkForLearning,
                                                    train_model)
from experiments.nn_training.data_generation import get_data


class TestNeuralNetwork(unittest.TestCase):

    def test_polynomial_features(self):
        degree = torch.randint(low=1, high=10, size=(1,)).item()
        x = torch.randn(size=(1,))
        pol = polynomial_features(x=x, degree=degree)
        self.assertIsInstance(pol, torch.Tensor)
        self.assertTrue(len(pol.flatten()), degree+1)
        self.assertTrue(torch.allclose(pol, (x ** torch.arange(1, degree+1)).reshape((1, -1))))

        n = torch.randint(low=2, high=10, size=(1,)).item()
        x = torch.randn(size=(n,))
        pol = polynomial_features(x=x, degree=degree)
        self.assertTrue(pol.shape, torch.Size((n, degree)))
        for i in range(n):
            self.assertTrue(torch.allclose(pol[i], (x[i] ** torch.arange(1, degree+1)).reshape((1, -1))))

    def test_get_shape_parameters(self):
        degree = 5
        nn_std = NeuralNetworkForStandardTraining(degree=degree)
        shape_parameters = nn_std.get_shape_parameters()
        self.assertIsInstance(shape_parameters, list)
        self.assertEqual(shape_parameters, [p.shape for p in nn_std.parameters()])

    def test_get_dimension_of_hyperparameters(self):
        degree = 5
        nn_std = NeuralNetworkForStandardTraining(degree=degree)
        dim = nn_std.get_dimension_of_hyperparameters()
        self.assertIsInstance(dim, int)
        self.assertEqual(dim, sum([torch.prod(torch.tensor(p.shape)).item() for p in nn_std.parameters()]))

    def test_nn_to_tensor(self):
        degree = 5
        nn_std = NeuralNetworkForStandardTraining(degree=degree)
        dim = nn_std.get_dimension_of_hyperparameters()
        all_parameters = nn_std.transform_parameters_to_tensor()
        self.assertIsInstance(all_parameters, torch.Tensor)
        self.assertEqual(len(all_parameters), dim)

    def test_load_tensor_as_parameters_of_nn(self):
        degree = 5
        nn_std = NeuralNetworkForStandardTraining(degree=degree)
        all_parameters = nn_std.transform_parameters_to_tensor()
        random_parameters = torch.randn(all_parameters.shape)
        self.assertFalse(torch.equal(all_parameters, random_parameters))
        nn_std.load_parameters_from_tensor(tensor=random_parameters)
        parameters_now = nn_std.transform_parameters_to_tensor()
        self.assertTrue(torch.equal(parameters_now, random_parameters))

    def test_call_and_compare_neural_networks(self):
        degree = 5
        nn_std = NeuralNetworkForStandardTraining(degree=degree)
        x = torch.randn(size=(10,))
        self.assertIsInstance(nn_std(x), torch.Tensor)
        self.assertTrue(nn_std(x).shape, x.shape)

        shape_parameters = nn_std.get_shape_parameters()
        nn_for_learning = NeuralNetworkForLearning(degree=degree, shape_parameters=shape_parameters)
        parameters_of_neural_network = nn_std.transform_parameters_to_tensor()
        x = torch.randn(size=(10,))
        self.assertIsInstance(nn_for_learning(x, parameters_of_neural_network), torch.Tensor)
        self.assertTrue(nn_for_learning(x, parameters_of_neural_network).shape, x.shape)
        self.assertTrue(torch.equal(nn_std(x), nn_for_learning(x, parameters_of_neural_network)))

    def test_train_model(self):
        degree = 5
        nn_std = NeuralNetworkForStandardTraining(degree=degree)
        shape_parameters = nn_std.get_shape_parameters()
        nn_for_learning = NeuralNetworkForLearning(degree=degree, shape_parameters=shape_parameters)
        loss_of_algorithm, loss_of_neural_network, parameters = get_data(
            neural_network=nn_for_learning,
            number_of_datapoints_per_dataset={'prior': 0, 'train': 1, 'test': 0, 'validation': 0})
        data = parameters['train'][0]
        net, losses, iterates = train_model(net=nn_std, data=data, criterion=loss_of_neural_network, n_it=100, lr=1e-4)
        self.assertIsInstance(net, NeuralNetworkForStandardTraining)
        self.assertIsInstance(losses, list)
        self.assertIsInstance(iterates, list)
        self.assertTrue(losses[0] > losses[-1])
        dim = nn_std.get_dimension_of_hyperparameters()
        for it in iterates:
            self.assertEqual(dim, len(it))
