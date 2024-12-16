import unittest
import torch
from experiments.mnist.data_generation import get_loss_of_neural_network
from experiments.mnist.neural_network import NeuralNetworkForLearning, NeuralNetworkForStandardTraining, train_model


class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        self.net_for_std_training = NeuralNetworkForStandardTraining()

    def test_input_shape(self):
        input_tensor = torch.rand(1, 1, 28, 20)
        with self.assertRaises(RuntimeError):
            self.net_for_std_training(input_tensor)
        input_tensor = torch.rand(1, 1, 20, 20)
        self.net_for_std_training(input_tensor)

    def test_output_shape(self):
        input_tensor = torch.rand(1, 1, 20, 20)
        output = self.net_for_std_training(input_tensor)
        self.assertEqual(output.shape, torch.Size((1, 10)))

    def test_same_output(self):
        input_tensor = torch.rand(1, 1, 20, 20)
        net_for_learning = NeuralNetworkForLearning()
        output_1 = self.net_for_std_training(input_tensor)
        output_2 = net_for_learning(x=input_tensor,
                                    neural_net_parameters=self.net_for_std_training.transform_state_dict_to_tensor())

        self.net_for_std_training.load_tensor_into_state_dict(self.net_for_std_training.transform_state_dict_to_tensor())
        output_3 = self.net_for_std_training(input_tensor)
        self.assertTrue(torch.equal(output_1, output_2))
        self.assertTrue(torch.equal(output_2, output_3))

    def test_transform_state_dict_to_tensor(self):
        weights = self.net_for_std_training.transform_state_dict_to_tensor()
        self.assertIsInstance(weights, torch.Tensor)
        self.assertTrue(torch.equal(weights.flatten(), weights))

    def test_transform_tensor_to_state_dict(self):
        weights = self.net_for_std_training.transform_state_dict_to_tensor()
        weights[-1] = 0
        state_dict = self.net_for_std_training.transform_tensor_into_state_dict(weights)
        self.net_for_std_training.load_state_dict(state_dict)
        self.assertTrue(torch.equal(weights, self.net_for_std_training.transform_state_dict_to_tensor()))

    def test_load_tensor_into_state_dict(self):
        weights = self.net_for_std_training.transform_state_dict_to_tensor()
        weights[-1] = 0
        self.net_for_std_training.load_tensor_into_state_dict(weights)
        self.assertTrue(torch.equal(weights, self.net_for_std_training.transform_state_dict_to_tensor()))

    def test_get_shape_parameters(self):
        shape_params = self.net_for_std_training.get_shape_parameters()
        self.assertIsInstance(shape_params, list)
        for item in shape_params:
            self.assertIsInstance(item, torch.Size)

    def test_get_dimension(self):
        dim = self.net_for_std_training.get_dimension_of_hyperparameters()
        self.assertIsInstance(dim, int)

    def test_train_model(self):
        weights = self.net_for_std_training.transform_state_dict_to_tensor()
        data = {'images': torch.rand(10, 1, 20, 20), 'labels': torch.randint(low=0, high=10, size=(10,))}
        criterion = get_loss_of_neural_network()
        n_it = 10
        net, losses, iterates = train_model(net=self.net_for_std_training,
                                            data=data, criterion=criterion,
                                            n_it=n_it, lr=1e-4)
        new_weights = net.transform_state_dict_to_tensor()
        self.assertFalse(torch.equal(weights, new_weights))
        self.assertIsInstance(losses, list)
        self.assertIsInstance(iterates, list)
        self.assertEqual(len(losses), n_it + 1)
        self.assertEqual(len(iterates), n_it + 1)
        self.assertIsInstance(losses[0], float)
        self.assertIsInstance(iterates[0], torch.Tensor)
        self.assertEqual(len(iterates[0].flatten()), net.get_dimension_of_hyperparameters())
