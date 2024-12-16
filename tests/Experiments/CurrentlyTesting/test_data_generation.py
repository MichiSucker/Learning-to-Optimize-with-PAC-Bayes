import unittest
import torch
import torchvision
import torchvision.transforms as transforms
from typing import Callable
from experiments.mnist.neural_network import NeuralNetworkForLearning, NeuralNetworkForStandardTraining
from experiments.mnist.data_generation import (check_and_extract_number_of_datapoints,
                                               get_loss_of_neural_network,
                                               get_loss_of_algorithm,
                                               extract_data_from_dataloader,
                                               split_data,
                                               create_parameter,
                                               get_parameters,
                                               get_data)


class TestDataGeneration(unittest.TestCase):

    def test_check_and_extract_number_of_datapoints(self):
        # Check that it raises an error if at least one of the data sets is not specified.
        # And check that the extracted numbers are correct.
        with self.assertRaises(ValueError):
            check_and_extract_number_of_datapoints({})
        with self.assertRaises(ValueError):
            check_and_extract_number_of_datapoints({'prior': 1})
        with self.assertRaises(ValueError):
            check_and_extract_number_of_datapoints({'prior': 1, 'train': 1})
        with self.assertRaises(ValueError):
            check_and_extract_number_of_datapoints({'prior': 1, 'train': 1, 'test': 1})
        number_data = {'prior': torch.randint(low=1, high=100, size=(1,)).item(),
                       'train': torch.randint(low=1, high=100, size=(1,)).item(),
                       'test': torch.randint(low=1, high=100, size=(1,)).item(),
                       'validation': torch.randint(low=1, high=100, size=(1,)).item()}
        n_prior, n_train, n_test, n_val = check_and_extract_number_of_datapoints(number_data)
        self.assertEqual(n_prior, number_data['prior'])
        self.assertEqual(n_train, number_data['train'])
        self.assertEqual(n_test, number_data['test'])
        self.assertEqual(n_val, number_data['validation'])

    def test_get_loss_of_neural_network(self):
        f = get_loss_of_neural_network()
        self.assertIsInstance(f, Callable)
        x, y = torch.ones((10, 1)), 0.5 * torch.ones((10, 1))
        val = f(x, y)
        self.assertIsInstance(val, torch.Tensor)
        self.assertIsInstance(val.item(), float)

    def test_get_loss_of_algorithm(self):
        net = NeuralNetworkForLearning()
        net_std = NeuralNetworkForStandardTraining()
        loss_of_net = get_loss_of_neural_network()
        f = get_loss_of_algorithm(net, loss_of_net)
        images = torch.randn((10, 1, 20, 20))
        labels = torch.randint(low=0, high=10, size=(10,))
        parameter = {'images': images, 'labels': labels}
        val = f(x=net_std.transform_state_dict_to_tensor(), parameter=parameter)
        self.assertIsInstance(val, torch.Tensor)
        self.assertIsInstance(val.item(), float)

    def test_extract_data_from_dataloader(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])
        dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=100, num_workers=0)
        test_images, test_labels = extract_data_from_dataloader(loader)
        self.assertIsInstance(test_images, torch.Tensor)
        self.assertIsInstance(test_labels, torch.Tensor)
        self.assertTrue(len(test_images) == len(test_labels))
        self.assertEqual(test_images[0].shape, torch.Size((1, 28, 28)))
        self.assertIsInstance(test_labels[0].item(), int)

    def test_split_data(self):
        prior_data, train_data, test_data, validation_data = split_data()
        for d in [prior_data, train_data, test_data, validation_data]:
            self.assertIsInstance(d, dict)
            self.assertTrue('images' in d.keys())
            self.assertTrue('labels' in d.keys())
            self.assertEqual(len(d.keys()), 2)
            self.assertTrue(len(d['images']) > 0)
            self.assertTrue(len(d['images']) == len(d['labels']))

    def test_create_parameter(self):
        images = torch.randn((2, 10))
        labels = torch.randn((2, ))
        param = create_parameter(images, labels)
        self.assertIsInstance(param, dict)
        self.assertTrue('images' in param.keys())
        self.assertTrue('labels' in param.keys())
        self.assertTrue('optimal_loss' in param.keys())
        self.assertEqual(len(param.keys()), 3)
        self.assertTrue(torch.equal(images, param['images']))
        self.assertTrue(torch.equal(labels, param['labels']))
        self.assertTrue(torch.equal(torch.tensor(0.0), param['optimal_loss']))

    def test_get_parameters(self):
        number_of_datapoints = {'prior': 1, 'train': 2, 'test': 3, 'validation': 4}
        parameters = get_parameters(number_of_datapoints)
        self.assertEqual(len(parameters.keys()), len(number_of_datapoints.keys()))
        for d in number_of_datapoints.keys():
            self.assertTrue(d in parameters.keys())
            self.assertEqual(len(parameters[d]), number_of_datapoints[d])

    def test_get_data(self):
        number_of_datapoints = {'prior': 1, 'train': 2, 'test': 3, 'validation': 4}
        net = NeuralNetworkForLearning()
        loss_of_algorithm, loss_of_neural_network, parameters = get_data(net, number_of_datapoints)
        self.assertIsInstance(loss_of_algorithm, Callable)
        self.assertIsInstance(loss_of_neural_network, Callable)
        self.assertIsInstance(parameters, dict)