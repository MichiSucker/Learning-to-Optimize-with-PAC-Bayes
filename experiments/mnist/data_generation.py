import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.v2 import Resize

from experiments.mnist.neural_network import NeuralNetworkForLearning
from typing import Callable, Tuple


def check_and_extract_number_of_datapoints(number_of_datapoints_per_dataset: dict) -> Tuple[int, int, int, int]:
    if (('prior' not in number_of_datapoints_per_dataset)
            or ('train' not in number_of_datapoints_per_dataset)
            or ('test' not in number_of_datapoints_per_dataset)
            or ('validation' not in number_of_datapoints_per_dataset)):
        raise ValueError("Missing number of datapoints.")
    else:
        return (number_of_datapoints_per_dataset['prior'],
                number_of_datapoints_per_dataset['train'],
                number_of_datapoints_per_dataset['test'],
                number_of_datapoints_per_dataset['validation'])


def get_loss_of_neural_network() -> Callable:
    crit = nn.CrossEntropyLoss()

    def f(y_1: torch.Tensor, y_2: torch.Tensor) -> torch.Tensor:
        return crit(y_1, y_2) / torch.mean(torch.exp(-torch.sqrt(1. - torch.argmax(y_1, dim=1).eq(y_2).float())))
    return f


def get_loss_of_algorithm(neural_network: NeuralNetworkForLearning, loss_of_neural_network: Callable) -> Callable:

    def loss_function_of_the_algorithm(x: torch.Tensor, parameter: dict) -> torch.Tensor:
        return loss_of_neural_network(neural_network(x=parameter['images'], neural_net_parameters=x),
                                      parameter['labels'])

    return loss_function_of_the_algorithm


def extract_data_from_dataloader(dataloader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    all_training_images, all_training_labels = [], []
    for img, label in dataloader:
        all_training_images.extend(img.to(torch.get_default_device()))  # torch.Transforms ignores default_device
        all_training_labels.extend(label)
    all_training_images = torch.stack(all_training_images)
    all_training_labels = torch.stack(all_training_labels)
    return all_training_images, all_training_labels


def split_data() -> Tuple[dict, dict, dict, dict]:
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,)),     # MNIST
         Resize((20, 20))
         ])

    batch_size = 100
    # There are 50.000 datapoints in the training set, so we can split it into subsets of size 16.500
    # to sample from for prior, train, and validation.
    n_train = 16500
    training_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0)

    # There are 10.000 datapoints in the test set, from which we can sample just like that.
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)

    all_training_images, all_training_labels = extract_data_from_dataloader(training_loader)
    all_test_images, all_test_labels = extract_data_from_dataloader(test_loader)

    prior_data = {'images': all_training_images[:n_train],
                  'labels': all_training_labels[:n_train]}
    train_data = {'images': all_training_images[n_train:2 * n_train],
                  'labels': all_training_labels[n_train:2 * n_train]}
    validation_data = {'images': all_training_images[2 * n_train:3 * n_train],
                       'labels': all_training_labels[2 * n_train:3 * n_train]}
    test_data = {'images': all_test_images,
                 'labels': all_test_labels}

    return prior_data, train_data, test_data, validation_data


def create_parameter(images: torch.Tensor, labels: torch.Tensor) -> dict:
    return {'images': images, 'labels': labels, 'optimal_loss': torch.tensor(0.0)}


def get_parameters(number_of_datapoints_per_dataset: dict) -> dict:

    n_prior, n_train, n_test, n_val = check_and_extract_number_of_datapoints(number_of_datapoints_per_dataset)
    prior_data, train_data, test_data, validation_data = split_data()
    # Note that the result only applies to the full-batch setting. Thus, one cannot use very large data sets here.
    size_of_each_dataset = 250
    parameters = {'prior': [], 'train': [], 'test': [], 'validation': []}
    for data, number_of_functions, name in [(prior_data, n_prior, 'prior'),
                                            (train_data, n_train, 'train'),
                                            (test_data, n_test, 'test'),
                                            (validation_data, n_val, 'validation')]:
        for _ in range(number_of_functions):
            idx = torch.randint(low=0, high=len(data['labels']), size=(size_of_each_dataset,))
            images, labels = data['images'][idx], data['labels'][idx]
            parameters[name].append(create_parameter(images=images, labels=labels))

    return parameters


def get_data(neural_network: NeuralNetworkForLearning,
             number_of_datapoints_per_dataset: dict) -> Tuple[Callable, Callable, dict]:

    parameters = get_parameters(number_of_datapoints_per_dataset)
    loss_of_neural_network = get_loss_of_neural_network()
    loss_of_algorithm = get_loss_of_algorithm(neural_network, loss_of_neural_network)

    return loss_of_algorithm, loss_of_neural_network, parameters
