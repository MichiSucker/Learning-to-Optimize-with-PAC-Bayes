import torch
import torch.nn as nn
import torch.nn.functional as f
from typing import List
import copy
from typing import Callable, Tuple


class NeuralNetworkForStandardTraining(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.fc1 = nn.Linear(12 * 3 * 3, 80)
        self.fc2 = nn.Linear(80, 40)
        self.fc3 = nn.Linear(40, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)     # Note that we use Cross-Entropy Loss, such that we do not need Softmax-activation here.
        return x

    def transform_tensor_into_state_dict(self, tensor: torch.Tensor) -> dict:
        c = 0
        state_dict = copy.deepcopy(self.state_dict())
        for p in self.state_dict():
            total_amount = torch.prod(torch.tensor(state_dict[p].shape))
            state_dict[p] = tensor[c:c + total_amount].reshape(state_dict[p].shape)
            c += total_amount
        return state_dict

    def load_tensor_into_state_dict(self, tensor: torch.Tensor) -> None:
        state_dict = self.transform_tensor_into_state_dict(tensor)
        self.load_state_dict(state_dict)

    def transform_state_dict_to_tensor(self) -> torch.Tensor:
        trainable_parameters = []
        for p in self.state_dict():
            trainable_parameters.extend(self.state_dict()[p].flatten())
        return torch.tensor(trainable_parameters)

    def get_shape_parameters(self) -> List[torch.Size]:
        return [p.size() for p in self.parameters() if p.requires_grad]

    def get_dimension_of_hyperparameters(self) -> int:
        return sum([torch.prod(torch.tensor(s)).item() for s in self.get_shape_parameters()])


def train_model(net: NeuralNetworkForStandardTraining,
                data: dict,
                criterion: Callable,
                n_it: int,
                lr: float) -> Tuple[NeuralNetworkForStandardTraining, list, list]:

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    iterates, losses = [], []

    for i in range(n_it + 1):
        iterates.append(net.transform_state_dict_to_tensor())
        optimizer.zero_grad()
        loss = criterion(net(data['images']), data['labels'])
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    return net, losses, iterates


class NeuralNetworkForLearning(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.network = NeuralNetworkForStandardTraining()

    def forward(self, x: torch.Tensor, neural_net_parameters: torch.Tensor) -> torch.Tensor:

        # From the neural_net_parameters (prediction of optimization algorithm), extract the weights of the neural
        # network into the corresponding torch.nn.functional-functions. Then, perform the prediction in the usual way,
        # that is, by calling them successively.
        state_dict = self.network.transform_tensor_into_state_dict(tensor=neural_net_parameters)
        return torch.func.functional_call(self.network, state_dict, x)
