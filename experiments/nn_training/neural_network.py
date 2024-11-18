import torch
import torch.nn as nn
import torch.nn.functional as f
from typing import Callable, Tuple, List


def polynomial_features(x: torch.Tensor, degree: int) -> torch.Tensor:
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, degree + 1)], 1).reshape((-1, degree))


class NeuralNetworkForStandardTraining(nn.Module):

    def __init__(self, degree: int):
        super().__init__()
        self.degree = degree
        self.fc1 = nn.Linear(self.degree, 10 * self.degree)
        self.fc2 = nn.Linear(10 * self.degree, 1)

    def get_shape_parameters(self) -> List:
        return [p.size() for p in self.parameters() if p.requires_grad]

    def get_dimension_of_hyperparameters(self) -> int:
        return sum([torch.prod(torch.tensor(s)).item() for s in self.get_shape_parameters()])

    def load_parameters_from_tensor(self, tensor: torch.Tensor) -> None:
        counter = 0
        # If the parameter is updated by the (learned) optimization algorithm, then they have corresponding entries in
        # the tensor, which should be loaded into the template.
        for param in self.parameters():
            if param.requires_grad:
                cur_size = torch.prod(torch.tensor(param.size()))
                cur_shape = param.shape
                param.data = tensor[counter:counter + cur_size].reshape(cur_shape)
                counter += cur_size

    def transform_parameters_to_tensor(self) -> torch.Tensor:
        all_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                all_params.append(param.flatten())
        return torch.concat(all_params, dim=0).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = f.relu(self.fc1(polynomial_features(x=x, degree=self.degree)))
        res = self.fc2(x)
        return res


def train_model(net: NeuralNetworkForStandardTraining,
                data: dict,
                criterion: Callable,
                n_it: int,
                lr: float) -> Tuple[nn.Module, list, list]:

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    iterates, losses = [], []

    for i in range(n_it + 1):
        iterates.append(net.transform_parameters_to_tensor())
        optimizer.zero_grad()
        loss = criterion(net(data['x_values']), data['y_values'])
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    return net, losses, iterates


class NeuralNetworkForLearning(nn.Module):

    def __init__(self, degree: int, shape_parameters: list) -> None:
        super().__init__()
        self.degree = degree
        self.shape_param = shape_parameters
        self.dim_param = [torch.prod(torch.tensor(p)) for p in shape_parameters]

    def forward(self, x: torch.Tensor, neural_net_parameters: torch.Tensor) -> torch.Tensor:
        x = polynomial_features(x=x, degree=self.degree)

        # From the neural_net_parameters (prediction of optimization algorithm), extract the weights of the neural
        # network into the corresponding torch.nn.functional-functions. Then, perform the prediction in the usual way,
        # that is, by calling them successively.
        c = 0
        for i in range(0, len(self.dim_param), 2):

            # Extract weights and biases and reshape them correctly using self.shape_param
            weights = neural_net_parameters[c:c+self.dim_param[i]]
            weights = weights.reshape(self.shape_param[i])
            bias = neural_net_parameters[c+self.dim_param[i]:c+self.dim_param[i]+self.dim_param[i+1]]
            bias = bias.reshape(self.shape_param[i+1])

            x = f.linear(input=x, weight=weights, bias=bias)
            c += self.dim_param[i] + self.dim_param[i+1]
            if len(self.shape_param) > 2 and (i+2 < len(self.dim_param)):
                x = f.relu(x)
        return x
