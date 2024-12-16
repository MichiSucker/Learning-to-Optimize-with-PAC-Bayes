from typing import Callable, Tuple, List
from numpy.typing import NDArray
import torch
from classes.LossFunction.class_LossFunction import LossFunction
from experiments.mnist.data_generation import get_data
from experiments.mnist.neural_network import NeuralNetworkForLearning, NeuralNetworkForStandardTraining
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from algorithms.gradient_descent import GradientDescent
from experiments.mnist.algorithm import MnistOptimizer
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import (
    PacBayesOptimizationAlgorithm)
from exponential_family.describing_property.reduction_property import instantiate_reduction_property_with
from classes.Constraint.class_ProbabilisticConstraint import ProbabilisticConstraint
from classes.Constraint.class_Constraint import create_list_of_constraints_from_functions, Constraint
from exponential_family.sufficient_statistics.sufficient_statistics import evaluate_sufficient_statistics
from exponential_family.natural_parameters.natural_parameters import evaluate_natural_parameters_at
from pathlib import Path
import pickle
import numpy as np


def get_number_of_datapoints() -> dict:
    return {'prior': 25, 'train': 50, 'test': 50, 'validation': 250}


def get_initialization_parameters() -> dict:
    return {'lr': 1e-3, 'num_iter_max': 100, 'num_iter_print_update': 200, 'num_iter_update_stepsize': 200,
            'with_print': True}


def get_fitting_parameters(maximal_number_of_iterations: int) -> dict:
    length_trajectory = 1
    restart_probability = length_trajectory / maximal_number_of_iterations
    return {'restart_probability': restart_probability,
            'length_trajectory': length_trajectory,
            # TODO: Rename n_max to number_of_training_iterations
            'n_max': int(10e3),
            'lr': 1e-4,
            'num_iter_update_stepsize': int(10e3),
            'factor_stepsize_update': 0.5}


def get_sampling_parameters(maximal_number_of_iterations: int) -> dict:
    length_trajectory = 1
    restart_probability = length_trajectory / maximal_number_of_iterations
    return {'lr': torch.tensor(1e-6),
            'length_trajectory': length_trajectory,
            'with_restarting': True,
            'restart_probability': restart_probability,
            'num_samples': 10,
            'num_iter_burnin': 0}


def get_update_parameters() -> dict:
    return {'num_iter_print_update': 250,
            'with_print': True,
            'bins': [1e6, 1e3, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-9, 1e-12, 1e-15][::-1]}


def get_parameters_of_estimation() -> dict:
    return {'quantile_distance': 0.075, 'quantiles': (0.01, 0.99), 'probabilities': (0.95, 1.0)}


def get_constraint_parameters(number_of_training_iterations: int) -> dict:
    describing_property, _, _ = get_describing_property()
    return {'describing_property': describing_property,
            'num_iter_update_constraint': int(number_of_training_iterations // 2)}


def get_pac_bayes_parameters(sufficient_statistics: Callable) -> dict:
    return {'sufficient_statistics': sufficient_statistics,
            'natural_parameters': evaluate_natural_parameters_at,
            'covering_number': torch.tensor(75000),
            'epsilon': torch.tensor(0.05),
            # TODO: Rename n_max to maximal_number_of_iterations
            'n_max': 200}


def instantiate_neural_networks() -> Tuple[NeuralNetworkForStandardTraining, NeuralNetworkForLearning]:
    neural_network_for_std_training = NeuralNetworkForStandardTraining()
    neural_network_for_learning = NeuralNetworkForLearning()
    return neural_network_for_std_training, neural_network_for_learning


def create_parametric_loss_functions_from_parameters(template_loss_function: Callable, parameters: dict) -> dict:
    loss_functions = {
        'prior': [ParametricLossFunction(function=template_loss_function,
                                         parameter=p) for p in parameters['prior']],
        'train': [ParametricLossFunction(function=template_loss_function,
                                         parameter=p) for p in parameters['train']],
        'test': [ParametricLossFunction(function=template_loss_function,
                                        parameter=p) for p in parameters['test']],
        'validation': [ParametricLossFunction(function=template_loss_function,
                                              parameter=p) for p in parameters['validation']],
    }
    return loss_functions


def get_algorithm_for_initialization(initial_state_for_std_algorithm: torch.Tensor,
                                     loss_function: LossFunction) -> OptimizationAlgorithm:
    alpha = torch.tensor(1e-5)
    return OptimizationAlgorithm(initial_state=initial_state_for_std_algorithm,
                                 implementation=GradientDescent(alpha=alpha),
                                 loss_function=loss_function)


def get_initial_state(dim: int) -> torch.Tensor:
    # Note that it is important to keep the same initial point: Here, the algorithm only gets on a single starting
    # point, so it depends on this concrete initialization.
    # torch.manual_seed(0)
    n = 2
    x_0 = torch.zeros(n * dim).reshape((n, -1))
    x_0[-1] = torch.randn(dim).reshape((1, dim))
    return x_0


def get_describing_property() -> Tuple[Callable, Callable, Callable]:
    return instantiate_reduction_property_with(factor=0.5, exponent=0.5)


def get_constraint(parameters_of_estimation: dict,
                   loss_functions_for_constraint: List[LossFunction]) -> Constraint:
    describing_property, _, _ = get_describing_property()
    list_of_constraints = create_list_of_constraints_from_functions(describing_property=describing_property,
                                                                    list_of_functions=loss_functions_for_constraint)
    probabilistic_constraint = ProbabilisticConstraint(list_of_constraints=list_of_constraints,
                                                       parameters_of_estimation=parameters_of_estimation)
    return probabilistic_constraint.create_constraint()


def compute_constants_for_sufficient_statistics(loss_functions_for_training: List[LossFunction],
                                                initial_state: torch.Tensor) -> torch.Tensor:
    _, _, empirical_second_moment = get_describing_property()
    return empirical_second_moment(list_of_loss_functions=loss_functions_for_training,
                                   point=initial_state[-1].flatten()) / len(loss_functions_for_training)


def get_sufficient_statistics(constants: torch.Tensor) -> Callable:

    _, convergence_risk_constraint, _ = get_describing_property()

    def sufficient_statistics(optimization_algorithm, loss_function, probability):
        return evaluate_sufficient_statistics(optimization_algorithm=optimization_algorithm,
                                              loss_function=loss_function,
                                              constants=constants,
                                              convergence_risk_constraint=convergence_risk_constraint,
                                              convergence_probability=probability)

    return sufficient_statistics


def instantiate_algorithm_for_learning(loss_functions: dict,
                                       dimension_of_hyperparameters: int) -> PacBayesOptimizationAlgorithm:

    initial_state = get_initial_state(dim=dimension_of_hyperparameters)
    parameters_of_estimation = get_parameters_of_estimation()
    constraint = get_constraint(
        parameters_of_estimation=parameters_of_estimation,
        loss_functions_for_constraint=loss_functions['validation']
    )
    constants = compute_constants_for_sufficient_statistics(loss_functions_for_training=loss_functions['train'],
                                                            initial_state=initial_state)
    sufficient_statistics = get_sufficient_statistics(constants=constants)
    algorithm_for_learning = PacBayesOptimizationAlgorithm(
        initial_state=initial_state,
        implementation=MnistOptimizer(dim=dimension_of_hyperparameters),
        loss_function=loss_functions['prior'][0],
        pac_parameters=get_pac_bayes_parameters(sufficient_statistics),
        constraint=constraint
    )
    return algorithm_for_learning


def create_folder_for_storing_data(path_of_experiment: str) -> str:
    savings_path = path_of_experiment + "/data/"
    Path(savings_path).mkdir(parents=True, exist_ok=True)
    return savings_path


def set_up_and_train_algorithm(path_of_experiment: str) -> None:

    # For us, GPU does not really give acceleration, but causes some problems with the current implementation (results
    # only apply to full-batch).
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.double)
    print(f"Using PyTorch with Version {torch.__version__}.")
    print(f"Default Tensor Type is set to {torch.get_default_dtype()}")
    print(f"Default Device is set to {torch.tensor([1.2, 3]).device}")

    savings_path = create_folder_for_storing_data(path_of_experiment)

    neural_network_for_std_training, neural_network_for_learning = instantiate_neural_networks()
    loss_function_for_algorithm, loss_function_for_neural_network, parameters = get_data(
        neural_network=neural_network_for_learning, number_of_datapoints_per_dataset=get_number_of_datapoints())
    loss_functions = create_parametric_loss_functions_from_parameters(
        template_loss_function=loss_function_for_algorithm, parameters=parameters)

    algorithm_for_learning = instantiate_algorithm_for_learning(
        loss_functions=loss_functions,
        dimension_of_hyperparameters=neural_network_for_std_training.get_dimension_of_hyperparameters())
    algorithm_for_initialization = get_algorithm_for_initialization(
        initial_state_for_std_algorithm=algorithm_for_learning.initial_state[-1].reshape((1, -1)),
        loss_function=loss_functions['prior'][0]
    )

    algorithm_for_learning.initialize_with_other_algorithm(other_algorithm=algorithm_for_initialization,
                                                           loss_functions=loss_functions['prior'],
                                                           parameters_of_initialization=get_initialization_parameters())

    fitting_parameters = get_fitting_parameters(maximal_number_of_iterations=algorithm_for_learning.n_max)
    sampling_parameters = get_sampling_parameters(maximal_number_of_iterations=algorithm_for_learning.n_max)
    constraint_parameters = get_constraint_parameters(number_of_training_iterations=fitting_parameters['n_max'])
    update_parameters = get_update_parameters()
    pac_bound, state_dict_samples_prior = algorithm_for_learning.pac_bayes_fit(
        loss_functions_prior=loss_functions['prior'],
        loss_functions_train=loss_functions['train'],
        fitting_parameters=fitting_parameters,
        sampling_parameters=sampling_parameters,
        constraint_parameters=constraint_parameters,
        update_parameters=update_parameters
    )

    save_data(savings_path=savings_path,
              pac_bound=pac_bound.numpy(),
              initial_state=algorithm_for_learning.initial_state.clone().numpy(),
              number_of_iterations=algorithm_for_learning.n_max,
              parameters=parameters,
              samples_prior=state_dict_samples_prior,
              best_sample=algorithm_for_learning.implementation.state_dict())


def save_data(savings_path,
              pac_bound: NDArray,
              initial_state: NDArray,
              number_of_iterations: int,
              parameters: dict,
              samples_prior: List[dict],
              best_sample: dict):
    np.save(savings_path + 'pac_bound', pac_bound)
    np.save(savings_path + 'initialization', initial_state)
    np.save(savings_path + 'number_of_iterations', number_of_iterations)
    with open(savings_path + 'parameters_problem', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(parameters, file)

    parameters_of_estimation = get_parameters_of_estimation()
    with open(savings_path + 'parameters_of_estimation', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(parameters_of_estimation, file)
    with open(savings_path + 'samples', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(samples_prior, file)
    with open(savings_path + 'best_sample', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(best_sample, file)
