from typing import Callable, List, Tuple, Dict
import torch
import numpy as np
from numpy.typing import NDArray
import pickle
from algorithms.nesterov_accelerated_gradient_descent import NesterovAcceleratedGradient
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import LossFunction, ParametricLossFunction
from classes.Constraint.class_Constraint import create_list_of_constraints_from_functions, Constraint
from classes.Constraint.class_ProbabilisticConstraint import ProbabilisticConstraint
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import \
    PacBayesOptimizationAlgorithm
from experiments.image_processing.algorithm import ConvNet
from experiments.image_processing.data_generation import get_image_height_and_width, get_data
from exponential_family.describing_property.reduction_property import instantiate_reduction_property_with
from pathlib import Path
from exponential_family.natural_parameters.natural_parameters import evaluate_natural_parameters_at
from exponential_family.sufficient_statistics.sufficient_statistics import evaluate_sufficient_statistics


def create_folder_for_storing_data(path_of_experiment: str) -> str:
    savings_path = path_of_experiment + "/data/"
    Path(savings_path).mkdir(parents=True, exist_ok=True)
    return savings_path


def get_number_of_datapoints() -> dict:
    return {'prior': 250, 'train': 250, 'test': 250, 'validation': 250}


def get_parameters_of_estimation() -> dict:
    return {'quantile_distance': 0.075, 'quantiles': (0.01, 0.99), 'probabilities': (0.95, 1.0)}


def get_update_parameters() -> dict:
    return {'num_iter_print_update': 1000,
            'with_print': True,
            'bins': [1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0][::-1]}


def get_sampling_parameters(maximal_number_of_iterations: int) -> dict:
    length_trajectory = 1
    restart_probability = length_trajectory / maximal_number_of_iterations
    return {'lr': torch.tensor(1e-6),
            'length_trajectory': length_trajectory,
            'with_restarting': True,
            'restart_probability': restart_probability,
            'num_samples': 100,
            'num_iter_burnin': 0}


def get_fitting_parameters(maximal_number_of_iterations: int) -> dict:
    length_trajectory = 1
    restart_probability = length_trajectory / maximal_number_of_iterations
    return {'restart_probability': restart_probability,
            'length_trajectory': length_trajectory,
            # TODO: Rename n_max to number_of_training_iterations
            'n_max': int(200e3),
            'lr': 1e-4,
            'num_iter_update_stepsize': int(20e3),
            'factor_stepsize_update': 0.5}


def get_initialization_parameters() -> dict:
    return {'lr': 1e-3, 'num_iter_max': 1000, 'num_iter_print_update': 200, 'num_iter_update_stepsize': 200,
            'with_print': True}


def get_constraint_parameters(number_of_training_iterations: int) -> dict:
    describing_property, _, _ = get_describing_property()
    return {'describing_property': describing_property,
            'num_iter_update_constraint': int(number_of_training_iterations // 4)}


def get_pac_bayes_parameters(sufficient_statistics: Callable) -> dict:
    return {'sufficient_statistics': sufficient_statistics,
            'natural_parameters': evaluate_natural_parameters_at,
            'covering_number': torch.tensor(75000),
            'epsilon': torch.tensor(0.05),
            # TODO: Rename n_max to maximal_number_of_iterations
            'n_max': 25}


def get_describing_property() -> Tuple[Callable, Callable, Callable]:
    return instantiate_reduction_property_with(factor=0.2, exponent=1.)


def get_dimension_of_optimization_variable() -> int:
    image_height, image_width = get_image_height_and_width()
    return image_height * image_width


def get_initial_states() -> Tuple[torch.Tensor, torch.Tensor]:
    dim = get_dimension_of_optimization_variable()
    initial_state_baseline_algorithm = torch.zeros((3, dim))
    initial_state_learned_algorithm = torch.zeros((2, dim))
    return initial_state_baseline_algorithm, initial_state_learned_algorithm


def get_baseline_algorithm(loss_function_of_algorithm: Callable, smoothness_parameter: float):
    initial_state_baseline_algorithm, _ = get_initial_states()
    baseline_algorith = OptimizationAlgorithm(
        implementation=NesterovAcceleratedGradient(alpha=torch.tensor(1/smoothness_parameter)),
        initial_state=initial_state_baseline_algorithm,
        loss_function=LossFunction(function=loss_function_of_algorithm)
    )
    return baseline_algorith


def get_constraint(loss_functions_for_constraint: List[LossFunction]) -> Constraint:
    parameters_of_estimation = get_parameters_of_estimation()
    describing_property, _, _ = get_describing_property()
    list_of_constraints = create_list_of_constraints_from_functions(describing_property=describing_property,
                                                                    list_of_functions=loss_functions_for_constraint)
    probabilistic_constraint = ProbabilisticConstraint(list_of_constraints=list_of_constraints,
                                                       parameters_of_estimation=parameters_of_estimation)
    return probabilistic_constraint.create_constraint()


def get_sufficient_statistics(constants: torch.Tensor) -> Callable:

    _, convergence_risk_constraint, _ = get_describing_property()

    def sufficient_statistics(optimization_algorithm, loss_function, probability):
        return evaluate_sufficient_statistics(optimization_algorithm=optimization_algorithm,
                                              loss_function=loss_function,
                                              constants=constants,
                                              convergence_risk_constraint=convergence_risk_constraint,
                                              convergence_probability=probability)

    return sufficient_statistics


def compute_constants_for_sufficient_statistics(loss_functions_for_training: List[LossFunction]) -> torch.Tensor:
    _, initial_state = get_initial_states()
    _, _, empirical_second_moment = get_describing_property()
    return empirical_second_moment(list_of_loss_functions=loss_functions_for_training,
                                   point=initial_state[-1].flatten()) / len(loss_functions_for_training)


def get_algorithm_for_learning(loss_functions: Dict) -> PacBayesOptimizationAlgorithm:

    _, initial_state_learned_algorithm = get_initial_states()
    image_height, image_width = get_image_height_and_width()
    constraint = get_constraint(loss_functions_for_constraint=loss_functions['validation'])
    constants = compute_constants_for_sufficient_statistics(loss_functions_for_training=loss_functions['train'])
    sufficient_statistics = get_sufficient_statistics(constants=constants)
    algorithm_for_learning = PacBayesOptimizationAlgorithm(
        initial_state=initial_state_learned_algorithm,
        implementation=ConvNet(img_height=image_height, img_width=image_width, kernel_size=3),
        loss_function=loss_functions['prior'][0],
        pac_parameters=get_pac_bayes_parameters(sufficient_statistics),
        constraint=constraint
    )
    return algorithm_for_learning


def create_parametric_loss_functions_from_parameters(template_loss_function: Callable, parameters: dict) -> Dict:

    loss_functions = {
        name: [ParametricLossFunction(function=template_loss_function, parameter=p)
               for p in parameters[name]] for name in list(parameters.keys())
    }
    return loss_functions


def save_data(savings_path: str,
              smoothness_parameter: float,
              pac_bound: NDArray,
              initialization_learned_algorithm: NDArray,
              initialization_baseline_algorithm: NDArray,
              number_of_iterations: int,
              parameters: dict,
              samples_prior: List[dict],
              best_sample: dict) -> None:

    np.save(savings_path + 'smoothness_parameter', smoothness_parameter)
    np.save(savings_path + 'pac_bound', pac_bound)
    np.save(savings_path + 'initialization_learned_algorithm', initialization_learned_algorithm)
    np.save(savings_path + 'initialization_baseline_algorithm', initialization_baseline_algorithm)
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


def set_up_and_train_algorithm(path_of_experiment: str, path_to_images: str) -> None:

    number_of_datapoints_per_dataset = get_number_of_datapoints()
    parameters, loss_function_of_algorithm, smoothness_parameter = get_data(
        path_to_images=path_to_images, number_of_datapoints_per_dataset=number_of_datapoints_per_dataset, device='cpu')
    loss_functions = create_parametric_loss_functions_from_parameters(
        template_loss_function=loss_function_of_algorithm, parameters=parameters)
    baseline_algorithm = get_baseline_algorithm(
        loss_function_of_algorithm=loss_function_of_algorithm, smoothness_parameter=smoothness_parameter)
    algorithm_for_learning = get_algorithm_for_learning(loss_functions=loss_functions)

    algorithm_for_learning.initialize_with_other_algorithm(other_algorithm=baseline_algorithm,
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

    savings_path = create_folder_for_storing_data(path_of_experiment)
    save_data(savings_path=savings_path, smoothness_parameter=smoothness_parameter,
              pac_bound=pac_bound.numpy(),
              initialization_learned_algorithm=algorithm_for_learning.initial_state.clone().numpy(),
              initialization_baseline_algorithm=baseline_algorithm.initial_state.clone().numpy(),
              number_of_iterations=algorithm_for_learning.n_max, parameters=parameters,
              samples_prior=state_dict_samples_prior, best_sample=algorithm_for_learning.implementation.state_dict())
