import torch
from pathlib import Path
from experiments.quadratics.data_generation import get_data
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import (
    PacBayesOptimizationAlgorithm)
from experiments.quadratics.algorithm import Quadratics
from describing_property.reduction_property import instantiate_reduction_property_with
from algorithms.heavy_ball import HeavyBallWithFriction
from natural_parameters.natural_parameters import evaluate_natural_parameters_at
from sufficient_statistics.sufficient_statistics import evaluate_sufficient_statistics
from classes.Constraint.class_ProbabilisticConstraint import ProbabilisticConstraint
from classes.Constraint.class_Constraint import create_list_of_constraints_from_functions
import numpy as np
import pickle


def get_number_of_datapoints():
    # TODO: Set number of samples back to 250 each
    return {'prior': 25, 'train': 25, 'test': 25, 'validation': 250}


def create_folder_for_storing_data(path_of_experiment):
    savings_path = path_of_experiment + "/data/"
    Path(savings_path).mkdir(parents=True, exist_ok=True)
    return savings_path


def create_parametric_loss_functions_from_parameters(template_loss_function, parameters):
    loss_functions = {
        'prior': [ParametricLossFunction(function=template_loss_function, parameter=p) for p in parameters['prior']],
        'train': [ParametricLossFunction(function=template_loss_function, parameter=p) for p in parameters['train']],
        'test': [ParametricLossFunction(function=template_loss_function, parameter=p) for p in parameters['test']],
        'validation': [ParametricLossFunction(function=template_loss_function, parameter=p)
                       for p in parameters['validation']],
    }
    return loss_functions


def get_initial_state(dim):
    return torch.zeros((2, dim))


def get_parameters_of_estimation():
    return {'quantile_distance': 0.075, 'quantiles': (0.01, 0.99), 'probabilities': (0.95, 1.0)}


def get_update_parameters():
    return {'num_iter_print_update': 1000,
            'with_print': True,
            'bins': [1e6, 1e4, 1e2, 1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10][::-1]}


def get_sampling_parameters(maximal_number_of_iterations):
    length_trajectory = 1
    restart_probability = length_trajectory / maximal_number_of_iterations
    return {'lr': torch.tensor(1e-6),
            'length_trajectory': length_trajectory,
            'with_restarting': True,
            'restart_probability': restart_probability,
            # TODO: Set number of samples to 100
            'num_samples': 5,
            'num_iter_burnin': 0}


def get_fitting_parameters(maximal_number_of_iterations):
    length_trajectory = 1
    restart_probability = length_trajectory / maximal_number_of_iterations
    return {'restart_probability': restart_probability,
            'length_trajectory': length_trajectory,
            # TODO: Rename n_max to number_of_training_iterations
            'n_max': int(200e3),
            'lr': 1e-4,
            'num_iter_update_stepsize': int(20e3),
            'factor_stepsize_update': 0.5}


def get_initialization_parameters():
    return {'lr': 1e-3, 'num_iter_max': 1000, 'num_iter_print_update': 200, 'num_iter_update_stepsize': 200,
            'with_print': True}


def get_describing_property():
    return instantiate_reduction_property_with(factor=1.0, exponent=0.5)


def get_constraint_parameters(number_of_training_iterations):
    describing_property, _, _ = get_describing_property()
    return {'describing_property': describing_property,
            'num_iter_update_constraint': int(number_of_training_iterations // 4)}


def get_pac_bayes_parameters(sufficient_statistics):
    return {'sufficient_statistics': sufficient_statistics,
            'natural_parameters': evaluate_natural_parameters_at,
            'covering_number': torch.tensor(75000),
            'epsilon': torch.tensor(0.05),
            # TODO: Rename n_max to maximal_number_of_iterations
            'n_max': 350}


def get_constraint(parameters_of_estimation, loss_functions_for_constraint):
    describing_property, _, _ = get_describing_property()
    list_of_constraints = create_list_of_constraints_from_functions(describing_property=describing_property,
                                                                    list_of_functions=loss_functions_for_constraint)
    probabilistic_constraint = ProbabilisticConstraint(list_of_constraints=list_of_constraints,
                                                       parameters_of_estimation=parameters_of_estimation)
    return probabilistic_constraint.create_constraint()


def compute_constants_for_sufficient_statistics(loss_functions_for_training, initial_state):
    _, _, empirical_second_moment = get_describing_property()
    return empirical_second_moment(list_of_loss_functions=loss_functions_for_training,
                                   point=initial_state[-1].flatten()) / len(loss_functions_for_training)


def get_sufficient_statistics(template_for_loss_function, constants):

    _, convergence_risk_constraint, _ = get_describing_property()

    def sufficient_statistics(optimization_algorithm, parameter, probability):
        return evaluate_sufficient_statistics(optimization_algorithm=optimization_algorithm,
                                              parameter_of_loss_function=parameter,
                                              template_for_loss_function=template_for_loss_function,
                                              constants=constants,
                                              convergence_risk_constraint=convergence_risk_constraint,
                                              convergence_probability=probability)

    return sufficient_statistics


def get_algorithm_for_learning(loss_function_for_algorithm, loss_functions, dimension_of_hyperparameters):

    initial_state = get_initial_state(dim=dimension_of_hyperparameters)
    parameters_of_estimation = get_parameters_of_estimation()
    constraint = get_constraint(
        parameters_of_estimation=parameters_of_estimation,
        loss_functions_for_constraint=loss_functions['validation']
    )
    constants = compute_constants_for_sufficient_statistics(loss_functions_for_training=loss_functions['train'],
                                                            initial_state=initial_state)
    sufficient_statistics = get_sufficient_statistics(template_for_loss_function=loss_function_for_algorithm,
                                                      constants=constants)
    algorithm_for_learning = PacBayesOptimizationAlgorithm(
        initial_state=initial_state,
        implementation=Quadratics(dim=dimension_of_hyperparameters),
        loss_function=loss_functions['prior'][0],
        pac_parameters=get_pac_bayes_parameters(sufficient_statistics),
        constraint=constraint
    )
    return algorithm_for_learning


def get_baseline_algorithm(loss_function, smoothness_constant, strong_convexity_constant, dim):

    initial_state = get_initial_state(dim=dim)
    alpha = 4/((torch.sqrt(smoothness_constant) + torch.sqrt(strong_convexity_constant)) ** 2)
    beta = ((torch.sqrt(smoothness_constant) - torch.sqrt(strong_convexity_constant)) /
            (torch.sqrt(smoothness_constant) + torch.sqrt(strong_convexity_constant))) ** 2

    std_algo = OptimizationAlgorithm(
        initial_state=initial_state,
        implementation=HeavyBallWithFriction(alpha=alpha, beta=beta),
        loss_function=loss_function
    )
    return std_algo


def set_up_and_train_algorithm(path_of_experiment):

    savings_path = create_folder_for_storing_data(path_of_experiment)
    parameters, loss_function_of_algorithm, mu_min, L_max, dim = get_data(get_number_of_datapoints())
    loss_functions = create_parametric_loss_functions_from_parameters(
        template_loss_function=loss_function_of_algorithm, parameters=parameters)
    baseline_algorithm = get_baseline_algorithm(
        loss_function=loss_functions['prior'][0], smoothness_constant=L_max, strong_convexity_constant=mu_min, dim=dim)
    algorithm_for_learning = get_algorithm_for_learning(
        loss_function_for_algorithm=loss_function_of_algorithm, loss_functions=loss_functions,
        dimension_of_hyperparameters=dim)
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

    np.save(savings_path + 'pac_bound', pac_bound.numpy())
    np.save(savings_path + 'initialization', algorithm_for_learning.initial_state.clone().numpy())
    np.save(savings_path + 'number_of_iterations', algorithm_for_learning.n_max)
    with open(savings_path + 'parameters_problem', 'wb') as file:
        pickle.dump(parameters, file)

    parameters_of_estimation = get_parameters_of_estimation()
    with open(savings_path + 'parameters_of_estimation', 'wb') as file:
        pickle.dump(parameters_of_estimation, file)
    with open(savings_path + 'samples', 'wb') as file:
        pickle.dump(state_dict_samples_prior, file)
    with open(savings_path + 'best_sample', 'wb') as file:
        pickle.dump(algorithm_for_learning.implementation.state_dict(), file)
