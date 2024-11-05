import torch
import pickle
import numpy as np
from pathlib import Path
from experiments.lasso.data_generation import get_data, get_dimensions
from classes.LossFunction.derived_classes.NonsmoothParametricLossFunction.\
    subclass_NonsmoothParametricLossFunction import NonsmoothParametricLossFunction
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import (
    PacBayesOptimizationAlgorithm)
from classes.Constraint.class_ProbabilisticConstraint import ProbabilisticConstraint
from classes.Constraint.class_Constraint import create_list_of_constraints_from_functions
from experiments.lasso.algorithm import SparsityNet
from algorithms.fista import FISTA
from describing_property.reduction_property import instantiate_reduction_property_with
from natural_parameters.natural_parameters import evaluate_natural_parameters_at
from sufficient_statistics.sufficient_statistics import evaluate_sufficient_statistics


def create_folder_for_storing_data(path_of_experiment):
    savings_path = path_of_experiment + "/data/"
    Path(savings_path).mkdir(parents=True, exist_ok=True)
    return savings_path


def get_number_of_datapoints():
    # TODO: Change to 250 each.
    return {'prior': 250, 'train': 250, 'test': 250, 'validation': 250}


def get_parameters_of_estimation():
    return {'quantile_distance': 0.075, 'quantiles': (0.01, 0.99), 'probabilities': (0.95, 1.0)}


def get_update_parameters():
    return {'num_iter_print_update': 1000,
            'with_print': True,
            'bins': [1e5, 5e4, 1e4, 5e3, 1e3, 5e2, 1e2, 5e1, 1e1][::-1]}


def get_sampling_parameters(maximal_number_of_iterations):
    length_trajectory = 1
    restart_probability = length_trajectory / maximal_number_of_iterations
    return {'lr': torch.tensor(1e-6),
            'length_trajectory': length_trajectory,
            'with_restarting': True,
            'restart_probability': restart_probability,
            # TODO: Change num_samples to 100
            'num_samples': 5,
            'num_iter_burnin': 0}


def get_fitting_parameters(maximal_number_of_iterations):
    length_trajectory = 1
    restart_probability = length_trajectory / maximal_number_of_iterations
    return {'restart_probability': restart_probability,
            'length_trajectory': length_trajectory,
            # TODO: Rename n_max to number_of_training_iterations
            # TODO: Change n_max to 200e3
            'n_max': int(100e3),
            'lr': 1e-4,
            'num_iter_update_stepsize': int(20e3),
            'factor_stepsize_update': 0.5}


def get_initialization_parameters():
    return {'lr': 1e-3, 'num_iter_max': 1000, 'num_iter_print_update': 200, 'num_iter_update_stepsize': 200,
            'with_print': True}


def get_describing_property():
    return instantiate_reduction_property_with(factor=1.25, exponent=0.5)


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


def get_initial_states():
    _, dimension_optimization_variable = get_dimensions()
    init_distribution = torch.distributions.Normal(0, 1)
    initial_state_fista = init_distribution.sample((3, dimension_optimization_variable))
    return initial_state_fista, initial_state_fista[1:, :].clone()


def get_algorithm_for_learning(loss_function_for_algorithm, loss_functions):

    _, initial_state_learned_algorithm = get_initial_states()
    parameters_of_estimation = get_parameters_of_estimation()
    constraint = get_constraint(
        parameters_of_estimation=parameters_of_estimation,
        loss_functions_for_constraint=loss_functions['validation']
    )
    constants = compute_constants_for_sufficient_statistics(
        loss_functions_for_training=loss_functions['train'], initial_state=initial_state_learned_algorithm)
    sufficient_statistics = get_sufficient_statistics(
        template_for_loss_function=loss_function_for_algorithm, constants=constants)
    algorithm_for_learning = PacBayesOptimizationAlgorithm(
        initial_state=initial_state_learned_algorithm,
        implementation=SparsityNet(dim=initial_state_learned_algorithm.shape[1]),
        loss_function=loss_functions['prior'][0],
        pac_parameters=get_pac_bayes_parameters(sufficient_statistics),
        constraint=constraint
    )
    return algorithm_for_learning


def create_parametric_loss_functions_from_parameters(template_loss_function, smooth_part, nonsmooth_part, parameters):

    loss_functions = {
        name: [NonsmoothParametricLossFunction(function=template_loss_function, smooth_part=smooth_part,
                                               nonsmooth_part=nonsmooth_part, parameter=p)
               for p in parameters[name]] for name in list(parameters.keys())
    }
    return loss_functions


def get_baseline_algorithm(smoothness_parameter, initial_state, loss_function):

    alpha = 1 / smoothness_parameter
    std_algo = OptimizationAlgorithm(
        initial_state=initial_state,
        implementation=FISTA(alpha=alpha),
        loss_function=loss_function
    )
    return std_algo


def set_up_and_train_algorithm(path_of_experiment):

    number_of_datapoints_per_dataset = get_number_of_datapoints()
    parameters, loss_function_of_algorithm, smooth_part, nonsmooth_part, smoothness_parameter = get_data(
        number_of_datapoints_per_dataset)
    loss_functions = create_parametric_loss_functions_from_parameters(
        template_loss_function=loss_function_of_algorithm, smooth_part=smooth_part, nonsmooth_part=nonsmooth_part,
        parameters=parameters)
    initial_state_fista, initial_state_learned_algorithm = get_initial_states()
    baseline_algorithm = get_baseline_algorithm(
        smoothness_parameter=smoothness_parameter, initial_state=initial_state_fista,
        loss_function=loss_functions['prior'][0])
    algorithm_for_learning = get_algorithm_for_learning(
        loss_function_for_algorithm=loss_function_of_algorithm, loss_functions=loss_functions)

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
    save_data(savings_path=savings_path, smoothness_parameter=smoothness_parameter.numpy(),
              pac_bound=pac_bound.numpy(),
              initialization_learned_algorithm=algorithm_for_learning.initial_state.clone().numpy(),
              initialization_baseline_algorithm=baseline_algorithm.initial_state.clone().numpy(),
              number_of_iterations=algorithm_for_learning.n_max, parameters=parameters,
              samples_prior=state_dict_samples_prior, best_sample=algorithm_for_learning.implementation.state_dict())


def save_data(savings_path, smoothness_parameter, pac_bound, initialization_learned_algorithm,
              initialization_baseline_algorithm, number_of_iterations, parameters, samples_prior, best_sample):

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