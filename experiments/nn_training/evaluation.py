from pathlib import Path
import numpy as np
import pickle
import time
from experiments.nn_training.neural_network import train_model
from experiments.nn_training.training import instantiate_neural_networks
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from algorithms.nn_optimizer import NnOptimizer
import torch
from experiments.nn_training.data_generation import get_loss_of_algorithm, get_loss_of_neural_network
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from experiments.nn_training.training import get_describing_property

# TODO: Refactor. Functions are way too long and not really readable
# TODO: Cover with tests.


def load_data(loading_path):  # pragma: no cover
    pac_bound = np.load(loading_path + 'pac_bound.npy')
    initial_state = np.load(loading_path + 'initialization.npy')
    n_train = np.load(loading_path + 'number_of_iterations.npy')
    with open(loading_path + 'parameters_problem', 'rb') as file:
        parameters = pickle.load(file)
    with open(loading_path + 'samples', 'rb') as file:
        samples = pickle.load(file)
    with open(loading_path + 'best_sample', 'rb') as file:
        best_sample = pickle.load(file)
    return pac_bound, initial_state, n_train, parameters, samples, best_sample


def create_folder_for_storing_data(path_of_experiment):  # pragma: no cover
    savings_path = path_of_experiment + "/data_after_evaluation/"
    Path(savings_path).mkdir(parents=True, exist_ok=True)
    return savings_path


def compute_ground_truth_loss(loss_of_neural_network, parameter):
    return loss_of_neural_network(parameter['ground_truth_values'], parameter['y_values'])


def compute_losses_over_iterations_for_learned_algorithm(learned_algorithm, loss_of_algorithm, parameter, number_of_iterations):
    learned_algorithm.reset_state_and_iteration_counter()
    current_loss_function = ParametricLossFunction(function=loss_of_algorithm, parameter=parameter)
    learned_algorithm.set_loss_function(current_loss_function)
    loss_over_iterations = [learned_algorithm.evaluate_loss_function_at_current_iterate().item()]
    for i in range(number_of_iterations):
        learned_algorithm.perform_step()
        loss_over_iterations.append(learned_algorithm.evaluate_loss_function_at_current_iterate().item())
    return loss_over_iterations


def does_satisfy_constraint(convergence_risk_constraint, loss_at_beginning, loss_at_end):
    return convergence_risk_constraint(loss_at_beginning=loss_at_beginning, loss_at_end=loss_at_end)


def compute_losses_over_iterations_for_adam(neural_network, loss_of_neural_network, parameter, number_of_iterations):
    lr_adam = 0.008
    neural_network_for_standard_training, losses_over_iterations_of_adam, _ = train_model(
        net=neural_network, data=parameter, criterion=loss_of_neural_network,
        n_it=number_of_iterations, lr=lr_adam
    )
    return losses_over_iterations_of_adam


def compute_losses(parameters_to_test, learned_algorithm, neural_network, loss_of_neural_network, loss_of_algorithm):

    n_train, n_test = learned_algorithm.n_max, 2 * learned_algorithm.n_max
    ground_truth_losses = []
    losses_of_adam = []
    losses_of_learned_algorithm = []
    number_of_times_constrained_satisfied = 0
    _, convergence_risk_constraint, _ = get_describing_property()

    for current_parameter in parameters_to_test:
        loss_over_iterations = compute_losses_over_iterations_for_learned_algorithm(learned_algorithm=learned_algorithm,
                                                                                    loss_of_algorithm=loss_of_algorithm,
                                                                                    parameter=current_parameter,
                                                                                    number_of_iterations=n_test)
        if does_satisfy_constraint(convergence_risk_constraint=convergence_risk_constraint,
                                   loss_at_beginning=loss_over_iterations[0],
                                   loss_at_end=loss_over_iterations[n_train]):

            number_of_times_constrained_satisfied += 1
            ground_truth_losses.append(compute_ground_truth_loss(loss_of_neural_network, current_parameter))
            losses_of_learned_algorithm.append(loss_over_iterations)
            neural_network.load_parameters_from_tensor(learned_algorithm.initial_state[-1].clone())
            losses_of_adam.append(compute_losses_over_iterations_for_adam(neural_network=neural_network,
                                                                          loss_of_neural_network=loss_of_neural_network,
                                                                          parameter=current_parameter,
                                                                          number_of_iterations=n_test))

    return (np.array(losses_of_adam),
            np.array(losses_of_learned_algorithm),
            np.array(ground_truth_losses),
            number_of_times_constrained_satisfied / len(parameters_to_test))


def time_problem_for_learned_algorithm(learned_algorithm, loss_function, maximal_number_of_iterations,
                                       optimal_loss, level_of_accuracy):
    learned_algorithm.reset_state_and_iteration_counter()
    learned_algorithm.set_loss_function(loss_function)
    current_loss = learned_algorithm.evaluate_loss_function_at_current_iterate()
    counter = 0
    start = time.time()
    while ((counter < maximal_number_of_iterations)
           and ((current_loss - optimal_loss) >= level_of_accuracy)):

        learned_algorithm.perform_step()
        current_loss = learned_algorithm.evaluate_loss_function_at_current_iterate()
        counter += 1

    end = time.time()
    return end - start


def time_problem_for_adam(neural_network, initialization, loss_of_neural_network, data, maximal_number_of_iterations,
                          optimal_loss, level_of_accuracy):
    neural_network.load_parameters_from_tensor(initialization)
    optimizer = torch.optim.Adam(neural_network.parameters())
    current_loss = loss_of_neural_network(neural_network(data['x_values']), data['y_values'])
    counter = 0
    start = time.time()
    while ((counter < maximal_number_of_iterations)
           and (current_loss - optimal_loss >= level_of_accuracy)):

        optimizer.zero_grad()
        current_loss = loss_of_neural_network(neural_network(data['x_values']), data['y_values'])
        current_loss.backward()
        optimizer.step()
        counter += 1

    end = time.time()
    return end - start


def compute_times(learned_algorithm: OptimizationAlgorithm,
                  neural_network,
                  loss_of_neural_network,
                  loss_func,
                  test_data,
                  levels_of_accuracy,
                  optimal_losses,
                  n_max):

    times_pac = {epsilon: [0.] for epsilon in levels_of_accuracy}
    times_std = {epsilon: [0.] for epsilon in levels_of_accuracy}

    for cur_data, cur_optimal_loss in zip(test_data, optimal_losses):
        for epsilon in levels_of_accuracy:

            cur_loss_function = ParametricLossFunction(function=loss_func, parameter=cur_data)
            cur_time_pac = time_problem_for_learned_algorithm(learned_algorithm=learned_algorithm,
                                                              loss_function=cur_loss_function,
                                                              maximal_number_of_iterations=n_max,
                                                              optimal_loss=cur_optimal_loss,
                                                              level_of_accuracy=epsilon)
            cur_time_adam = time_problem_for_adam(neural_network=neural_network,
                                                  initialization=learned_algorithm.initial_state[-1].clone(),
                                                  loss_of_neural_network=loss_of_neural_network,
                                                  data=cur_data,
                                                  maximal_number_of_iterations=n_max,
                                                  optimal_loss=cur_optimal_loss,
                                                  level_of_accuracy=epsilon)

            times_pac[epsilon].append(cur_time_pac)
            times_std[epsilon].append(cur_time_adam)

    return times_pac, times_std


def evaluate_algorithm(loading_path, path_of_experiment):
    savings_path = create_folder_for_storing_data(path_of_experiment)
    pac_bound, initial_state, n_train, parameters, samples, best_sample = load_data(loading_path)
    neural_network_for_standard_training, neural_network_for_learning = instantiate_neural_networks()
    loss_of_neural_network = get_loss_of_neural_network()
    loss_of_algorithm = get_loss_of_algorithm(neural_network=neural_network_for_standard_training,
                                              loss_of_neural_network=loss_of_neural_network)
    learned_algorithm = OptimizationAlgorithm(
        implementation=NnOptimizer(dim=neural_network_for_standard_training.get_dimension_of_hyperparameters()),
        initial_state=torch.tensor(initial_state),
        loss_function=ParametricLossFunction(function=loss_of_algorithm, parameter=parameters['test'][0])
    )
    learned_algorithm.n_max = n_train

    ground_truth_losses, losses_of_adam, losses_of_learned_algorithm, percentage_constrained_satisfied = compute_losses(
        parameters_to_test=parameters['test'], learned_algorithm=learned_algorithm,
        neural_network=neural_network_for_learning, loss_of_neural_network=loss_of_neural_network,
        loss_of_algorithm=loss_of_algorithm
    )
    times_pac, times_std = compute_times(
        learned_algorithm=learned_algorithm, neural_network=neural_network_for_standard_training,
        loss_of_neural_network=loss_of_neural_network, loss_func=loss_of_algorithm, test_data=parameters['test'],
        levels_of_accuracy=[1e0, 1e-1, 1e-2], optimal_losses=ground_truth_losses, n_max=5000
    )

    # TODO: Save data again.

