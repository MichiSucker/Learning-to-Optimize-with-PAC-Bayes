from pathlib import Path
import numpy as np
import pickle
import time
from experiments.nn_training.neural_network import train_model
from experiments.nn_training.training import instantiate_neural_networks
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from experiments.nn_training.algorithm import NnOptimizer
import torch
from experiments.nn_training.data_generation import get_loss_of_algorithm, get_loss_of_neural_network
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from experiments.nn_training.training import get_describing_property


class EvaluationAssistant:

    def __init__(self,
                 test_set,
                 number_of_iterations_during_training,
                 number_of_iterations_for_testing,
                 loss_of_algorithm,
                 initial_state,
                 optimal_hyperparameters,
                 implementation_class):
        self.test_set = test_set
        self.number_of_iterations_during_training = number_of_iterations_during_training
        self.number_of_iterations_for_testing = number_of_iterations_for_testing
        self.loss_of_algorithm = loss_of_algorithm
        self.initial_state = initial_state
        self.optimal_hyperparameters = optimal_hyperparameters
        self.implementation_class = implementation_class
        self.loss_of_neural_network = None
        self.implementation_arguments = None
        self.lr_adam = None

    def set_up_learned_algorithm(self, arguments_of_implementation_class):
        if arguments_of_implementation_class is None:
            learned_algorithm = OptimizationAlgorithm(
                implementation=self.implementation_class(),
                initial_state=self.initial_state,
                loss_function=ParametricLossFunction(function=self.loss_of_algorithm,
                                                     parameter=self.test_set[0]))
        elif (arguments_of_implementation_class is not None) and (self.implementation_arguments is not None):
            learned_algorithm = OptimizationAlgorithm(
                implementation=self.implementation_class(self.implementation_arguments),
                initial_state=self.initial_state,
                loss_function=ParametricLossFunction(function=self.loss_of_algorithm,
                                                     parameter=self.test_set[0]))
        else:
            raise Exception("Arguments of implementation are not specified correctly.")
        learned_algorithm.implementation.load_state_dict(self.optimal_hyperparameters)
        return learned_algorithm


def load_data(loading_path):
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
    savings_path = path_of_experiment + "/data/"
    Path(savings_path).mkdir(parents=True, exist_ok=True)
    return savings_path


def compute_ground_truth_loss(loss_of_neural_network, parameter):
    return loss_of_neural_network(parameter['ground_truth_values'], parameter['y_values'])


def compute_losses_over_iterations_for_learned_algorithm(learned_algorithm,
                                                         evaluation_assistant: EvaluationAssistant,
                                                         parameter):
    learned_algorithm.reset_state_and_iteration_counter()
    current_loss_function = ParametricLossFunction(function=evaluation_assistant.loss_of_algorithm, parameter=parameter)
    learned_algorithm.set_loss_function(current_loss_function)
    loss_over_iterations = [learned_algorithm.evaluate_loss_function_at_current_iterate().item()]
    for i in range(evaluation_assistant.number_of_iterations_for_testing):
        learned_algorithm.perform_step()
        loss_over_iterations.append(learned_algorithm.evaluate_loss_function_at_current_iterate().item())
    return loss_over_iterations


def does_satisfy_constraint(convergence_risk_constraint, loss_at_beginning, loss_at_end):
    return convergence_risk_constraint(loss_at_beginning=loss_at_beginning, loss_at_end=loss_at_end)


def compute_losses_over_iterations_for_adam(neural_network, evaluation_assistant: EvaluationAssistant, parameter):
    neural_network_for_standard_training, losses_over_iterations_of_adam, _ = train_model(
        net=neural_network, data=parameter, criterion=evaluation_assistant.loss_of_neural_network,
        n_it=evaluation_assistant.number_of_iterations_for_testing, lr=evaluation_assistant.lr_adam
    )
    return losses_over_iterations_of_adam


def compute_losses(evaluation_assistant: EvaluationAssistant,
                   learned_algorithm,
                   neural_network_for_standard_training):

    ground_truth_losses = []
    losses_of_adam = []
    losses_of_learned_algorithm = []
    number_of_times_constrained_satisfied = 0
    _, convergence_risk_constraint, _ = get_describing_property()

    for test_parameter in evaluation_assistant.test_set:

        loss_over_iterations = compute_losses_over_iterations_for_learned_algorithm(
            learned_algorithm=learned_algorithm, evaluation_assistant=evaluation_assistant, parameter=test_parameter)

        if does_satisfy_constraint(
                convergence_risk_constraint=convergence_risk_constraint, loss_at_beginning=loss_over_iterations[0],
                loss_at_end=loss_over_iterations[evaluation_assistant.number_of_iterations_during_training]):

            number_of_times_constrained_satisfied += 1
            losses_of_learned_algorithm.append(loss_over_iterations)
            ground_truth_losses.append(compute_ground_truth_loss(evaluation_assistant.loss_of_neural_network,
                                                                 test_parameter))
            neural_network_for_standard_training.load_parameters_from_tensor(
                learned_algorithm.initial_state[-1].clone())
            losses_of_adam.append(compute_losses_over_iterations_for_adam(
                neural_network=neural_network_for_standard_training, evaluation_assistant=evaluation_assistant,
                parameter=test_parameter))

    return (np.array(losses_of_adam),
            np.array(losses_of_learned_algorithm),
            np.array(ground_truth_losses),
            number_of_times_constrained_satisfied / len(evaluation_assistant.test_set))


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


def time_problem_for_adam(neural_network, loss_of_neural_network, parameter, maximal_number_of_iterations,
                          optimal_loss, level_of_accuracy, lr_adam):
    optimizer = torch.optim.Adam(neural_network.parameters(), lr=lr_adam)
    current_loss = loss_of_neural_network(neural_network(parameter['x_values']), parameter['y_values'])
    counter = 0
    start = time.time()
    while ((counter < maximal_number_of_iterations)
           and (current_loss - optimal_loss >= level_of_accuracy)):

        optimizer.zero_grad()
        current_loss = loss_of_neural_network(neural_network(parameter['x_values']), parameter['y_values'])
        current_loss.backward()
        optimizer.step()
        counter += 1

    end = time.time()
    return end - start


def compute_times(learned_algorithm: OptimizationAlgorithm,
                  neural_network_for_standard_training,
                  evaluation_assistant: EvaluationAssistant,
                  ground_truth_losses,
                  stop_procedure_after_at_most):

    levels_of_accuracy = [1e0, 1e-1, 1e-2]
    times_pac = {epsilon: [0.] for epsilon in levels_of_accuracy}
    times_std = {epsilon: [0.] for epsilon in levels_of_accuracy}

    for parameter, ground_truth_loss in zip(evaluation_assistant.test_set, ground_truth_losses):
        for epsilon in levels_of_accuracy:

            cur_loss_function = ParametricLossFunction(function=evaluation_assistant.loss_of_algorithm,
                                                       parameter=parameter)
            times_pac[epsilon].append(time_problem_for_learned_algorithm(
                learned_algorithm=learned_algorithm, loss_function=cur_loss_function,
                maximal_number_of_iterations=stop_procedure_after_at_most, optimal_loss=ground_truth_loss,
                level_of_accuracy=epsilon))

            neural_network_for_standard_training.load_parameters_from_tensor(
                learned_algorithm.initial_state[-1].clone())
            times_std[epsilon].append(time_problem_for_adam(
                neural_network=neural_network_for_standard_training,
                loss_of_neural_network=evaluation_assistant.loss_of_neural_network, parameter=parameter,
                maximal_number_of_iterations=stop_procedure_after_at_most, optimal_loss=ground_truth_loss,
                level_of_accuracy=epsilon, lr_adam=evaluation_assistant.lr_adam))

    return times_pac, times_std


def set_up_evaluation_assistant(loading_path):
    pac_bound, initial_state, n_train, parameters, samples, best_sample = load_data(loading_path)
    neural_network_for_standard_training, neural_network_for_learning = instantiate_neural_networks()
    loss_of_neural_network = get_loss_of_neural_network()
    loss_of_algorithm = get_loss_of_algorithm(neural_network=neural_network_for_learning,
                                              loss_of_neural_network=loss_of_neural_network)

    evaluation_assistant = EvaluationAssistant(
        test_set=parameters['test'], number_of_iterations_during_training=n_train,
        number_of_iterations_for_testing=2*n_train,
        loss_of_algorithm=loss_of_algorithm, initial_state=torch.tensor(initial_state),
        optimal_hyperparameters=best_sample, implementation_class=NnOptimizer)
    evaluation_assistant.loss_of_neural_network = loss_of_neural_network
    evaluation_assistant.implementation_arguments = (
        neural_network_for_standard_training.get_dimension_of_hyperparameters())
    evaluation_assistant.lr_adam = 0.008  # Originally, this was found by gridsearch.
    return evaluation_assistant, neural_network_for_standard_training


def evaluate_algorithm(loading_path, path_of_experiment):

    evaluation_assistant, neural_network_for_standard_training = set_up_evaluation_assistant(loading_path)
    learned_algorithm = evaluation_assistant.set_up_learned_algorithm(
        arguments_of_implementation_class=evaluation_assistant.implementation_arguments)

    losses_of_adam, losses_of_learned_algorithm, ground_truth_losses, percentage_constrained_satisfied = compute_losses(
        evaluation_assistant=evaluation_assistant, learned_algorithm=learned_algorithm,
        neural_network_for_standard_training=neural_network_for_standard_training
    )

    times_of_learned_algorithm, times_of_adam = compute_times(
        learned_algorithm=learned_algorithm, neural_network_for_standard_training=neural_network_for_standard_training,
        evaluation_assistant=evaluation_assistant, ground_truth_losses=ground_truth_losses,
        stop_procedure_after_at_most=5000)

    save_data(savings_path=create_folder_for_storing_data(path_of_experiment),
              times_of_learned_algorithm=times_of_learned_algorithm,
              losses_of_learned_algorithm=losses_of_learned_algorithm,
              times_of_adam=times_of_adam, losses_of_adam=losses_of_adam,
              ground_truth_losses=ground_truth_losses,
              percentage_constrained_satisfied=percentage_constrained_satisfied)


def save_data(savings_path, times_of_learned_algorithm, losses_of_learned_algorithm, times_of_adam, losses_of_adam,
              ground_truth_losses, percentage_constrained_satisfied):

    with open(savings_path + 'times_of_learned_algorithm', 'wb') as file:
        pickle.dump(times_of_learned_algorithm, file)
    with open(savings_path + 'times_of_adam', 'wb') as file:
        pickle.dump(times_of_adam, file)
    np.save(savings_path + 'losses_of_adam', np.array(losses_of_adam))
    np.save(savings_path + 'losses_of_learned_algorithm', np.array(losses_of_learned_algorithm))
    np.save(savings_path + 'ground_truth_losses', np.array(ground_truth_losses))
    np.save(savings_path + 'empirical_probability', percentage_constrained_satisfied)