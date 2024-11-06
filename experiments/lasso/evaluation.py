from classes.LossFunction.derived_classes.NonsmoothParametricLossFunction.\
    subclass_NonsmoothParametricLossFunction import NonsmoothParametricLossFunction
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from experiments.lasso.training import get_describing_property, get_baseline_algorithm
import torch
import pickle
import numpy as np
from pathlib import Path
import time
from experiments.lasso.algorithm import SparsityNet
from experiments.lasso.data_generation import get_loss_function_of_algorithm


class EvaluationAssistant:

    def __init__(self,
                 test_set,
                 loss_of_algorithm,
                 smooth_part,
                 nonsmooth_part,
                 initial_state_learned_algorithm,
                 number_of_iterations_during_training,
                 optimal_hyperparameters,
                 implementation_class):
        self.test_set = test_set
        self.initial_state_learned_algorithm = initial_state_learned_algorithm
        self.number_of_iterations_during_training = number_of_iterations_during_training
        self.number_of_iterations_for_testing = 2 * number_of_iterations_during_training
        self.loss_of_algorithm = loss_of_algorithm
        self.smooth_part = smooth_part
        self.nonsmooth_part = nonsmooth_part
        self.optimal_hyperparameters = optimal_hyperparameters
        self.implementation_class = implementation_class
        self.dim = initial_state_learned_algorithm.shape[1]
        self.number_of_iterations_for_approximation = 5000
        self.implementation_arguments = None
        self.smoothness_parameter = None
        self.initial_state_baseline_algorithm = None

    def set_up_learned_algorithm(self, arguments_of_implementation_class):
        if arguments_of_implementation_class is None:
            learned_algorithm = OptimizationAlgorithm(
                implementation=self.implementation_class(),
                initial_state=self.initial_state_learned_algorithm,
                loss_function=ParametricLossFunction(function=self.loss_of_algorithm,
                                                     parameter=self.test_set[0]))
        elif (arguments_of_implementation_class is not None) and (self.implementation_arguments is not None):
            learned_algorithm = OptimizationAlgorithm(
                implementation=self.implementation_class(self.implementation_arguments),
                initial_state=self.initial_state_learned_algorithm,
                loss_function=ParametricLossFunction(function=self.loss_of_algorithm,
                                                     parameter=self.test_set[0]))
        else:
            raise Exception("Arguments of implementation are not specified correctly.")
        learned_algorithm.implementation.load_state_dict(self.optimal_hyperparameters)
        return learned_algorithm


def load_data(loading_path):
    pac_bound = np.load(loading_path + 'pac_bound.npy')
    initial_state_learned_algorithm = torch.tensor(np.load(loading_path + 'initialization_learned_algorithm.npy'))
    initialization_baseline_algorithm = torch.tensor(np.load(loading_path + 'initialization_baseline_algorithm.npy'))
    n_train = np.load(loading_path + 'number_of_iterations.npy')
    smoothness_parameter = np.load(loading_path + 'smoothness_parameter.npy')
    with open(loading_path + 'parameters_problem', 'rb') as file:
        parameters = pickle.load(file)
    with open(loading_path + 'samples', 'rb') as file:
        samples = pickle.load(file)
    with open(loading_path + 'best_sample', 'rb') as file:
        best_sample = pickle.load(file)
    return (pac_bound, initial_state_learned_algorithm, initialization_baseline_algorithm, n_train, parameters, samples,
            best_sample, smoothness_parameter)


def create_folder_for_storing_data(path_of_experiment):  # pragma: no cover
    savings_path = path_of_experiment + "/data/"
    Path(savings_path).mkdir(parents=True, exist_ok=True)
    return savings_path


def save_data(savings_path, times_of_learned_algorithm, losses_of_learned_algorithm, times_of_baseline_algorithm,
              losses_of_baseline_algorithm, percentage_constrained_satisfied, number_of_iterations_for_approximation,
              approximate_optimal_losses):

    with open(savings_path + 'times_of_learned_algorithm', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(times_of_learned_algorithm, file)
    with open(savings_path + 'times_of_baseline_algorithm', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(times_of_baseline_algorithm, file)
    np.save(savings_path + 'losses_of_baseline_algorithm', np.array(losses_of_baseline_algorithm))
    np.save(savings_path + 'losses_of_learned_algorithm', np.array(losses_of_learned_algorithm))
    np.save(savings_path + 'empirical_probability', percentage_constrained_satisfied)
    np.save(savings_path + 'approximate_optimal_losses', approximate_optimal_losses)
    np.save(savings_path + 'number_of_iterations_for_approximation', number_of_iterations_for_approximation)


def set_up_evaluation_assistant(loading_path):
    (pac_bound, initial_state_learned_algorithm, initialization_baseline_algorithm, n_train, parameters, samples,
     best_sample, smoothness_parameter) = load_data(loading_path)
    loss_of_algorithm, smooth_part, nonsmooth_part = get_loss_function_of_algorithm()
    evaluation_assistant = EvaluationAssistant(
        test_set=parameters['test'], loss_of_algorithm=loss_of_algorithm,
        smooth_part=smooth_part, nonsmooth_part=nonsmooth_part,
        initial_state_learned_algorithm=initial_state_learned_algorithm, number_of_iterations_during_training=n_train,
        optimal_hyperparameters=best_sample, implementation_class=SparsityNet)
    evaluation_assistant.implementation_arguments = initial_state_learned_algorithm.shape[1]
    evaluation_assistant.smoothness_parameter = torch.tensor(smoothness_parameter)
    evaluation_assistant.initial_state_baseline_algorithm = initialization_baseline_algorithm
    return evaluation_assistant


def does_satisfy_constraint(convergence_risk_constraint, loss_at_beginning, loss_at_end):
    return convergence_risk_constraint(loss_at_beginning=loss_at_beginning, loss_at_end=loss_at_end)


def compute_losses_over_iterations(algorithm, evaluation_assistant: EvaluationAssistant, parameter):
    algorithm.reset_state_and_iteration_counter()
    current_loss_function = NonsmoothParametricLossFunction(function=evaluation_assistant.loss_of_algorithm,
                                                            smooth_part=evaluation_assistant.smooth_part,
                                                            nonsmooth_part=evaluation_assistant.nonsmooth_part,
                                                            parameter=parameter)
    algorithm.set_loss_function(current_loss_function)
    loss_over_iterations = [algorithm.evaluate_loss_function_at_current_iterate().item()]
    for i in range(evaluation_assistant.number_of_iterations_for_testing):
        algorithm.perform_step()
        loss_over_iterations.append(algorithm.evaluate_loss_function_at_current_iterate().item())
    return loss_over_iterations


def compute_losses(evaluation_assistant: EvaluationAssistant, learned_algorithm, baseline_algorithm):

    losses_of_baseline_algorithm = []
    losses_of_learned_algorithm = []
    number_of_times_constrained_satisfied = 0
    _, convergence_risk_constraint, _ = get_describing_property()

    for test_parameter in evaluation_assistant.test_set:

        loss_over_iterations = compute_losses_over_iterations(
            algorithm=learned_algorithm, evaluation_assistant=evaluation_assistant, parameter=test_parameter)

        if does_satisfy_constraint(
                convergence_risk_constraint=convergence_risk_constraint, loss_at_beginning=loss_over_iterations[0],
                loss_at_end=loss_over_iterations[evaluation_assistant.number_of_iterations_during_training]):

            number_of_times_constrained_satisfied += 1
            losses_of_learned_algorithm.append(loss_over_iterations)
            losses_of_baseline_algorithm.append(compute_losses_over_iterations(
                algorithm=baseline_algorithm, evaluation_assistant=evaluation_assistant, parameter=test_parameter))

    return (np.array(losses_of_baseline_algorithm),
            np.array(losses_of_learned_algorithm),
            number_of_times_constrained_satisfied / len(evaluation_assistant.test_set))


def approximate_optimal_loss(baseline_algorithm: OptimizationAlgorithm, evaluation_assistant: EvaluationAssistant):
    optimal_losses = []
    for parameter in evaluation_assistant.test_set:

        baseline_algorithm.reset_state_and_iteration_counter()
        current_loss_function = NonsmoothParametricLossFunction(function=evaluation_assistant.loss_of_algorithm,
                                                                smooth_part=evaluation_assistant.smooth_part,
                                                                nonsmooth_part=evaluation_assistant.nonsmooth_part,
                                                                parameter=parameter)
        baseline_algorithm.set_loss_function(current_loss_function)

        for _ in range(evaluation_assistant.number_of_iterations_for_approximation):
            baseline_algorithm.perform_step()

        optimal_losses.append(baseline_algorithm.evaluate_loss_function_at_current_iterate().item())

    return np.array(optimal_losses)


def time_problem(algorithm, loss_function, maximal_number_of_iterations, optimal_loss, level_of_accuracy):

    algorithm.reset_state_and_iteration_counter()
    algorithm.set_loss_function(loss_function)
    current_loss = algorithm.evaluate_loss_function_at_current_iterate()
    counter = 0
    start = time.time()

    while ((counter < maximal_number_of_iterations)
           and ((current_loss - optimal_loss) >= level_of_accuracy)):

        algorithm.perform_step()
        current_loss = algorithm.evaluate_loss_function_at_current_iterate()
        counter += 1

    end = time.time()
    return end - start


def compute_times(learned_algorithm: OptimizationAlgorithm,
                  baseline_algorithm: OptimizationAlgorithm,
                  evaluation_assistant: EvaluationAssistant,
                  optimal_losses,
                  stop_procedure_after_at_most):

    levels_of_accuracy = [1e1, 5e0, 1e0]
    times_pac = {epsilon: [0.] for epsilon in levels_of_accuracy}
    times_std = {epsilon: [0.] for epsilon in levels_of_accuracy}

    for parameter, optimal_loss in zip(evaluation_assistant.test_set, optimal_losses):
        for epsilon in levels_of_accuracy:

            cur_loss_function = NonsmoothParametricLossFunction(function=evaluation_assistant.loss_of_algorithm,
                                                                smooth_part=evaluation_assistant.smooth_part,
                                                                nonsmooth_part=evaluation_assistant.nonsmooth_part,
                                                                parameter=parameter)
            times_pac[epsilon].append(time_problem(
                algorithm=learned_algorithm, loss_function=cur_loss_function,
                maximal_number_of_iterations=stop_procedure_after_at_most, optimal_loss=optimal_loss,
                level_of_accuracy=epsilon))

            times_std[epsilon].append(time_problem(
                algorithm=baseline_algorithm, loss_function=cur_loss_function,
                maximal_number_of_iterations=stop_procedure_after_at_most, optimal_loss=optimal_loss,
                level_of_accuracy=epsilon))

    return times_pac, times_std


def set_up_algorithms(evaluation_assistant):
    learned_algorithm = evaluation_assistant.set_up_learned_algorithm(
        arguments_of_implementation_class=evaluation_assistant.implementation_arguments)
    baseline_algorithm = get_baseline_algorithm(
        loss_function=learned_algorithm.loss_function,
        initial_state=evaluation_assistant.initial_state_baseline_algorithm,
        smoothness_parameter=evaluation_assistant.smoothness_parameter)
    return learned_algorithm, baseline_algorithm


def evaluate_algorithm(loading_path, path_of_experiment):

    evaluation_assistant = set_up_evaluation_assistant(loading_path)
    learned_algorithm, baseline_algorithm = set_up_algorithms(evaluation_assistant)
    losses_of_baseline, losses_of_learned_algorithm, percentage_constrained_satisfied = compute_losses(
        evaluation_assistant=evaluation_assistant, learned_algorithm=learned_algorithm,
        baseline_algorithm=baseline_algorithm
    )
    optimal_losses = approximate_optimal_loss(
        baseline_algorithm=baseline_algorithm, evaluation_assistant=evaluation_assistant
    )
    times_of_learned_algorithm, times_of_baseline_algorithm = compute_times(
        learned_algorithm=learned_algorithm, baseline_algorithm=baseline_algorithm,
        evaluation_assistant=evaluation_assistant,
        stop_procedure_after_at_most=evaluation_assistant.number_of_iterations_for_approximation,
        optimal_losses=optimal_losses)
    save_data(savings_path=create_folder_for_storing_data(path_of_experiment),
              times_of_learned_algorithm=times_of_learned_algorithm,
              losses_of_learned_algorithm=losses_of_learned_algorithm,
              times_of_baseline_algorithm=times_of_baseline_algorithm,
              losses_of_baseline_algorithm=losses_of_baseline,
              approximate_optimal_losses=optimal_losses,
              number_of_iterations_for_approximation=evaluation_assistant.number_of_iterations_for_approximation,
              percentage_constrained_satisfied=percentage_constrained_satisfied)
