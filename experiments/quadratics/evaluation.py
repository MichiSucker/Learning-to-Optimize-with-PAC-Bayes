import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Callable
import torch
import pickle
from pathlib import Path
import time
from tqdm import tqdm

from classes.LossFunction.class_LossFunction import LossFunction
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from experiments.quadratics.algorithm import Quadratics
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from experiments.quadratics.data_generation import get_loss_function_of_algorithm
from experiments.quadratics.training import get_describing_property, get_baseline_algorithm


class EvaluationAssistant:

    def __init__(self,
                 test_set: List,
                 loss_of_algorithm: Callable,
                 initial_state: torch.Tensor,
                 number_of_iterations_during_training: int,
                 optimal_hyperparameters: dict,
                 implementation_class: Callable):
        self.test_set = test_set
        self.initial_state = initial_state
        self.number_of_iterations_during_training = number_of_iterations_during_training
        self.number_of_iterations_for_testing = 2 * number_of_iterations_during_training
        self.loss_of_algorithm = loss_of_algorithm
        self.optimal_hyperparameters = optimal_hyperparameters
        self.implementation_class = implementation_class
        self.dim = initial_state.shape[1]
        self.implementation_arguments = None
        self.strong_convexity_parameter = None
        self.smoothness_parameter = None

    def set_up_learned_algorithm(self, arguments_of_implementation_class: dict | None) -> OptimizationAlgorithm:
        if arguments_of_implementation_class is None:
            learned_algorithm = OptimizationAlgorithm(
                implementation=self.implementation_class(),
                initial_state=self.initial_state,
                loss_function=ParametricLossFunction(function=self.loss_of_algorithm,
                                                     parameter=self.test_set[0]))
        elif (arguments_of_implementation_class is not None) and (self.implementation_arguments is not None):
            learned_algorithm = OptimizationAlgorithm(
                implementation=self.implementation_class(**self.implementation_arguments),
                initial_state=self.initial_state,
                loss_function=ParametricLossFunction(function=self.loss_of_algorithm,
                                                     parameter=self.test_set[0]))
        else:
            raise Exception("Arguments of implementation are not specified correctly.")
        learned_algorithm.implementation.load_state_dict(self.optimal_hyperparameters)
        return learned_algorithm


def load_data(loading_path: str) -> Tuple:
    pac_bound = np.load(loading_path + 'pac_bound.npy')
    initial_state = torch.tensor(np.load(loading_path + 'initialization.npy'))
    n_train = np.load(loading_path + 'number_of_iterations.npy')
    smoothness_parameter = np.load(loading_path + 'smoothness_parameter.npy')
    strong_convexity_parameter = np.load(loading_path + 'strong_convexity_parameter.npy')
    with open(loading_path + 'parameters_problem', 'rb') as file:
        parameters = pickle.load(file)
    with open(loading_path + 'samples', 'rb') as file:
        samples = pickle.load(file)
    with open(loading_path + 'best_sample', 'rb') as file:
        best_sample = pickle.load(file)
    return (pac_bound, initial_state, n_train, parameters, samples, best_sample,
            strong_convexity_parameter, smoothness_parameter)


def create_folder_for_storing_data(path_of_experiment: str) -> str:
    savings_path = path_of_experiment + "/data/"
    Path(savings_path).mkdir(parents=True, exist_ok=True)
    return savings_path


def save_data(savings_path: str,
              times_of_learned_algorithm: dict,
              losses_of_learned_algorithm: NDArray,
              times_of_baseline_algorithm: dict,
              losses_of_baseline_algorithm: NDArray,
              ground_truth_losses: List,
              percentage_constrained_satisfied: float) -> None:

    with open(savings_path + 'times_of_learned_algorithm', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(times_of_learned_algorithm, file)
    with open(savings_path + 'times_of_baseline_algorithm', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(times_of_baseline_algorithm, file)
    np.save(savings_path + 'losses_of_baseline_algorithm', np.array(losses_of_baseline_algorithm))
    np.save(savings_path + 'losses_of_learned_algorithm', np.array(losses_of_learned_algorithm))
    np.save(savings_path + 'ground_truth_losses', np.array(ground_truth_losses))
    np.save(savings_path + 'empirical_probability', percentage_constrained_satisfied)


def set_up_evaluation_assistant(loading_path: str) -> EvaluationAssistant:
    (pac_bound, initial_state, n_train, parameters, samples, best_sample, strong_convexity_parameter,
     smoothness_parameter) = load_data(loading_path)
    loss_of_algorithm = get_loss_function_of_algorithm()

    evaluation_assistant = EvaluationAssistant(test_set=parameters['test'], loss_of_algorithm=loss_of_algorithm,
                                               initial_state=initial_state,
                                               number_of_iterations_during_training=n_train,
                                               optimal_hyperparameters=best_sample, implementation_class=Quadratics)
    evaluation_assistant.smoothness_parameter = torch.tensor(smoothness_parameter)
    evaluation_assistant.strong_convexity_parameter = torch.tensor(strong_convexity_parameter)
    return evaluation_assistant


def does_satisfy_constraint(convergence_risk_constraint: Callable,
                            loss_at_beginning: float,
                            loss_at_end: float) -> bool:
    return convergence_risk_constraint(loss_at_beginning=loss_at_beginning, loss_at_end=loss_at_end)


def compute_losses(evaluation_assistant: EvaluationAssistant,
                   learned_algorithm: OptimizationAlgorithm,
                   baseline_algorithm: OptimizationAlgorithm) -> Tuple[NDArray, NDArray, float]:

    losses_of_baseline_algorithm = []
    losses_of_learned_algorithm = []
    number_of_times_constrained_satisfied = 0
    _, convergence_risk_constraint, _ = get_describing_property()

    progress_bar = tqdm(evaluation_assistant.test_set)
    progress_bar.set_description('Compute losses')
    for test_parameter in progress_bar:

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


def compute_losses_over_iterations(algorithm: OptimizationAlgorithm,
                                   evaluation_assistant: EvaluationAssistant,
                                   parameter: dict) -> List[float]:
    algorithm.reset_state_and_iteration_counter()
    current_loss_function = ParametricLossFunction(function=evaluation_assistant.loss_of_algorithm, parameter=parameter)
    algorithm.set_loss_function(current_loss_function)
    loss_over_iterations = [algorithm.evaluate_loss_function_at_current_iterate().item()]
    for i in range(evaluation_assistant.number_of_iterations_for_testing):
        algorithm.perform_step()
        loss_over_iterations.append(algorithm.evaluate_loss_function_at_current_iterate().item())
        if loss_over_iterations[-1] < 1e-16:
            loss_over_iterations.extend(
                [loss_over_iterations[-1]] * (evaluation_assistant.number_of_iterations_for_testing - i - 1)
            )
            break
    return loss_over_iterations


def time_problem(algorithm: OptimizationAlgorithm,
                 loss_function: LossFunction,
                 maximal_number_of_iterations: int,
                 optimal_loss: torch.Tensor,
                 level_of_accuracy: float) -> float:

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
                  ground_truth_losses: List,
                  stop_procedure_after_at_most: int) -> Tuple[dict, dict]:

    levels_of_accuracy = [1e-2, 1e-4, 1e-6]
    times_pac = {epsilon: [0.] for epsilon in levels_of_accuracy}
    times_std = {epsilon: [0.] for epsilon in levels_of_accuracy}

    progress_bar = tqdm(zip(evaluation_assistant.test_set, ground_truth_losses))
    progress_bar.set_description('Compute times')

    for parameter, ground_truth_loss in progress_bar:
        for epsilon in levels_of_accuracy:

            cur_loss_function = ParametricLossFunction(function=evaluation_assistant.loss_of_algorithm,
                                                       parameter=parameter)
            times_pac[epsilon].append(time_problem(
                algorithm=learned_algorithm, loss_function=cur_loss_function,
                maximal_number_of_iterations=stop_procedure_after_at_most, optimal_loss=ground_truth_loss,
                level_of_accuracy=epsilon))

            times_std[epsilon].append(time_problem(
                algorithm=baseline_algorithm, loss_function=cur_loss_function,
                maximal_number_of_iterations=stop_procedure_after_at_most, optimal_loss=ground_truth_loss,
                level_of_accuracy=epsilon))

    return times_pac, times_std


def sanity_check_for_baseline(algorithm: OptimizationAlgorithm, savings_path: str) -> None:

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    iterations = torch.arange(2e4 + 1)
    losses = [algorithm.evaluate_loss_function_at_current_iterate().item()]
    for _ in range(len(iterations) - 1):
        algorithm.perform_step()
        losses.append(algorithm.evaluate_loss_function_at_current_iterate().item())

    losses = np.array(losses)
    ax.plot(iterations.numpy(), losses)
    ax.set_yscale('log')
    ax.set_ylabel('loss')
    ax.set_xlabel('iteration')
    ax.grid('on')

    fig.savefig(savings_path + 'sanity_check.pdf', dpi=300, bbox_inches='tight')


def evaluate_algorithm(loading_path: str, path_of_experiment: str) -> None:
    savings_path = create_folder_for_storing_data(path_of_experiment)
    evaluation_assistant = set_up_evaluation_assistant(loading_path)
    learned_algorithm = evaluation_assistant.set_up_learned_algorithm(
        arguments_of_implementation_class=evaluation_assistant.implementation_arguments)
    baseline_algorithm = get_baseline_algorithm(
        loss_function=learned_algorithm.loss_function, smoothness_constant=evaluation_assistant.smoothness_parameter,
        strong_convexity_constant=evaluation_assistant.strong_convexity_parameter, dim=evaluation_assistant.dim)

    sanity_check_for_baseline(algorithm=baseline_algorithm, savings_path=savings_path)

    losses_of_baseline, losses_of_learned_algorithm, percentage_constrained_satisfied = compute_losses(
        evaluation_assistant=evaluation_assistant, learned_algorithm=learned_algorithm,
        baseline_algorithm=baseline_algorithm
    )

    times_of_learned_algorithm, times_of_baseline_algorithm = compute_times(
        learned_algorithm=learned_algorithm, baseline_algorithm=baseline_algorithm,
        evaluation_assistant=evaluation_assistant,
        stop_procedure_after_at_most=int(1e4),
        ground_truth_losses=[0. for _ in range(len(evaluation_assistant.test_set))])

    save_data(savings_path=savings_path,
              times_of_learned_algorithm=times_of_learned_algorithm,
              losses_of_learned_algorithm=losses_of_learned_algorithm,
              times_of_baseline_algorithm=times_of_baseline_algorithm,
              losses_of_baseline_algorithm=losses_of_baseline,
              ground_truth_losses=[0. for _ in range(len(evaluation_assistant.test_set))],
              percentage_constrained_satisfied=percentage_constrained_satisfied)
