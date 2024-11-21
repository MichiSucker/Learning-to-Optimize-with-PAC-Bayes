import torch
from typing import List, Callable, Tuple
from pathlib import Path
from classes.LossFunction.class_LossFunction import LossFunction
from classes.OptimizationAlgorithm.derived_classes.subclass_ParametricOptimizationAlgorithm import \
    ParametricOptimizationAlgorithm
from experiments.quadratics.data_generation import get_data
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from experiments.quadratics.algorithm import Quadratics
from algorithms.heavy_ball import HeavyBallWithFriction
import numpy as np
import copy
from tqdm import tqdm
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from experiments.helpers import set_size


def get_number_of_datapoints() -> dict:
    return {'prior': 250, 'train': 250, 'test': 250, 'validation': 250}


def create_folder_for_storing_data(path_of_experiment: str) -> str:
    savings_path = path_of_experiment + "/train_with_ratio_of_losses/data/"
    Path(savings_path).mkdir(parents=True, exist_ok=True)
    return savings_path


def create_parametric_loss_functions_from_parameters(template_loss_function: Callable, parameters: dict) -> dict:
    loss_functions = {
        'prior': [ParametricLossFunction(function=template_loss_function, parameter=p) for p in parameters['prior']],
        'train': [ParametricLossFunction(function=template_loss_function, parameter=p) for p in parameters['train']],
        'test': [ParametricLossFunction(function=template_loss_function, parameter=p) for p in parameters['test']],
        'validation': [ParametricLossFunction(function=template_loss_function, parameter=p)
                       for p in parameters['validation']],
    }
    return loss_functions


def get_initial_state(dim: int) -> torch.Tensor:
    return torch.zeros((2, dim))


def get_constraint_parameters(number_of_training_iterations: int) -> dict:
    return {'describing_property': None,
            'num_iter_update_constraint': number_of_training_iterations + 1}


def get_update_parameters() -> dict:
    return {'num_iter_print_update': 1000,
            'with_print': True,
            'bins': [1e6, 1e4, 1e2, 1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10][::-1]}


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


def get_algorithm_for_learning(loss_functions: dict,
                               dimension_of_hyperparameters: int) -> ParametricOptimizationAlgorithm:

    initial_state = get_initial_state(dim=dimension_of_hyperparameters)
    algorithm_for_learning = ParametricOptimizationAlgorithm(initial_state=initial_state,
                                                             implementation=Quadratics(),
                                                             loss_function=loss_functions['prior'][0])
    return algorithm_for_learning


def get_baseline_algorithm(loss_function: LossFunction,
                           smoothness_constant: torch.Tensor,
                           strong_convexity_constant: torch.Tensor,
                           dim: int) -> OptimizationAlgorithm:

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


def compare_algorithms(algorithm_ratios: ParametricOptimizationAlgorithm,
                       algorithm_sum: ParametricOptimizationAlgorithm,
                       loss_functions_to_test: List[LossFunction]
                       ) -> Tuple[NDArray, NDArray]:

    n_test = 2 * algorithm_ratios.n_max
    losses_ratio, losses_sum = [], []
    progress_bar = tqdm(loss_functions_to_test)
    progress_bar.set_description("Compare algorithms")
    for f in progress_bar:

        # Initialize.
        algorithm_ratios.reset_state_and_iteration_counter()
        algorithm_sum.reset_state_and_iteration_counter()
        algorithm_ratios.set_loss_function(f)
        algorithm_sum.set_loss_function(f)
        cur_losses_ratio = [algorithm_ratios.evaluate_loss_function_at_current_iterate().item()]
        cur_losses_sum = [algorithm_sum.evaluate_loss_function_at_current_iterate().item()]

        # Run algorithm the way we proposed to do.
        for i in range(n_test):
            algorithm_ratios.perform_step()
            cur_losses_ratio.append(
                algorithm_ratios.evaluate_loss_function_at_current_iterate().item())

            # If loss got too small, we stop, as this leads to numerical instabilities
            if cur_losses_ratio[-1] < 1e-16:
                cur_losses_ratio.extend(
                    [cur_losses_ratio[-1]] * (n_test - i - 1)
                )
                break
        losses_ratio.append(cur_losses_ratio)

        # Run algorithm with sum of losses
        for i in range(n_test):
            algorithm_sum.perform_step()
            cur_losses_sum.append(
                algorithm_sum.evaluate_loss_function_at_current_iterate().item())

            # If loss got too small, we stop, as this leads to numerical instabilities
            if cur_losses_sum[-1] < 1e-16:
                cur_losses_sum.extend(
                    [cur_losses_sum[-1]] * (n_test - i - 1)
                )
                break
        losses_sum.append(cur_losses_sum)

    return np.array(losses_ratio), np.array(losses_sum)


def plot_results(loading_path):

    width = 469.75499  # Arxiv
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts quantile_distance little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    }
    plt.rcParams.update(tex_fonts)

    names = {'with': 'with', 'without': 'without', 'other': 'other'}
    colors = {'with': '#57cc99', 'without': '#7209b7', 'other': '#f72585'}

    number_of_iterations_training = np.load(loading_path + 'number_of_iterations_training.npy')
    number_of_iterations_testing = np.load(loading_path + 'number_of_iterations_testing.npy')
    losses_with_randomization_test = np.load(loading_path + 'losses_ratio_test.npy')
    losses_without_randomization_test = np.load(loading_path + 'losses_sum_test.npy')
    losses_with_randomization_train = np.load(loading_path + 'losses_ratio_train.npy')
    losses_without_randomization_train = np.load(loading_path + 'losses_sum_train.npy')

    iterates = np.arange(number_of_iterations_testing + 1)

    subplots = (2, 4)
    size = set_size(width=width, subplots=subplots)
    fig = plt.figure(figsize=size)
    G = gridspec.GridSpec(subplots[0], subplots[1])

    axes_1 = fig.add_subplot(G[:, 2:])
    axes_2 = fig.add_subplot(G[:, 0:2])

    axes_1.plot(iterates, np.mean(losses_with_randomization_test, axis=0),
                linestyle='dashed', label=names['with'], color=colors['with'])
    axes_1.plot(iterates, np.median(losses_with_randomization_test, axis=0),
                linestyle='dotted', color=colors['with'])
    axes_1.fill_between(iterates,
                        np.quantile(losses_with_randomization_test, q=0, axis=0),
                        np.quantile(losses_with_randomization_test, q=0.95, axis=0),
                        color=colors['with'], alpha=0.5)

    axes_1.plot(iterates, np.mean(losses_without_randomization_test, axis=0),
                linestyle='dashed', label=names['without'], color=colors['without'])
    axes_1.plot(iterates, np.median(losses_without_randomization_test, axis=0),
                linestyle='dotted', color=colors['without'])
    axes_1.fill_between(iterates,
                        np.quantile(losses_without_randomization_test, q=0, axis=0),
                        np.quantile(losses_without_randomization_test, q=0.95, axis=0),
                        color=colors['without'], alpha=0.5)

    axes_1.axvline(x=number_of_iterations_training, ymin=0, ymax=1, linestyle='dashed', alpha=0.5,
                   label='$n_{\\rm{train}}$', color='black')

    axes_1.set_ylabel('$\ell(x^{(k)}, \\theta)$')
    axes_1.set_xlabel('$n_{\\rm{it}}$')
    axes_1.set_yscale('log')
    axes_1.set_title('Test')
    axes_1.legend()
    axes_1.grid('on')

    axes_2.plot(iterates, np.mean(losses_with_randomization_train, axis=0),
                linestyle='dashed', label=names['with'], color=colors['with'])
    axes_2.plot(iterates, np.median(losses_with_randomization_train, axis=0),
                linestyle='dotted', color=colors['with'])
    axes_2.fill_between(iterates,
                        np.quantile(losses_with_randomization_train, q=0, axis=0),
                        np.quantile(losses_with_randomization_train, q=0.95, axis=0),
                        color=colors['with'], alpha=0.5)

    axes_2.plot(iterates, np.mean(losses_without_randomization_train, axis=0),
                linestyle='dashed', label=names['without'], color=colors['without'])
    axes_2.plot(iterates, np.median(losses_without_randomization_train, axis=0),
                linestyle='dotted', color=colors['without'])
    axes_2.fill_between(iterates,
                        np.quantile(losses_without_randomization_train, q=0, axis=0),
                        np.quantile(losses_without_randomization_train, q=0.95, axis=0),
                        color=colors['without'], alpha=0.5)

    axes_2.axvline(x=number_of_iterations_training, ymin=0, ymax=1, linestyle='dashed', alpha=0.5,
                   label='$n_{\\rm{train}}$', color='black')

    axes_2.set_ylabel('$\ell(x^{(k)}, \\theta)$')
    axes_2.set_xlabel('$n_{\\rm{it}}$')
    axes_2.set_yscale('log')
    axes_2.set_title('Train')
    axes_2.legend()
    axes_2.grid('on')

    plt.tight_layout()
    fig.savefig(loading_path + 'comparison_ratio_of_losses.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(loading_path + 'comparison_ratio_of_losses.png', dpi=300, bbox_inches='tight')


def set_up_and_train_algorithm(path_of_experiment: str) -> None:
    # This is pretty important! Without increased accuracy, the model will struggle to train, because at some point
    # (about loss of 1e-6) the incurred losses are subject to numerical instabilities, which do not provide meaningful
    # information for learning.
    torch.set_default_dtype(torch.double)
    savings_path = create_folder_for_storing_data(path_of_experiment)
    n_max = 200
    parameters, loss_function_of_algorithm, mu_min, L_max, dim = get_data(get_number_of_datapoints())
    loss_functions = create_parametric_loss_functions_from_parameters(
        template_loss_function=loss_function_of_algorithm, parameters=parameters)
    baseline_algorithm = get_baseline_algorithm(
        loss_function=loss_functions['prior'][0], smoothness_constant=L_max, strong_convexity_constant=mu_min, dim=dim)

    algorithm_ratio = get_algorithm_for_learning(
        loss_functions=loss_functions, dimension_of_hyperparameters=dim)
    algorithm_ratio.n_max = n_max
    algorithm_ratio.initialize_with_other_algorithm(
        other_algorithm=baseline_algorithm, loss_functions=loss_functions['prior'],
        parameters_of_initialization=get_initialization_parameters())

    # Try to eliminate effect of initialization by loading state-dict (i.e. hyperparameters) of other algorithm.
    algorithm_sum = copy.deepcopy(algorithm_ratio)

    # Train first algorithm with ratio
    fitting_parameters = get_fitting_parameters(maximal_number_of_iterations=n_max)
    number_of_training_iterations = fitting_parameters['n_max']
    constraint_parameters = get_constraint_parameters(number_of_training_iterations=number_of_training_iterations)
    algorithm_ratio.fit(loss_functions=loss_functions['prior'],
                        fitting_parameters=fitting_parameters,
                        constraint_parameters=constraint_parameters,
                        update_parameters=get_update_parameters())

    # Train the other with sum of losses
    algorithm_sum.fit_with_function_values(loss_functions=loss_functions['prior'],
                                           fitting_parameters=fitting_parameters,
                                           constraint_parameters=constraint_parameters,
                                           update_parameters=get_update_parameters())

    losses_ratio_test, losses_sum_test = compare_algorithms(
        algorithm_ratios=algorithm_ratio, algorithm_sum=algorithm_sum,
        loss_functions_to_test=loss_functions['test'])

    losses_ratio_train, losses_sum_train = compare_algorithms(
        algorithm_ratios=algorithm_ratio, algorithm_sum=algorithm_sum,
        loss_functions_to_test=loss_functions['prior'])

    save_data(savings_path=savings_path, number_of_iterations_training=n_max, number_of_iterations_testing=2 * n_max,
              losses_ratio_test=losses_ratio_test, losses_sum_test=losses_sum_test,
              losses_ratio_train=losses_ratio_train, losses_sum_train=losses_sum_train)


def save_data(savings_path: str,
              number_of_iterations_training: int,
              number_of_iterations_testing: int,
              losses_ratio_test: NDArray,
              losses_sum_test: NDArray,
              losses_ratio_train: NDArray,
              losses_sum_train: NDArray
              ) -> None:

    np.save(savings_path + 'number_of_iterations_training', number_of_iterations_training)
    np.save(savings_path + 'number_of_iterations_testing', number_of_iterations_testing)
    np.save(savings_path + 'losses_ratio_test', losses_ratio_test)
    np.save(savings_path + 'losses_sum_test', losses_sum_test)
    np.save(savings_path + 'losses_ratio_train', losses_ratio_train)
    np.save(savings_path + 'losses_sum_train', losses_sum_train)

