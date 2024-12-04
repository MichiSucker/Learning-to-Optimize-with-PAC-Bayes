from experiments.nn_training.training import set_up_and_train_algorithm
from experiments.nn_training.evaluation import evaluate_algorithm
from experiments.nn_training.plotting import create_evaluation_plots
from pathlib import Path
import torch


def create_folder_for_experiment(path_to_experiment_folder: str) -> str:
    path_of_experiment = path_to_experiment_folder + "/neural_network_training/"
    Path(path_of_experiment).mkdir(parents=True, exist_ok=True)
    return path_of_experiment


def run(path_to_experiment_folder: str) -> None:

    print("Starting experiment on training a neural network.")
    torch.manual_seed(47)  # This is for exact reproducibility.

    # This is pretty important! Without increased accuracy, the model will struggle to train, because at some point
    # (about loss of 1e-6) the incurred losses are subject to numerical instabilities, which do not provide meaningful
    # information for learning.
    torch.set_default_dtype(torch.double)

    path_of_experiment = create_folder_for_experiment(path_to_experiment_folder)
    set_up_and_train_algorithm(path_of_experiment=path_of_experiment)
    evaluate_algorithm(path_of_experiment=path_of_experiment, loading_path=path_of_experiment + '/data/')
    create_evaluation_plots(loading_path=path_of_experiment + 'data/', path_of_experiment=path_of_experiment)
    print("Finished experiment on training a neural network.")
