from experiments.quadratics.training import set_up_and_train_algorithm
from experiments.quadratics.evaluation import evaluate_algorithm
from experiments.quadratics.plotting import create_evaluation_plots
from pathlib import Path
import torch


def create_folder_for_experiment(path_to_experiment_folder: str) -> str:
    path_of_experiment = path_to_experiment_folder + "/quadratics/"
    Path(path_of_experiment).mkdir(parents=True, exist_ok=True)
    return path_of_experiment


def run(path_to_experiment_folder: str) -> None:

    print("Starting experiment on quadratic functions.")
    torch.manual_seed(17)   # If you want to reproduce exactly.

    # This is pretty important! Without increased accuracy, the model will struggle to train, because at some point
    # (about loss of 1e-6) the incurred losses are subject to numerical instabilities, which do not provide meaningful
    # information for learning.
    torch.set_default_dtype(torch.double)

    path_of_experiment = create_folder_for_experiment(path_to_experiment_folder)
    set_up_and_train_algorithm(path_of_experiment=path_of_experiment)
    evaluate_algorithm(path_of_experiment=path_of_experiment, loading_path=path_of_experiment + 'data/')
    create_evaluation_plots(loading_path=path_of_experiment + '/data/', path_of_experiment=path_of_experiment)
    print("Finished experiment on quadratic functions.")
