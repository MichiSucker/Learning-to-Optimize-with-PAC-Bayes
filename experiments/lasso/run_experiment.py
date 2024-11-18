from experiments.lasso.training import set_up_and_train_algorithm
from experiments.lasso.evaluation import evaluate_algorithm
from experiments.lasso.plotting import create_evaluation_plots
from pathlib import Path
import torch


def create_folder_for_experiment(path_to_experiment_folder: str) -> str:
    path_of_experiment = path_to_experiment_folder + "/lasso/"
    Path(path_of_experiment).mkdir(parents=True, exist_ok=True)
    return path_of_experiment


def run(path_to_experiment_folder: str) -> None:
    # This is pretty important again. Also, it makes sure that all tensor types do match.
    torch.set_default_dtype(torch.float64)
    path_of_experiment = create_folder_for_experiment(path_to_experiment_folder)
    set_up_and_train_algorithm(path_of_experiment=path_of_experiment)
    evaluate_algorithm(path_of_experiment=path_of_experiment, loading_path=path_of_experiment + 'data/')
    create_evaluation_plots(loading_path=path_of_experiment + 'data/', path_of_experiment=path_of_experiment)
