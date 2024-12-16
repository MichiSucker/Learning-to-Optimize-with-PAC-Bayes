from experiments.mnist.training import set_up_and_train_algorithm
from experiments.mnist.evaluation import evaluate_algorithm
from experiments.mnist.plotting import create_evaluation_plots
from pathlib import Path
import torch


def create_folder_for_experiment(path_to_experiment_folder: str) -> str:
    path_of_experiment = path_to_experiment_folder + "/mnist/"
    Path(path_of_experiment).mkdir(parents=True, exist_ok=True)
    return path_of_experiment


def run(path_to_experiment_folder: str) -> None:
    # DISCLAIMER: To reproduce the exact results, you first have to change 1e-12 to 1e-5
    # in ParametricOptimizationAlgorithm/compute_ratio_of_losses. There seems to be some instabilities with CEL for too
    # small values.
    print("Starting experiment on MNIST.")

    # If you want to reproduce exactly.
    torch.manual_seed(5)
    seed = torch.randint(low=0, high=100, size=(1,)).item()
    torch.manual_seed(seed)

    # This is pretty important! Without increased accuracy, the model will struggle to train, because at some point
    # (about loss of 1e-6) the incurred losses are subject to numerical instabilities, which do not provide meaningful
    # information for learning.
    torch.set_default_dtype(torch.double)

    path_of_experiment = create_folder_for_experiment(path_to_experiment_folder)
    print("\tStarting training.")
    set_up_and_train_algorithm(path_of_experiment=path_of_experiment)
    print("\tFinished training.")
    print("\tStarting evaluation.")
    evaluate_algorithm(loading_path=path_of_experiment + '/data/', path_of_experiment=path_of_experiment)
    print("\tFinished evaluation.")
    print("\tCreating evaluation plot.")
    create_evaluation_plots(loading_path=path_of_experiment + '/data/', path_of_experiment=path_of_experiment)
    print("Finished experiment on MNIST.")
