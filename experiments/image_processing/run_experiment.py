from experiments.image_processing.training import set_up_and_train_algorithm
from experiments.image_processing.evaluation import evaluate_algorithm
from experiments.image_processing.plotting import create_evaluation_plots
from pathlib import Path
import torch


def create_folder_for_experiment(path_to_experiment_folder: str) -> str:
    path_of_experiment = path_to_experiment_folder + "/image_processing/"
    Path(path_of_experiment).mkdir(parents=True, exist_ok=True)
    return path_of_experiment


def run(path_to_experiment_folder: str) -> None:

    print("Starting image-processing experiment.")
    torch.manual_seed(2)   # If you want to reproduce exactly.
    seed = torch.randint(low=0, high=100, size=(1,)).item()
    torch.manual_seed(seed)

    path_to_images = '/home/michael/Desktop/Experiments/Images/'
    path_of_experiment = create_folder_for_experiment(path_to_experiment_folder)
    print("\tStarting training.")
    set_up_and_train_algorithm(path_of_experiment=path_of_experiment, path_to_images=path_to_images)
    print("\tFinished training.")
    print("\tStarting evaluation.")
    evaluate_algorithm(path_of_experiment=path_of_experiment, loading_path=path_of_experiment + 'data/')
    print("\tFinished evaluation.")
    print("\tCreating evaluation plot.")
    create_evaluation_plots(loading_path=path_of_experiment + 'data/', path_of_experiment=path_of_experiment)

    print("Finished image-processing experiment.")
