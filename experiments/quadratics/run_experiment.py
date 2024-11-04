from experiments.quadratics.training import set_up_and_train_algorithm
from experiments.quadratics.evaluation import evaluate_algorithm
from pathlib import Path


def create_folder_for_experiment(path_to_experiment_folder):
    path_of_experiment = path_to_experiment_folder + "/quadratics/"
    Path(path_of_experiment).mkdir(parents=True, exist_ok=True)
    return path_of_experiment


def run(path_to_experiment_folder):

    path_of_experiment = create_folder_for_experiment(path_to_experiment_folder)
    set_up_and_train_algorithm(path_of_experiment=path_of_experiment)
    evaluate_algorithm(path_of_experiment=path_of_experiment, loading_path=path_of_experiment + 'data/')
    # create_evaluation_plots(loading_path=path_of_experiment + '/data/', path_of_experiment=path_of_experiment)