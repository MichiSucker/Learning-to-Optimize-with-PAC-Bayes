import torch
from pathlib import Path
from experiments.quadratics.data_generation import get_data
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction


def get_number_of_datapoints():
    return {'prior': 250, 'train': 250, 'test': 250, 'validation': 250}


def create_folder_for_storing_data(path_of_experiment):
    savings_path = path_of_experiment + "/data/"
    Path(savings_path).mkdir(parents=True, exist_ok=True)
    return savings_path


def create_parametric_loss_functions_from_parameters(template_loss_function, parameters):
    loss_functions = {
        'prior': [ParametricLossFunction(function=template_loss_function, parameter=p) for p in parameters['prior']],
        'train': [ParametricLossFunction(function=template_loss_function, parameter=p) for p in parameters['train']],
        'test': [ParametricLossFunction(function=template_loss_function, parameter=p) for p in parameters['test']],
        'validation': [ParametricLossFunction(function=template_loss_function, parameter=p)
                       for p in parameters['validation']],
    }
    return loss_functions


def set_up_and_train_algorithm(path_of_experiment):

    savings_path = create_folder_for_storing_data(path_of_experiment)
    parameters, loss_function_of_algorithm, mu_min, L_max, dim = get_data(get_number_of_datapoints())
    loss_functions = create_parametric_loss_functions_from_parameters(
        template_loss_function=loss_function_of_algorithm, parameters=parameters)
