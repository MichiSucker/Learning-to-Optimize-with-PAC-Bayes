from classes.OptimizationAlgorithm.derived_classes.subclass_ParametricOptimizationAlgorithm import (
    ParametricOptimizationAlgorithm)
import torch
from tqdm import tqdm


class PacBayesOptimizationAlgorithm(ParametricOptimizationAlgorithm):

    def __init__(self, initial_state, implementation, loss_function, pac_parameters, constraint=None):
        super().__init__(initial_state=initial_state,
                         implementation=implementation,
                         loss_function=loss_function,
                         constraint=constraint)
        self.sufficient_statistics = pac_parameters['sufficient_statistics']
        self.natural_parameters = pac_parameters['natural_parameters']
        self.covering_number = pac_parameters['covering_number']
        self.epsilon = pac_parameters['epsilon']
        self.n_max = pac_parameters['n_max']
        self.pac_bound = None
        self.optimal_lambda = None

    def evaluate_sufficient_statistics_on_all_parameters_and_hyperparameters(
            self, list_of_parameters, list_of_hyperparameters, estimated_convergence_probabilities):

        values_of_sufficient_statistics = torch.zeros((len(list_of_parameters), len(list_of_hyperparameters), 2))
        pbar = tqdm(enumerate(list_of_hyperparameters))
        pbar.set_description('Compute Sufficient Statistics')
        for j, current_hyperparameters in pbar:

            self.set_hyperparameters_to(current_hyperparameters)
            for i, current_parameters in enumerate(list_of_parameters):
                values_of_sufficient_statistics[i, j, :] = self.sufficient_statistics(
                    self, parameter=current_parameters, probability=estimated_convergence_probabilities[j])

        # Note that we have to take the mean over parameters here, as the Pac-Bound holds for the empirical mean and
        # one cannot exchange exp and summation.
        return torch.mean(values_of_sufficient_statistics, dim=0)

    def get_upper_bound_as_function_of_lambda(self, potentials):
        return lambda lamb: -(torch.logsumexp(potentials(lamb), dim=0)
                              + torch.log(self.epsilon)
                              - torch.log(self.covering_number)) / (self.natural_parameters(lamb)[0])
