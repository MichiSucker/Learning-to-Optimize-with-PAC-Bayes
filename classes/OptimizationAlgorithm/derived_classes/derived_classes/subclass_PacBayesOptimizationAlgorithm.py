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

    def compute_pac_bound(self,
                          samples_prior,
                          potentials_prior,
                          estimated_convergence_probabilities,
                          list_of_parameters_train):

        potentials_posterior = self.get_posterior_potentials_as_function_of_lambda(
            list_of_parameters_train=list_of_parameters_train,
            samples_prior=samples_prior,
            estimated_convergence_probabilities=estimated_convergence_probabilities,
            potentials_prior=potentials_prior
        )
        upper_bound_as_function_of_lambda = self.get_upper_bound_as_function_of_lambda(potentials=potentials_posterior)
        best_value, best_lambda = self.minimize_upper_bound_in_lambda(upper_bound_as_function_of_lambda)
        self.set_variable__optimal_lambda__to(best_lambda)
        self.set_variable__pac_bound__to(best_value)
        return potentials_posterior(best_lambda)

    def get_posterior_potentials_as_function_of_lambda(self,
                                                       list_of_parameters_train,
                                                       samples_prior,
                                                       estimated_convergence_probabilities,
                                                       potentials_prior):

        values_sufficient_statistics = self.evaluate_sufficient_statistics_on_all_parameters_and_hyperparameters(
            list_of_parameters=list_of_parameters_train,
            list_of_hyperparameters=samples_prior,
            estimated_convergence_probabilities=estimated_convergence_probabilities
        )

        return lambda x: torch.matmul(values_sufficient_statistics, self.natural_parameters(x)) + potentials_prior

    def get_upper_bound_as_function_of_lambda(self, potentials):
        return lambda lamb: -(torch.logsumexp(potentials(lamb), dim=0)
                              + torch.log(self.epsilon)
                              - torch.log(self.covering_number)) / (self.natural_parameters(lamb)[0])

    def minimize_upper_bound_in_lambda(self, upper_bound):
        capital_lambda = torch.linspace(start=1e-8, end=1e2, steps=int(self.covering_number))
        values_upper_bound = torch.stack([upper_bound(lamb) for lamb in capital_lambda])
        best_lambda = capital_lambda[torch.argmin(values_upper_bound)]
        if best_lambda == capital_lambda[0]:
            print("Note: Optimal lambda found at left boundary!")
        if best_lambda == capital_lambda[-1]:
            print("Note: Optimal lambda found at right boundary!")
        best_value = torch.min(values_upper_bound)
        return best_value, best_lambda

    def set_variable__pac_bound__to(self, pac_bound):
        if self.pac_bound is None:
            self.pac_bound = pac_bound
        else:
            raise Exception("PAC-bound already set.")

    def set_variable__optimal_lambda__to(self, optimal_lambda):
        if self.optimal_lambda is None:
            self.optimal_lambda = optimal_lambda
        else:
            raise Exception("Optimal lambda already set.")
