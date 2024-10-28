from classes.OptimizationAlgorithm.derived_classes.subclass_ParametricOptimizationAlgorithm import (
    ParametricOptimizationAlgorithm)


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
