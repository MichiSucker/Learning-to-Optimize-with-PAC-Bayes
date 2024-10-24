from sympy.plotting.intervalmath import interval

from classes.Constraint.class_Constraint import Constraint, create_list_of_constraints_from_functions
from classes.Constraint.class_BayesianProbabilityEstimator import BayesianProbabilityEstimator


class ProbabilisticConstraint:

    def __init__(self, list_of_constraints, parameters_of_estimation):
        self.list_of_constraints = list_of_constraints
        self.parameters_of_estimation = parameters_of_estimation
        self.bayesian_estimator = BayesianProbabilityEstimator(list_of_constraints=self.list_of_constraints,
                                                               parameters_of_estimation=parameters_of_estimation)
        self.constraint = self.create_constraint()

    def create_constraint(self):

        def posterior_mean_inside_interval(opt_algo, also_return_value=False):
            posterior_mean, lower_quantile, upper_quantile, number_of_iterates = (
                self.bayesian_estimator.estimate_probability(input_to_constraint=opt_algo))
            if also_return_value:
                return (self.parameters_of_estimation['probabilities'][0]
                        <= posterior_mean
                        <= self.parameters_of_estimation['probabilities'][1],
                        posterior_mean)
            else:
                return (self.parameters_of_estimation['probabilities'][0]
                        <= posterior_mean
                        <= self.parameters_of_estimation['probabilities'][1])

        return Constraint(function=posterior_mean_inside_interval)
