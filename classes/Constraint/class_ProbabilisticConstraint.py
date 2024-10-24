from classes.Constraint.class_Constraint import Constraint, create_list_of_constraints_from_functions
from classes.Constraint.class_BayesianProbabilityEstimator import BayesianProbabilityEstimator


class ProbabilisticConstraint:

    def __init__(self, list_of_constraints):
        self.list_of_constraint = list_of_constraints





def constraint_from_probabilistic_constraint(describing_property, loss_functions, parameters_probabilistic_constraint):

    # Extract
    p_l, p_u = parameters_probabilistic_constraint['p_l'], parameters_probabilistic_constraint['p_u']

    # Build constraint:
    # 1) List of single constraints: Does the algorithm satisfy the describing property?
    list_of_constraints = create_list_of_constraints_from_functions(describing_property=describing_property,
                                                                    list_of_functions=loss_functions)

    # 2) Merge these into quantile_distance probabilistic constraint:
    # Does the algorithm satisfy the reducing property with high probability?
    quantile_distance = parameters_probabilistic_constraint['quantile_distance']
    lower_quantile = parameters_probabilistic_constraint['lower_quantile']
    upper_quantile = parameters_probabilistic_constraint['upper_quantile']
    probabilistic_constraint = ProbabilisticConstraint(list_of_constraints=list_of_constraints,
                                                       quantile_distance=quantile_distance,
                                                       lower_quantile=lower_quantile, upper_quantile=upper_quantile,
                                                       p_l=p_l, p_u=p_u)

    # 3) Build fixed constraint based on this probabilistic constraint:
    # Does the algorithm satisfy the constraint with a probability in the interval [p_l, p_u] ?
    def interval_inside(opt_algo, return_val=False):
        p_m, l_q, u_q, n_iterates = probabilistic_constraint(opt_algo, a=1, b=1)
        print(p_m)
        if return_val:
            return p_l <= p_m <= p_u, p_m  # (l_q >= p_l) and (u_q <= p_u), p_m
        else:
            return p_l <= p_m <= p_u  # (l_q >= p_l) and (u_q <= p_u)

    return Constraint(fun=interval_inside)