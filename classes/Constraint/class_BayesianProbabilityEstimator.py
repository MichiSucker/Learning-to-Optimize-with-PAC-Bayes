import numpy as np
from scipy.stats import beta
from typing import List, Tuple, Callable, Dict, Any
from classes.Constraint.class_Constraint import Constraint


class BayesianProbabilityEstimator:

    def __init__(self,
                 list_of_constraints: List[Constraint],
                 parameters_of_estimation: Dict):
        self.list_of_constraints = list_of_constraints
        self.parameters_of_estimation = parameters_of_estimation
        self.quantile_distance = parameters_of_estimation['quantile_distance']
        self.lower_quantile = parameters_of_estimation['quantiles'][0]
        self.upper_quantile = parameters_of_estimation['quantiles'][1]
        self.lower_probability = parameters_of_estimation['probabilities'][0]
        self.upper_probability = parameters_of_estimation['probabilities'][1]

    def get_parameters_of_estimation(self) -> dict:
        return self.parameters_of_estimation

    def get_list_of_constraints(self) -> List[Constraint]:
        return self.list_of_constraints

    def set_list_of_constraints(self, new_list_of_constraints: List[Constraint]) -> None:
        self.list_of_constraints = new_list_of_constraints

    def set_parameters_of_estimation(self, new_parameters: dict) -> None:
        if not (('quantile_distance' in new_parameters.keys())
                and ('quantiles' in new_parameters.keys())
                and ('probabilities' in new_parameters.keys())):
            raise ValueError('Missing parameters.')
        if not (check_quantile_distance(new_parameters['quantile_distance'])
                and check_quantiles(new_parameters['quantiles'])
                and check_probabilities(new_parameters['probabilities'])):
            raise ValueError('Invalid parameters.')

        self.parameters_of_estimation = new_parameters
        self.quantile_distance = new_parameters['quantile_distance']
        self.lower_quantile = new_parameters['quantiles'][0]
        self.upper_quantile = new_parameters['quantiles'][1]
        self.lower_probability = new_parameters['probabilities'][0]
        self.upper_probability = new_parameters['probabilities'][1]

    def set_quantile_distance(self, quantile_distance: float) -> None:
        if not check_quantile_distance(quantile_distance):
            raise ValueError('Invalid quantile distance.')
        self.parameters_of_estimation['quantile_distance'] = quantile_distance
        self.quantile_distance = quantile_distance

    def get_quantile_distance(self) -> float:
        return self.quantile_distance

    def get_quantiles(self) -> Tuple[float, float]:
        return self.lower_quantile, self.upper_quantile

    def get_probabilities(self) -> Tuple[float, float]:
        return self.lower_probability, self.upper_probability

    def set_quantiles(self, quantiles: Tuple[float, float]) -> None:
        if not check_quantiles(quantiles):
            raise ValueError('Invalid quantiles.')
        self.parameters_of_estimation['quantiles'] = quantiles
        self.lower_quantile = quantiles[0]
        self.upper_quantile = quantiles[1]

    def set_probabilities(self, probabilities: Tuple[float, float]) -> None:
        if not check_probabilities(probabilities):
            raise ValueError('Invalid probabilities.')
        self.parameters_of_estimation['probabilities'] = probabilities
        self.lower_probability = probabilities[0]
        self.upper_probability = probabilities[1]

    def estimate_probability(self, input_to_constraint: Any) -> Tuple[float, float, float, int]:

        # Setup non-informative prior
        a, b = 1, 1
        prior = beta(a=a, b=b)
        current_upper_quantile, current_lower_quantile = prior.ppf(self.upper_quantile), prior.ppf(self.lower_quantile)
        n_iterates = 0

        while current_upper_quantile - current_lower_quantile > self.quantile_distance:

            n_iterates += 1
            result = sample_and_evaluate_random_constraint(input_to_constraint=input_to_constraint,
                                                           list_of_constraints=self.list_of_constraints)
            a, b, current_upper_quantile, current_lower_quantile = update_parameters_and_uncertainty(
                result=result, a=a, b=b, upper_quantile=self.upper_quantile, lower_quantile=self.lower_quantile)

            # To-Do: Refactor this function into smaller ones (too many parameters!)
            if ((self.upper_probability is not None) and (self.lower_probability is not None)
                    and estimation_should_be_stopped(
                        current_upper_quantile=current_upper_quantile,
                        current_lower_quantile=current_lower_quantile,
                        current_posterior_mean=(a/(a + b)),
                        desired_upper_probability=self.upper_probability,
                        desired_lower_probability=self.lower_probability,
                        desired_quantile_distance=self.quantile_distance)):
                break

        # Compute posterior mean (closed form, since still Beta distribution)
        posterior_mean = a / (a + b)
        return posterior_mean, current_lower_quantile, current_upper_quantile, n_iterates


def check_quantile_distance(quantile_distance: float) -> bool:
    if 0 <= quantile_distance <= 1:
        return True
    else:
        return False


def check_quantiles(quantiles: tuple) -> bool:
    if (quantiles[0] <= quantiles[1]) and (0 <= quantiles[0] <= 1) and (0 <= quantiles[1] <= 1):
        return True
    else:
        return False


def check_probabilities(probabilities: tuple) -> bool:
    if (probabilities[0] <= probabilities[1]) and (0 <= probabilities[0] <= 1) and (0 <= probabilities[1] <= 1):
        return True
    else:
        return False


def sample_and_evaluate_random_constraint(input_to_constraint: Any, list_of_constraints: list[Callable]) -> int:
    if len(list_of_constraints) == 0:
        raise ValueError('There are no constraints to evaluate.')
    idx = np.random.randint(low=0, high=len(list_of_constraints))
    cur_fun = list_of_constraints[idx]
    return int(cur_fun(input_to_constraint))


def update_parameters_and_uncertainty(result: int,
                                      a: int,
                                      b: int,
                                      upper_quantile: float,
                                      lower_quantile: float) -> Tuple[int, int, float, float]:
    a += result
    b += 1 - result
    posterior = beta(a=a, b=b)
    current_upper_quantile = posterior.ppf(upper_quantile)
    current_lower_quantile = posterior.ppf(lower_quantile)
    return a, b, current_upper_quantile, current_lower_quantile


def estimation_should_be_stopped(current_upper_quantile: float,
                                 current_lower_quantile: float,
                                 current_posterior_mean: float,
                                 desired_upper_probability: float,
                                 desired_lower_probability: float,
                                 desired_quantile_distance: float) -> bool:
    current_quantile_distance = current_upper_quantile - current_lower_quantile
    small_quantile_distance_and_too_high_probability = ((current_quantile_distance < 2 * desired_quantile_distance)
                                                        and (current_lower_quantile > desired_upper_probability))
    small_quantile_distance_and_too_low_probability = ((current_quantile_distance < 2 * desired_quantile_distance)
                                                       and (current_upper_quantile < desired_lower_probability))
    mean_way_too_high = (current_posterior_mean - current_quantile_distance > desired_upper_probability)
    mean_way_too_low = (current_posterior_mean + current_quantile_distance < desired_lower_probability)

    # Stop the process earlier, if it is highly unlikely that the true probability lies within [p_l, p_u].
    if small_quantile_distance_and_too_high_probability or mean_way_too_high:
        return True
    elif small_quantile_distance_and_too_low_probability or mean_way_too_low:
        return True
    else:
        return False
