import copy
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm


class ConstraintChecker:

    def __init__(self,
                 check_constraint_every: int,
                 there_is_a_constraint: bool):
        self.check_constraint_every = check_constraint_every
        self.there_is_a_constraint = there_is_a_constraint
        self.found_point_inside_constraint = False
        self.point_inside_constraint = None

    def should_check_constraint(self, iteration_number: int) -> bool:
        if (self.there_is_a_constraint
            and (iteration_number >= 1)
                and (iteration_number % self.check_constraint_every == 0)):
            return True
        else:
            return False

    def set_variable__there_is_a_constraint__to(self, new_bool: bool) -> None:
        self.there_is_a_constraint = new_bool

    def set_variable__check_constraint_every__to(self, new_number: int) -> None:
        self.check_constraint_every = new_number

    def update_point_inside_constraint_or_reject(self, optimization_algorithm: OptimizationAlgorithm) -> None:
        satisfies_constraint = optimization_algorithm.evaluate_constraint()

        if satisfies_constraint:
            self.found_point_inside_constraint = True
            self.point_inside_constraint = copy.deepcopy(optimization_algorithm.implementation.state_dict())

        elif self.found_point_inside_constraint and (not satisfies_constraint):
            optimization_algorithm.implementation.load_state_dict(self.point_inside_constraint)

    def final_check(self, optimization_algorithm: OptimizationAlgorithm) -> None:
        if optimization_algorithm.constraint is not None:
            satisfies_constraint = optimization_algorithm.evaluate_constraint()
            if satisfies_constraint:
                return
            elif self.found_point_inside_constraint and (not satisfies_constraint):
                optimization_algorithm.implementation.load_state_dict(self.point_inside_constraint)
            else:
                raise Exception("Did not find a point that lies within the constraint!")
