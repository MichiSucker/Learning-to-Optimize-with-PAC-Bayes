import torch


class OptimizationAlgorithm:

    def __init__(self, implementation, initial_state, loss_function, constraint=None):
        self.implementation = implementation
        self.loss_function = loss_function
        self.initial_state = initial_state.clone()
        self.current_state = initial_state.clone()
        self.current_iterate = self.current_state[-1]
        self.iteration_counter = 0
        self.constraint = constraint

    def get_initial_state(self):
        return self.initial_state

    def get_implementation(self):
        return self.implementation

    def get_current_state(self):
        return self.current_state

    def get_current_iterate(self):
        return self.current_iterate

    def get_iteration_counter(self):
        return self.iteration_counter

    def set_iteration_counter(self, n):
        if not isinstance(n, int):
            raise TypeError('Iteration counter has to be a non-negative integer.')
        self.iteration_counter = n

    def reset_iteration_counter_to_zero(self):
        self.iteration_counter = 0

    def reset_to_initial_state(self):
        self.current_state = self.initial_state.clone()

    def reset_state_and_iteration_counter(self):
        self.reset_to_initial_state()
        self.reset_iteration_counter_to_zero()

    def set_current_state(self, new_state):
        if new_state.shape != self.current_state.shape:
            raise ValueError('Shape of new state does not match shape of current state.')
        self.current_state = new_state.clone()
        self.current_iterate = self.current_state[-1]

    def set_constraint(self, function):
        self.constraint = function

    def perform_step(self):
        self.iteration_counter += 1
        self.current_iterate = self.implementation.forward(self)
        with torch.no_grad():
            self.implementation.update_state(self)

    def evaluate_loss_function_at_current_iterate(self):
        return self.loss_function(self.current_iterate)

    def evaluate_constraint_at_current_iterate(self):
        return self.constraint(self.current_iterate)
