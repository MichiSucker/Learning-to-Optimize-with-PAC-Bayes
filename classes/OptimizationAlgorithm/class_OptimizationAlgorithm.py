import torch


class OptimizationAlgorithm:

    def __init__(self, implementation, initial_state):
        self.implementation = implementation
        self.initial_state = initial_state.clone()
        self.current_state = initial_state.clone()
        self.current_iterate = self.current_state[-1]
        self.iteration_counter = 0

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
        self.iteration_counter = n

    def reset_iteration_counter_to_zero(self):
        self.iteration_counter = 0

    def reset_to_initial_state(self):
        self.current_state = self.initial_state.clone()

    def set_current_state(self, new_state):
        self.current_state = new_state.clone()
        self.current_iterate = self.current_state[-1]

    def perform_step(self):
        self.iteration_counter += 1
        self.current_iterate = self.implementation.forward(self)
        with torch.no_grad():
            self.implementation.update_state(self)
