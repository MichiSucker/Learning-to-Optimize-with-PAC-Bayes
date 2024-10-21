import torch


class OptimizationAlgorithm:

    def __init__(self, implementation, initial_state):
        self.implementation = implementation
        self.initial_state = initial_state

    def get_initial_state(self):
        return self.initial_state

    def get_implementation(self):
        return self.implementation
