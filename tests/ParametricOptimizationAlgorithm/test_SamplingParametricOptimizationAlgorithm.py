import unittest
import io
import sys
from classes.OptimizationAlgorithm.derived_classes.subclass_ParametricOptimizationAlgorithm import (
    ParametricOptimizationAlgorithm, compute_initialization_loss, TrajectoryRandomizer, InitializationAssistant)
import torch
from algorithms.dummy import Dummy, NonTrainableDummy
from classes.LossFunction.class_LossFunction import LossFunction
import copy


def dummy_function(x):
    return 0.5 * torch.linalg.norm(x) ** 2


class TestSamplingParametricOptimizationAlgorithm(unittest.TestCase):

    def setUp(self):
        self.dim = torch.randint(low=1, high=1000, size=(1,)).item()
        self.length_state = 1  # Take one, because it has to be compatible with Dummy()
        self.initial_state = torch.randn(size=(self.length_state, self.dim))
        self.current_state = self.initial_state.clone()
        self.loss_function = LossFunction(function=dummy_function)
        self.optimization_algorithm = ParametricOptimizationAlgorithm(implementation=Dummy(),
                                                                      initial_state=self.initial_state,
                                                                      loss_function=self.loss_function)
