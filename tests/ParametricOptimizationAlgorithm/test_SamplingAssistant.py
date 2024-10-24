import unittest
from classes.OptimizationAlgorithm.derived_classes.subclass_ParametricOptimizationAlgorithm import SamplingAssistant


class TestSamplingAssistant(unittest.TestCase):

    def setUp(self):
        self.learning_rate = 1
        self.sampling_assistant = SamplingAssistant(learning_rate=self.learning_rate)

    def test_creation(self):
        self.assertIsInstance(self.sampling_assistant, SamplingAssistant)

    def test_decay_learning_rate(self):
        iteration = 10
        self.assertEqual(self.sampling_assistant.current_learning_rate, self.sampling_assistant.initial_learning_rate)
        self.sampling_assistant.decay_learning_rate(iteration=iteration)
        self.assertEqual(self.sampling_assistant.current_learning_rate,
                         self.sampling_assistant.initial_learning_rate / iteration)
