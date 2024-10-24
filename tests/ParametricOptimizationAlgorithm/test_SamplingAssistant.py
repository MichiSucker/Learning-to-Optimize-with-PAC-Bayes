import unittest
from classes.OptimizationAlgorithm.derived_classes.subclass_ParametricOptimizationAlgorithm import SamplingAssistant


class TestSamplingAssistant(unittest.TestCase):

    def setUp(self):
        self.learning_rate = 1
        self.number_of_iterations_burnin = 100
        self.desired_number_of_samples = 10
        self.sampling_assistant = SamplingAssistant(learning_rate=self.learning_rate,
                                                    desired_number_of_samples=self.desired_number_of_samples,
                                                    number_of_iterations_burnin=self.number_of_iterations_burnin)

    def test_creation(self):
        self.assertIsInstance(self.sampling_assistant, SamplingAssistant)

    def test_decay_learning_rate(self):
        iteration = 10
        self.assertEqual(self.sampling_assistant.current_learning_rate, self.sampling_assistant.initial_learning_rate)
        self.sampling_assistant.decay_learning_rate(iteration=iteration)
        self.assertEqual(self.sampling_assistant.current_learning_rate,
                         self.sampling_assistant.initial_learning_rate / iteration)

    def test_get_progressbar(self):
        pbar = self.sampling_assistant.get_progressbar()
        self.assertTrue(hasattr(pbar, 'desc'))
        self.assertTrue(hasattr(pbar, 'iterable'))
        self.assertEqual(pbar.desc, 'Sampling: ')

    def test_prepare_output(self):
        with self.assertRaises(Exception):
            samples, state_dict_samples, estimated_probabilities = self.sampling_assistant.prepare_output()

        self.sampling_assistant.desired_number_of_samples = 1
        self.sampling_assistant.samples.append(1)
        with self.assertRaises(Exception):
            samples, state_dict_samples, estimated_probabilities = self.sampling_assistant.prepare_output()

        self.sampling_assistant.samples_state_dict.append(1)
        with self.assertRaises(Exception):
            samples, state_dict_samples, estimated_probabilities = self.sampling_assistant.prepare_output()

        self.sampling_assistant.estimated_probabilities.append(1)
        samples, state_dict_samples, estimated_probabilities = self.sampling_assistant.prepare_output()
        self.assertIsInstance(samples, list)
        self.assertIsInstance(state_dict_samples, list)
        self.assertIsInstance(estimated_probabilities, list)
        self.assertEqual(len(samples), len(state_dict_samples))
        self.assertEqual(len(state_dict_samples), len(estimated_probabilities))

