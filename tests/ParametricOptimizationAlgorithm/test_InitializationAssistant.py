import unittest
import sys
import io
from classes.OptimizationAlgorithm.derived_classes.subclass_ParametricOptimizationAlgorithm import (
    InitializationAssistant)


class TestInitializationAssistant(unittest.TestCase):

    def setUp(self):
        self.printing_enabled = True
        self.maximal_number_of_iterations = 100
        self.initialization_assistant = InitializationAssistant(
            printing_enabled=self.printing_enabled,
            maximal_number_of_iterations=self.maximal_number_of_iterations
        )

    def test_creation(self):
        self.assertIsInstance(self.initialization_assistant, InitializationAssistant)

    def test_starting_message(self):
        # This is just a weak test: We only test whether it created an output.
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        self.initialization_assistant.starting_message()
        self.assertTrue(len(capturedOutput.getvalue()) > 0)
        sys.stdout = sys.__stdout__

        self.initialization_assistant.printing_enabled = False
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        self.initialization_assistant.starting_message()
        self.assertTrue(len(capturedOutput.getvalue()) == 0)
        sys.stdout = sys.__stdout__

    def test_final_message(self):
        # This is just a weak test: We only test whether it created an output.
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        self.initialization_assistant.final_message()
        self.assertTrue(len(capturedOutput.getvalue()) > 0)
        sys.stdout = sys.__stdout__

        self.initialization_assistant.printing_enabled = False
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        self.initialization_assistant.final_message()
        self.assertTrue(len(capturedOutput.getvalue()) == 0)
        sys.stdout = sys.__stdout__

    def test_get_progressbar(self):
        pbar = self.initialization_assistant.get_progressbar()
        self.assertTrue(hasattr(pbar, 'desc'))
        self.assertTrue(hasattr(pbar, 'iterable'))
        self.assertEqual(pbar.desc, 'Initialize algorithm: ')
        self.assertEqual(list(pbar.iterable), list(range(self.initialization_assistant.maximal_number_of_iterations)))

