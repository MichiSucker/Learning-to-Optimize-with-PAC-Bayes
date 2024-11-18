import unittest
import torch
import io
import sys
from classes.Helpers.class_TrainingAssistant import TrainingAssistant
from classes.Helpers.class_ConstraintChecker import ConstraintChecker


class TestTrainingAssistant(unittest.TestCase):

    def setUp(self):
        self.printing_enabled = True
        self.print_update_every = 10
        self.maximal_number_of_iterations = 100
        self.factor_update_stepsize = 0.5
        self.update_stepsize_every = 10
        self.training_assistant = TrainingAssistant(
            printing_enabled=self.printing_enabled,
            print_update_every=self.print_update_every,
            maximal_number_of_iterations=self.maximal_number_of_iterations,
            update_stepsize_every=self.update_stepsize_every,
            factor_update_stepsize=self.factor_update_stepsize)

    def test_creation(self):
        self.assertIsInstance(self.training_assistant, TrainingAssistant)

    def test_starting_message(self):
        # This is just a weak test: We only test whether it created an output.
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        self.training_assistant.print_starting_message()
        self.assertTrue(len(capturedOutput.getvalue()) > 0)
        sys.stdout = sys.__stdout__

        # Check that there was an output.
        self.training_assistant.printing_enabled = False
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        self.training_assistant.print_starting_message()
        self.assertTrue(len(capturedOutput.getvalue()) == 0)
        sys.stdout = sys.__stdout__

    def test_final_message(self):
        # This is just a weak test: We only test whether it created an output.
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        self.training_assistant.print_final_message()
        self.assertTrue(len(capturedOutput.getvalue()) > 0)
        sys.stdout = sys.__stdout__

        # Check that there was an output.
        self.training_assistant.printing_enabled = False
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        self.training_assistant.print_final_message()
        self.assertTrue(len(capturedOutput.getvalue()) == 0)
        sys.stdout = sys.__stdout__

    def test_get_progressbar(self):
        pbar = self.training_assistant.get_progressbar()
        self.assertTrue(hasattr(pbar, 'desc'))
        self.assertTrue(hasattr(pbar, 'iterable'))
        self.assertEqual(pbar.desc, 'Fit algorithm: ')
        self.assertEqual(list(pbar.iterable), list(range(self.training_assistant.maximal_number_of_iterations)))

    def test_should_update_stepsize_of_optimizer(self):
        # Step-size should be updated if random_multiple >= 0 and random_multiple % update_stepsize_every == 0.
        random_multiple = torch.randint(1, 10, size=(1,)).item() * self.update_stepsize_every
        self.assertTrue(self.training_assistant.should_update_stepsize_of_optimizer(iteration=random_multiple))
        self.assertFalse(self.training_assistant.should_update_stepsize_of_optimizer(iteration=random_multiple-1))
        self.assertFalse(self.training_assistant.should_update_stepsize_of_optimizer(iteration=0))

    def test_update_stepsize_of_optimizer(self):
        dummy_parameters = [torch.tensor([1., 2.], requires_grad=True)]
        lr = 4e-3
        optimizer = torch.optim.Adam(dummy_parameters, lr=lr)
        self.training_assistant.update_stepsize_of_optimizer(optimizer=optimizer)
        for g in optimizer.param_groups:
            self.assertEqual(g['lr'], self.training_assistant.factor_update_stepsize * lr)

    def test_get_bins(self):
        bins = self.training_assistant.get_variable__bins()
        self.assertIsInstance(bins, list)
        self.assertEqual(bins, sorted(bins))

    def test_set_bins(self):
        new_bins = [1, 2, 3]
        old_bins = self.training_assistant.get_variable__bins()
        self.assertNotEqual(old_bins, new_bins)
        self.training_assistant.set_variable__bins__to(new_bins=new_bins)
        self.assertEqual(self.training_assistant.get_variable__bins(), new_bins)

    def test_should_print_update(self):
        random_multiple = torch.randint(1, 9, size=(1,)).item() * self.print_update_every
        self.assertTrue(self.training_assistant.should_print_update(random_multiple))
        self.assertFalse(self.training_assistant.should_print_update(random_multiple-1))
        self.assertFalse(self.training_assistant.should_print_update(0))
        self.training_assistant.printing_enabled = False
        self.assertFalse(self.training_assistant.should_print_update(random_multiple))

    def test_print_update(self):
        # This is just a weak test: We only test whether it created an output.
        constraint_checker = ConstraintChecker(check_constraint_every=1, there_is_a_constraint=True)
        constraint_checker.found_point_inside_constraint = True
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        self.training_assistant.print_update(iteration=10, constraint_checker=constraint_checker)
        self.assertTrue(len(capturedOutput.getvalue()) > 0)
        sys.stdout = sys.__stdout__
        self.assertTrue(len(capturedOutput.getvalue()) > 0)

    def test_reset_running_loss_and_loss_histogram(self):
        self.training_assistant.running_loss = 100
        self.training_assistant.loss_histogram = [1, 2, 3]
        self.training_assistant.reset_running_loss_and_loss_histogram()
        self.assertEqual(self.training_assistant.running_loss, 0)
        self.assertEqual(self.training_assistant.loss_histogram, [])
