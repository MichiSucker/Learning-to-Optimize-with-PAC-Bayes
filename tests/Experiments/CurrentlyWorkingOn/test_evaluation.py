import unittest
from experiments.lasso.evaluation import evaluate_algorithm


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.path_to_experiment = '/home/michael/Desktop/JMLR_New/Experiments/lasso/'
        self.dummy_savings_path = self.path_to_experiment + 'dummy_data/'
        self.loading_path = self.path_to_experiment + 'data/'
