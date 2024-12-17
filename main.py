import unittest
import coverage

import experiments.quadratics.run_experiment
import experiments.image_processing.run_experiment
import experiments.lasso.run_experiment
import experiments.nn_training.run_experiment
import experiments.mnist.run_experiment


def run_tests():

    cov = coverage.Coverage(omit=['*test_*', 'main.py', '__init__.py'])
    cov.start()

    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir='tests/Experiments/Quadratics')
    runner = unittest.runner.TextTestRunner().run(test_suite)

    cov.stop()
    cov.save()
    cov.html_report()


def run_experiments(path_to_experiment):
    experiments.quadratics.run_experiment.run(path_to_experiment)
    # experiments.image_processing.run_experiment.run(path_to_experiment)
    # experiments.lasso.run_experiment.run(path_to_experiment)
    # experiments.nn_training.run_experiment.run(path_to_experiment)
    # experiments.mnist.run_experiment.run(path_to_experiment)


if __name__ == '__main__':

    path = ... # Need to be specified first.
    run_experiments(path_to_experiment=path)
