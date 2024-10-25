import unittest
import coverage

TESTING_LEVEL = 'SKIP_EXPENSIVE_TESTS'

if __name__ == '__main__':

    cov = coverage.Coverage(omit=['*test_*'])
    cov.start()

    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir='tests')
    runner = unittest.runner.TextTestRunner().run(test_suite)

    cov.stop()
    cov.save()
    cov.report()
