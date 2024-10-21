import unittest
import coverage


if __name__ == '__main__':

    cov = coverage.Coverage()
    cov.start()

    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir='tests')
    runner = unittest.runner.TextTestRunner().run(test_suite)

    cov.stop()
    cov.save()
    cov.report()
