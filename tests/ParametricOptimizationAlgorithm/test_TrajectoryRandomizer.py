import unittest
from unittest import TestCase

from classes.OptimizationAlgorithm.derived_classes.subclass_ParametricOptimizationAlgorithm import TrajectoryRandomizer


class TestTrajectoryRandomizer(unittest.TestCase):

    def setUp(self):
        self.should_restart = False
        self.restart_probability = 0.65
        self.trajectory_randomizer = TrajectoryRandomizer(should_restart=self.should_restart,
                                                          restart_probability=self.restart_probability)

    def test_creation(self):
        self.assertIsInstance(self.trajectory_randomizer, TrajectoryRandomizer)

    def test_get_should_restart(self):
        self.assertFalse(self.trajectory_randomizer.get_should_restart())

    def test_set_should_restart(self):
        with self.assertRaises(TypeError):
            self.trajectory_randomizer.set_should_restart(1)
        with self.assertRaises(TypeError):
            self.trajectory_randomizer.set_should_restart(1.)
        with self.assertRaises(TypeError):
            self.trajectory_randomizer.set_should_restart('1')
        with self.assertRaises(TypeError):
            self.trajectory_randomizer.set_should_restart(lambda x: 1)
        self.trajectory_randomizer.set_should_restart(True)
        self.assertTrue(self.trajectory_randomizer.get_should_restart())

    def test_get_restart_probability(self):
        self.assertEqual(self.restart_probability, self.trajectory_randomizer.get_restart_probability())
