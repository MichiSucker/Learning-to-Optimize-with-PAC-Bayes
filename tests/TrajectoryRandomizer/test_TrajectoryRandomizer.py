import unittest
from classes.Helpers.class_TrajectoryRandomizer import TrajectoryRandomizer


class TestTrajectoryRandomizer(unittest.TestCase):

    def setUp(self):
        self.should_restart = False
        self.restart_probability = 0.65
        self.length_partial_trajectory = 1
        self.trajectory_randomizer = TrajectoryRandomizer(should_restart=self.should_restart,
                                                          restart_probability=self.restart_probability,
                                                          length_partial_trajectory=self.length_partial_trajectory)

    def test_creation(self):
        self.assertIsInstance(self.trajectory_randomizer, TrajectoryRandomizer)

    def test_get_variable__should_restart(self):
        self.assertFalse(self.trajectory_randomizer.get_variable__should_restart())

    def test_set_variable__should_restart(self):
        # Check that it only can be set to boolean.
        with self.assertRaises(TypeError):
            self.trajectory_randomizer.set_variable__should_restart__to(1)
        with self.assertRaises(TypeError):
            self.trajectory_randomizer.set_variable__should_restart__to(1.)
        with self.assertRaises(TypeError):
            self.trajectory_randomizer.set_variable__should_restart__to('1')
        with self.assertRaises(TypeError):
            self.trajectory_randomizer.set_variable__should_restart__to(lambda x: 1)
        self.trajectory_randomizer.set_variable__should_restart__to(True)
        self.assertTrue(self.trajectory_randomizer.get_variable__should_restart())

    def test_get_restart_probability(self):
        self.assertEqual(self.restart_probability, self.trajectory_randomizer.get_variable__restart_probability())
