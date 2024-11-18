
class TrajectoryRandomizer:

    def __init__(self,
                 should_restart: bool,
                 restart_probability: float,
                 length_partial_trajectory: int):
        self.should_restart = should_restart
        self.restart_probability = restart_probability
        self.length_partial_trajectory = length_partial_trajectory

    def get_variable__should_restart(self) -> bool:
        return self.should_restart

    def set_variable__should_restart__to(self, should_restart: bool) -> None:
        if not isinstance(should_restart, bool):
            raise TypeError("Type of 'should_restart' has to be bool.")
        self.should_restart = should_restart

    def get_variable__restart_probability(self) -> float:
        return self.restart_probability
