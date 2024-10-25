import copy
from tqdm import tqdm


class SamplingAssistant:

    def __init__(self, learning_rate, desired_number_of_samples, number_of_iterations_burnin):
        self.initial_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.number_of_correct_samples = 0
        self.desired_number_of_samples = desired_number_of_samples
        self.number_of_iterations_burnin = number_of_iterations_burnin
        self.point_that_satisfies_constraint = None
        self.noise_distributions = None
        self.samples = []
        self.samples_state_dict = []
        self.estimated_probabilities = []
        self.progressbar = tqdm(total=self.desired_number_of_samples + self.number_of_iterations_burnin)
        self.progressbar.set_description('Sampling')

    def decay_learning_rate(self, iteration):
        self.current_learning_rate = self.initial_learning_rate / iteration

    def set_point_that_satisfies_constraint(self, state_dict):
        if not isinstance(state_dict, dict):
            raise TypeError("Provided point is not a dict (in particular not a state_dict()).")
        self.point_that_satisfies_constraint = copy.deepcopy(state_dict)

    def set_noise_distributions(self, noise_distributions):
        if not isinstance(noise_distributions, dict):
            raise TypeError("Provided input is not a dict.")
        self.noise_distributions = noise_distributions

    def should_continue(self):
        return self.number_of_correct_samples < self.desired_number_of_samples + self.number_of_iterations_burnin

    def should_store_sample(self, iteration):
        return iteration >= self.number_of_iterations_burnin

    def reject_sample(self, optimization_algorithm):
        optimization_algorithm.implementation.load_state_dict(self.point_that_satisfies_constraint)

    def store_sample(self, implementation, estimated_probability):
        self.samples.append([p.detach().clone() for p in implementation.parameters() if p.requires_grad])
        self.samples_state_dict.append(copy.deepcopy(implementation.state_dict()))
        self.estimated_probabilities.append(estimated_probability)
        self.number_of_correct_samples += 1
        self.progressbar.update(1)

    def prepare_output(self):
        if any((len(self.samples) < self.desired_number_of_samples,
                len(self.samples_state_dict) < self.desired_number_of_samples,
                len(self.estimated_probabilities) < self.desired_number_of_samples)):
            raise Exception("Did not find enough samples to prepare output.")

        if ((len(self.samples) != len(self.samples_state_dict))
                or (len(self.samples) != len(self.estimated_probabilities))):
            raise Exception("Something went wrong: Number of samples do not match up.")

        return (self.samples[-self.desired_number_of_samples:],
                self.samples_state_dict[-self.desired_number_of_samples:],
                self.estimated_probabilities[-self.desired_number_of_samples:])
