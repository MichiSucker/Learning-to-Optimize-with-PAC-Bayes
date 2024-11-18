import torch.optim
from tqdm import tqdm


class InitializationAssistant:

    def __init__(self,
                 printing_enabled: bool,
                 maximal_number_of_iterations: int,
                 update_stepsize_every: int,
                 factor_update_stepsize: float,
                 print_update_every: int):
        self.printing_enabled = printing_enabled
        self.maximal_number_of_iterations = maximal_number_of_iterations
        self.update_stepsize_every = update_stepsize_every
        self.print_update_every = print_update_every
        self.factor_update_stepsize = factor_update_stepsize
        self.running_loss = 0

    def print_starting_message(self) -> None:
        if self.printing_enabled:
            print("Initialize algorithm to follow standard algorithm.")
            print(f"Optimizing for {self.maximal_number_of_iterations} iterations.")

    def print_final_message(self) -> None:
        if self.printing_enabled:
            print("Finished initialization.\n\n")

    def get_progressbar(self):
        pbar = tqdm(range(self.maximal_number_of_iterations))
        pbar.set_description('Initialize algorithm')
        return pbar

    def should_update_stepsize_of_optimizer(self, iteration: int) -> bool:
        return (iteration >= 1) and (iteration % self.update_stepsize_every == 0)

    def update_stepsize_of_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        for g in optimizer.param_groups:
            g['lr'] = self.factor_update_stepsize * g['lr']

    def should_print_update(self, iteration: int) -> bool:
        return (iteration >= 1) and self.printing_enabled and (iteration % self.print_update_every == 0)

    def print_update(self, iteration: int) -> None:
        print(f"\nIteration: {iteration}")
        print("\tAvg. Loss = {:.2f}".format(self.running_loss / self.print_update_every))
        self.running_loss = 0
