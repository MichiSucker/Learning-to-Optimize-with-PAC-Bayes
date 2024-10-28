import numpy as np
from tqdm import tqdm
import torch
from classes.Helpers.class_ConstraintChecker import ConstraintChecker


class TrainingAssistant:

    def __init__(self,
                 printing_enabled: bool,
                 print_update_every: int,
                 maximal_number_of_iterations: int,
                 update_stepsize_every: int,
                 factor_update_stepsize: float,
                 bins: list = None):
        self.printing_enabled = printing_enabled
        self.print_update_every = print_update_every
        self.maximal_number_of_iterations = maximal_number_of_iterations
        self.update_stepsize_every = update_stepsize_every
        self.factor_update_stepsize = factor_update_stepsize
        self.running_loss = 0
        self.loss_histogram = []
        if not bins:
            self.bins = [1e0, 1e-4, 1e-8, 1e-12, 1e-16, 1e-20, 1e-24, 1e-28][::-1]
        else:
            self.bins = bins

    def print_starting_message(self):
        if self.printing_enabled:
            print("---------------------------------------------------------------------------------------------------")
            print("Fit Algorithm:")
            print("---------------------------------------------------------------------------------------------------")
            print(f"\t-Optimizing for {self.maximal_number_of_iterations} iterations.")
            print(f"\t-Updating step-size every {self.update_stepsize_every} iterations.")

    def print_final_message(self):
        if self.printing_enabled:
            print("---------------------------------------------------------------------------------------------------")
            print("End Fitting Algorithm.")

    def get_progressbar(self):
        pbar = tqdm(range(self.maximal_number_of_iterations))
        pbar.set_description('Fit algorithm')
        return pbar

    def should_update_stepsize_of_optimizer(self, iteration: int) -> bool:
        if (iteration >= 1) and (iteration % self.update_stepsize_every == 0):
            return True
        else:
            return False

    def update_stepsize_of_optimizer(self, optimizer: torch.optim.Optimizer):
        for g in optimizer.param_groups:
            g['lr'] = self.factor_update_stepsize * g['lr']

    def get_variable__bins(self) -> list:
        return self.bins

    def set_variable__bins__to(self, new_bins: list):
        self.bins = new_bins

    def should_print_update(self, iteration: int) -> bool:
        if iteration >= 1 and self.printing_enabled and iteration % self.print_update_every == 0:
            return True
        else:
            return False

    def print_update(self, iteration: int, constraint_checker: ConstraintChecker):
        print("\t\t\t-----------------------------------------------------------------------------------------")
        print(f"\t\t\tIteration: {iteration}; Found point inside constraint: "
              f"{constraint_checker.found_point_inside_constraint}")
        vals, bins = np.histogram(self.loss_histogram, bins=self.bins)
        print(f"\t\t\t\tLosses:")
        for j in range(len(vals) - 1, -1, -1):
            print(f"\t\t\t\t\t[{bins[j + 1]:.0e}, {bins[j]:.0e}] : {vals[j]}/{self.print_update_every}")
        print("\t\t\t-----------------------------------------------------------------------------------------")

    def reset_running_loss_and_loss_histogram(self):
        self.loss_histogram = []
        self.running_loss = 0