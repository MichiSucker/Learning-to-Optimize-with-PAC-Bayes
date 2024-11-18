import torch
import torch.nn as nn
from algorithms.fista import soft_thresholding
from typing import Tuple
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm


def split_zero_nonzero(v: torch.Tensor,
                       zeros: torch.BoolTensor,
                       non_zeros: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor]:
    v_zeros, v_non_zeros = torch.zeros(len(v)), torch.zeros(len(v))
    v_zeros[zeros] = v[zeros]
    v_non_zeros[non_zeros] = v[non_zeros]
    return v_zeros, v_non_zeros


class SparsityNet(nn.Module):

    def __init__(self, dim: int, smoothness: torch.Tensor):
        super(SparsityNet, self).__init__()

        self.smoothness: torch.Tensor = smoothness
        self.eps: torch.Tensor = torch.tensor(1e-20).float()  # For stability
        self.dim: int = dim

        size_out: int = 2
        size_in: int = 8
        h_size: int = 20
        self.update = nn.Sequential(
            nn.Conv2d(size_in, h_size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(h_size, h_size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(h_size, h_size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(h_size, size_out, kernel_size=1, bias=False),
        )

        size_out: int = 8
        size_in: int = 12
        h_size: int = 10
        self.step_size = nn.Sequential(
            nn.Linear(size_in, 3 * h_size, bias=False),
            nn.ReLU(),
            nn.Linear(3 * h_size, 2 * h_size, bias=False),
            nn.ReLU(),
            nn.Linear(2 * h_size, h_size, bias=False),
            nn.ReLU(),
            nn.Linear(h_size, size_out, bias=False),
        )

    def forward(self, opt_algo: OptimizationAlgorithm) -> torch.Tensor:

        # Extract regularization parameter
        reg_param = opt_algo.loss_function.smooth_part.get_parameter()['mu']

        # Compute loss
        loss = opt_algo.loss_function(opt_algo.current_state[1]).reshape((1,))
        old_loss = opt_algo.loss_function(opt_algo.current_state[0]).reshape((1,))
        smooth_loss = opt_algo.loss_function.smooth_part(opt_algo.current_state[1]).reshape((1,))
        old_smooth_loss = opt_algo.loss_function.smooth_part(opt_algo.current_state[0]).reshape((1,))
        nonsmooth_loss = opt_algo.loss_function.nonsmooth_part(opt_algo.current_state[1]).reshape((1,))
        old_nonsmooth_loss = opt_algo.loss_function.nonsmooth_part(opt_algo.current_state[0]).reshape((1,))

        # Compute sparsity
        zeros = opt_algo.current_state[1].eq(0)
        non_zeros = opt_algo.current_state[1].ne(0)

        # Extrapolate
        diff = opt_algo.current_state[1] - opt_algo.current_state[0]
        diff_norm = torch.linalg.norm(diff)

        if diff_norm > self.eps:
            diff = diff / diff_norm

        diff_zero, diff_nonzero = split_zero_nonzero(diff, zeros=zeros, non_zeros=non_zeros)
        # Note that this is (decisively) NOT the same as standard normalization, which would break the link
        diff_zero_norm = diff_norm * torch.linalg.norm(diff_zero)
        diff_nonzero_norm = diff_norm * torch.linalg.norm(diff_nonzero)

        y_k = opt_algo.current_state[1]

        # Compute gradient
        grad = opt_algo.loss_function.compute_gradient_of_smooth_part(y_k)
        grad_norm = torch.linalg.norm(grad)

        if grad_norm > self.eps:
            grad = grad / grad_norm

        grad_zero, grad_nonzero = split_zero_nonzero(grad, zeros=zeros, non_zeros=non_zeros)
        # Note that this is (decisively) NOT the same as standard normalization, which would break the link
        grad_zero_norm = grad_norm * torch.linalg.norm(grad_zero)
        grad_nonzero_norm = grad_norm * torch.linalg.norm(grad_nonzero)

        # Compute stopping criteria
        # Note that grad is normalized!
        prox = soft_thresholding(y_k - grad_norm * grad / self.smoothness, tau=reg_param / self.smoothness)
        stopping_criterion_vec = (opt_algo.current_state[1] - prox)
        stopping_crit = torch.linalg.norm(stopping_criterion_vec)

        if stopping_crit > self.eps:
            stopping_criterion_vec = stopping_criterion_vec / stopping_crit

        stopping_zero, stopping_nonzero = split_zero_nonzero(v=stopping_criterion_vec, zeros=zeros, non_zeros=non_zeros)
        # Note that this is (decisively) NOT the same as standard normalization, which would break the link
        stopping_zero_norm = stopping_crit * torch.linalg.norm(stopping_zero)
        stopping_nonzero_norm = stopping_crit * torch.linalg.norm(stopping_nonzero)

        alpha = self.step_size(torch.concat((
            reg_param.reshape((1,)),    # This seems to be very important. This changes between problem instances.
            torch.log(1. + old_loss) - torch.log(1. + loss),  # Seems to be helpful.
            torch.log(1. + old_smooth_loss) - torch.log(1. + smooth_loss),
            torch.log(1. + old_nonsmooth_loss) - torch.log(1. + nonsmooth_loss),  # Maybe remove these?!
            torch.log(1. + grad_zero_norm.reshape((1,))),
            torch.log(1. + grad_nonzero_norm.reshape((1,))),
            torch.log(1. + diff_zero_norm.reshape((1,))),
            torch.log(1. + diff_nonzero_norm.reshape((1,))),
            torch.log(1. + stopping_zero_norm.reshape((1,))),
            torch.log(1. + stopping_nonzero_norm.reshape((1,))),
            torch.dot(grad_zero, diff_zero).reshape((1,)),
            torch.dot(grad_nonzero, diff_nonzero).reshape((1,)),
        )))

        direction = (self.update(torch.concat((
            alpha[0] * grad_zero.reshape((1, -1, 1, self.dim)),
            alpha[1] * grad_nonzero.reshape((1, -1, 1, self.dim)),
            alpha[2] * stopping_zero.reshape((1, -1, 1, self.dim)),
            alpha[3] * stopping_nonzero.reshape((1, -1, 1, self.dim)),
            alpha[4] * diff_zero.reshape((1, -1, 1, self.dim)),
            alpha[5] * diff_nonzero.reshape((1, -1, 1, self.dim)),
            alpha[6] * (grad_nonzero * diff_nonzero).reshape((1, -1, 1, self.dim)),
            alpha[7] * (grad_zero * diff_zero).reshape((1, -1, 1, self.dim)),
        ), dim=1))).reshape((2, self.dim))

        dir_1 = direction[0].flatten() * zeros.to(torch.double)
        dir_2 = direction[1].flatten() * non_zeros.to(torch.double)

        update = soft_thresholding(y_k + (dir_1 - grad_norm * grad + diff_norm * dir_2)/self.smoothness,
                                   tau=reg_param/self.smoothness)

        return update

    @staticmethod
    def update_state(opt_algo: OptimizationAlgorithm) -> None:
        opt_algo.current_state[0] = opt_algo.current_state[1].detach().clone()
        opt_algo.current_state[1] = opt_algo.current_iterate.detach().clone()
