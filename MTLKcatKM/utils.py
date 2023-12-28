import json

import math
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike

import torch
from torch.optim import Optimizer
from typing import Callable, Iterable, Tuple
from torch.distributions.bernoulli import Bernoulli


def exempt_parameters(src_list, ref_list):
    """Remove element from src_list that is in ref_list"""
    res = []
    for x in src_list:
        flag = True
        for y in ref_list:
            if x is y:
                flag = False
                break
        if flag:
            res.append(x)
    return res


def get_device(device=None, local_rank=0):
    if device is None or not torch.cuda.is_available() or device == 'cpu':
        return torch.device('cpu')
    return torch.device(device, local_rank)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {} local rank {}): {}".format(
            args.rank, args.local_rank, "env://"
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method="env://", world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


class TrainNormalizer:
    """Normalize a Tensor and restore it later. """

    def __init__(self, train_path, label_index: List[int], device=None):
        """tensor is taken as a sample to calculate the mean and std"""
        _, ext = os.path.splitext(train_path)
        sep = "," if ext == ".csv" else "\t"
        data = pd.read_csv(train_path, sep=sep).iloc[:, label_index]
        labels = data.values

        self.mean = labels.mean(axis=0, where=~np.isnan(labels)).reshape([1, -1])
        self.std = labels.std(axis=0, where=~np.isnan(labels)).reshape([1, -1]) + 1e-5

    def norm(self, array: ArrayLike):
        assert array.shape[1] == self.mean.shape[1]
        return (array - self.mean) / self.std

    def denorm(self, normed_array: ArrayLike):
        return normed_array * self.std + self.mean

    def state_dict(self):
        return {
            'mean': self.mean,
            'std': self.std
        }

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def show_me_grad(model: torch.nn.Module):
    for name, parms in model.named_parameters():
        # if parms.grad is not None:
        #     # print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),
        #     #       ' -->grad_value:', torch.mean(parms.grad))
        #     continue
        if parms.requires_grad and parms.grad is not None:
            print('-->name:', name)


class WarmCosine:
    def __init__(self, warmup=4e3, tmax=1e5, eta_min=5e-4):
        if warmup is None:
            self.warmup = 0
        else:
            warmup_step = int(warmup)
            assert warmup_step > 0
            self.warmup = warmup_step
            self.lr_step = (1 - eta_min) / warmup_step
        self.tmax = int(tmax)
        self.eta_min = eta_min

    def step(self, step):
        if step >= self.warmup:
            return (
                    self.eta_min
                    + (1 - self.eta_min)
                    * (1 + math.cos(math.pi * (step - self.warmup) / self.tmax))
                    / 2
            )

        else:
            return self.eta_min + self.lr_step * step


class ChildTuningAdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            reserve_p=1.0,
            mode=None
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

        self.gradient_mask = None
        self.reserve_p = reserve_p
        self.mode = mode

    def set_gradient_mask(self, gradient_mask):
        self.gradient_mask = gradient_mask

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # =================== HACK BEGIN =======================
                if self.mode is not None:
                    if self.mode == 'ChildTuning-D':
                        if p in self.gradient_mask:
                            grad *= self.gradient_mask[p]
                    else:
                        # ChildTuning-F
                        grad_mask = Bernoulli(grad.new_full(size=grad.size(), fill_value=self.reserve_p))
                        grad *= grad_mask.sample() / self.reserve_p
                # =================== HACK END =======================

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss


if __name__ == "__main__":
    _train_path = "./data/modeling/train_dataset.csv"
    normalizer = TrainNormalizer(train_path=_train_path, label_index=[-5, -6])
    print("DONE!!!")


class OrganismTokenizer:
    def __init__(self, organism_dictionary):
        self.organism_dictionary = organism_dictionary

        with open(organism_dictionary) as f:
            self.dictionary = json.load(f)

    def tokenize(self, organism):
        return self.dictionary.get(organism, len(self.dictionary))
