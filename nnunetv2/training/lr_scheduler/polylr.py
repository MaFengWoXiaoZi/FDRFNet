from torch.optim.lr_scheduler import _LRScheduler
import torch
from math import cos, pi


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

class CosineAnnealingLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, min_lr: float = 0, current_step: int = None, periodical_epochs: int = 50):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.ctr = 0
        self.periodical_epochs = periodical_epochs
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        # 余弦退火学习率计算
        cos_factor = 0.5 * (1 + torch.cos(torch.tensor(current_step / self.periodical_epochs * torch.pi)))
        new_lr = self.min_lr + (self.initial_lr - self.min_lr) * cos_factor
        
        # 更新每个参数组的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr.item()

class CosineAnnealingWithWarmupLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float = 0.01, max_steps: int = 300, min_lr: float = 0, current_step: int = None, warmup_step: int = 30):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.ctr = 0
        self.warmup_step = warmup_step
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        if current_step < self.warmup_step:
            new_lr = self.initial_lr * current_step / self.warmup_step
        else:
            new_lr = self.min_lr + (self.initial_lr - self.min_lr) * (1 + cos(pi * (current_step - self.warmup_step) / (self.max_steps - self.warmup_step))) / 2

        # 更新每个参数组的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
