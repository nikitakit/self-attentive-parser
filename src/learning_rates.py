from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmupThenReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, optimizer, warmup_steps, *args, **kwargs):
        """
        Args:
            optimizer (Optimizer): Optimizer to wrap
            warmup_steps: number of steps before reaching base learning rate
            *args: Arguments for ReduceLROnPlateau
            **kwargs: Arguments for ReduceLROnPlateau
        """
        super().__init__(optimizer, *args, **kwargs)
        self.warmup_steps = warmup_steps
        self.steps_taken = 0
        self.base_lrs = list(map(lambda group: group["lr"], optimizer.param_groups))
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

    def get_lr(self):
        assert self.steps_taken <= self.warmup_steps
        return [
            base_lr * (self.steps_taken / self.warmup_steps)
            for base_lr in self.base_lrs
        ]

    def step(self, metrics=None):
        self.steps_taken += 1
        if self.steps_taken <= self.warmup_steps:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group["lr"] = lr
        elif metrics is not None:
            super().step(metrics)
