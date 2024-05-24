import torch
from math import ceil
from itertools import permutations
from torchaudio.transforms import MelScale
from dataclasses import dataclass, field, fields
from typing import List, Type, Any, Callable, Optional, Union
from loguru import logger
from utils.decorators import *


@logger_wraps()
@dataclass(slots=False)
class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
    optimizer: torch.optim.Optimizer
    warmup_steps: int
    last_epoch: int = field(default=-1)
    lr_lambda: callable = field(init=False)

    def __post_init__(self):
        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(self.warmup_steps)
            return 1.0

        self.lr_lambda = lr_lambda
        super().__init__(self.optimizer, self.lr_lambda, last_epoch=self.last_epoch)
        
    def __repr__(self):
        # __init__
        class_name = self.__class__.__name__
        init_fields = [f for f in fields(self) if f.init]
        field_strs = [f"{field.name}={getattr(self, field.name)!r}" for field in init_fields]

        # __post_init__
        lr_lambda_repr = f"lr_lambda = {self.lr_lambda}"
        post_init_reprs = [lr_lambda_repr]

        return f"<{class_name}({', '.join(field_strs + post_init_reprs)})>"