"""Flow matching diffusion utilities: noise schedule, logit-normal time sampling, and EMA."""

from __future__ import annotations

import copy
from typing import Iterator

import torch
import torch.nn as nn


def sample_logit_normal_t(
    n: int,
    P_mean: float = 0.0,
    P_std: float = 1.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Sample timesteps from the logit-normal distribution.

    Returns scalar t in (0, 1).
    """
    z = torch.randn(n, device=device) * P_std + P_mean
    return torch.sigmoid(z)


class EMA:
    """Exponential moving average for model parameters.

    Maintains two sets of EMA parameters with different decay rates.
    """

    def __init__(
        self,
        model: nn.Module,
        decay1: float = 0.9999,
        decay2: float = 0.9996,
    ):
        self.decay1 = decay1
        self.decay2 = decay2
        self.ema_params1: list[torch.Tensor] = [
            p.detach().clone() for p in model.parameters()
        ]
        self.ema_params2: list[torch.Tensor] = [
            p.detach().clone() for p in model.parameters()
        ]

    def update(self, model: nn.Module) -> None:
        source_params = list(model.parameters())
        if len(self.ema_params1) > 0 and self.ema_params1[0].device != source_params[0].device:
            device = source_params[0].device
            self.ema_params1 = [p.to(device) for p in self.ema_params1]
            self.ema_params2 = [p.to(device) for p in self.ema_params2]
        for targ, src in zip(self.ema_params1, source_params):
            targ.mul_(self.decay1).add_(src.detach(), alpha=1.0 - self.decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.mul_(self.decay2).add_(src.detach(), alpha=1.0 - self.decay2)

    def copy_to(self, model: nn.Module, which: int = 1) -> None:
        params = self.ema_params1 if which == 1 else self.ema_params2
        for targ, src in zip(model.parameters(), params):
            targ.data.copy_(src.data)

    def state_dict(self) -> dict:
        return {
            "decay1": self.decay1,
            "decay2": self.decay2,
            "ema_params1": [p.cpu() for p in self.ema_params1],
            "ema_params2": [p.cpu() for p in self.ema_params2],
        }

    def load_state_dict(self, state_dict: dict, model_device: torch.device | None = None) -> None:
        self.decay1 = state_dict["decay1"]
        self.decay2 = state_dict["decay2"]
        self.ema_params1 = [
            p.to(model_device) for p in state_dict["ema_params1"]
        ]
        self.ema_params2 = [
            p.to(model_device) for p in state_dict["ema_params2"]
        ]
