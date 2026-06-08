"""Denoiser wrapper: x0-prediction training, ODE sampling, EMA."""

from __future__ import annotations

import torch
import torch.nn as nn

from .diffusion import EMA, sample_logit_normal_t
from .sampling import euler_step, heun_step
from .model import JiT_models


class Denoiser(nn.Module):
    """Denoiser wrapping the JiT backbone with flow matching and sampling."""

    def __init__(self, args):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        self.ema = EMA(self.net, args.ema_decay1, args.ema_decay2)

        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        return torch.where(drop, torch.full_like(labels, self.num_classes), labels)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = sample_logit_normal_t(
            x.size(0), self.P_mean, self.P_std, device=x.device
        ).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1.0 - t) * e
        v = (x - z) / (1.0 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), labels_dropped)
        v_pred = (x_pred - z) / (1.0 - t).clamp_min(self.t_eps)

        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    @torch.no_grad()
    def generate(self, labels: torch.Tensor) -> torch.Tensor:
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(
            bsz, 3, self.img_size, self.img_size, device=device
        )
        timesteps = (
            torch.linspace(0.0, 1.0, self.steps + 1, device=device)
            .view(-1, *([1] * z.ndim))
            .expand(-1, bsz, -1, -1, -1)
        )

        if self.method == "euler":
            stepper = euler_step
        elif self.method == "heun":
            stepper = heun_step
        else:
            raise NotImplementedError(f"Unknown sampling method: {self.method}")

        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(
                self.net, z, t, t_next, labels,
                num_classes=self.num_classes,
                cfg_scale=self.cfg_scale,
                t_eps=self.t_eps,
                cfg_interval=self.cfg_interval,
            )

        # Last step always Euler
        z = euler_step(
            self.net, z, timesteps[-2], timesteps[-1], labels,
            num_classes=self.num_classes,
            cfg_scale=self.cfg_scale,
            t_eps=self.t_eps,
            cfg_interval=self.cfg_interval,
        )
        return z

    def update_ema(self) -> None:
        self.ema.update(self.net)
