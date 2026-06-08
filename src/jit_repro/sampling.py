"""ODE samplers (Euler, Heun) with classifier-free guidance for JiT."""

from __future__ import annotations

import torch
import torch.nn as nn


def cfg_forward(
    net: nn.Module,
    z: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    cfg_scale: float = 1.5,
    t_eps: float = 1e-5,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Classifier-free guidance velocity prediction.

    v = v_uncond + cfg * (v_cond - v_uncond), with interval masking.
    """
    t_flat = t.flatten()

    x_cond = net(z, t_flat, labels)
    v_cond = (x_cond - z) / (1.0 - t).clamp_min(t_eps)

    x_uncond = net(z, t_flat, torch.full_like(labels, num_classes))
    v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(t_eps)

    low, high = cfg_interval
    interval_mask = (t < high) & ((low == 0) | (t > low))
    cfg_scale_interval = torch.where(interval_mask, cfg_scale, 1.0)

    return v_uncond + cfg_scale_interval * (v_cond - v_uncond)


def euler_step(
    net: nn.Module,
    z: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    cfg_scale: float = 1.5,
    t_eps: float = 1e-5,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Single Euler ODE step."""
    v_pred = cfg_forward(
        net, z, t, labels, num_classes, cfg_scale, t_eps, cfg_interval
    )
    return z + (t_next - t) * v_pred


def heun_step(
    net: nn.Module,
    z: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    cfg_scale: float = 1.5,
    t_eps: float = 1e-5,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Single Heun (improved Euler) ODE step."""
    v_pred_t = cfg_forward(
        net, z, t, labels, num_classes, cfg_scale, t_eps, cfg_interval
    )
    z_next_euler = z + (t_next - t) * v_pred_t

    v_pred_t_next = cfg_forward(
        net, z_next_euler, t_next, labels, num_classes, cfg_scale, t_eps, cfg_interval
    )
    v_pred = 0.5 * (v_pred_t + v_pred_t_next)
    return z + (t_next - t) * v_pred
