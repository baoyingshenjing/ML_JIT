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
        self.scmr_lambda = getattr(args, "scmr_lambda", 0.0)
        self.scmr_stopgrad = getattr(args, "scmr_stopgrad", False)
        self.scmr_warmup_epochs = getattr(args, "scmr_warmup_epochs", 50)
        self.snp_enable = getattr(args, "snp_enable", False)
        self.snp_mask_min = getattr(args, "snp_mask_min", 32)
        self.snp_mask_max = getattr(args, "snp_mask_max", 128)
        self.snp_noise_in = getattr(args, "snp_noise_in", 0.5)
        self.snp_noise_out = getattr(args, "snp_noise_out", 1.0)
        self.tctm_enable = getattr(args, "tctm_enable", False)
        self.tctm_t_threshold = getattr(args, "tctm_t_threshold", 0.3)
        self.tctm_merge_ratio = getattr(args, "tctm_merge_ratio", 0.5)
        self.train_progress = 0.0
        self.num_t_bins = 5

        self.ema = EMA(self.net, args.ema_decay1, args.ema_decay2)

        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        return torch.where(drop, torch.full_like(labels, self.num_classes), labels)

    def set_train_progress(self, epoch_float: float) -> None:
        self.train_progress = epoch_float

    @property
    def current_lambda(self) -> float:
        if self.scmr_lambda <= 0.0:
            return 0.0
        if self.scmr_warmup_epochs <= 0:
            return self.scmr_lambda
        scale = min(max(self.train_progress / self.scmr_warmup_epochs, 0.0), 1.0)
        return self.scmr_lambda * scale

    def _sample_view(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = sample_logit_normal_t(
            x.size(0), self.P_mean, self.P_std, device=x.device
        ).view(-1, *([1] * (x.ndim - 1)))

        if self.training and self.snp_enable:
            e = self._structured_noise(x)
        else:
            e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1.0 - t) * e
        return t, z, e

    def _structured_noise(self, x: torch.Tensor) -> torch.Tensor:
        """SNP: random rectangular regions get different noise scales.

        For each image in the batch, sample 1-3 random rectangles.
        Inside rectangles: snp_noise_in (lower noise, preserve structure).
        Outside rectangles: snp_noise_out (full noise).
        """
        B, C, H, W = x.shape
        e = torch.randn_like(x)
        # Start with full noise everywhere
        noise_map = torch.full((B, 1, H, W), self.snp_noise_out, device=x.device)

        # Random number of rectangles per image (1 to 3)
        num_rects = torch.randint(1, 4, (B,), device=x.device)
        for b in range(B):
            for _ in range(num_rects[b].item()):
                size = torch.randint(
                    self.snp_mask_min, self.snp_mask_max + 1, (2,), device=x.device
                )
                y = torch.randint(0, max(1, H - size[0].item() + 1), (1,), device=x.device).item()
                x0 = torch.randint(0, max(1, W - size[1].item() + 1), (1,), device=x.device).item()
                noise_map[b, 0, y:y + size[0].item(), x0:x0 + size[1].item()] = self.snp_noise_in

        return e * noise_map

    def _add_t_bin_metrics(
        self,
        metrics: dict[str, torch.Tensor],
        t: torch.Tensor,
        flow_loss: torch.Tensor,
        scmr_loss: torch.Tensor,
        x_pred_mse: torch.Tensor,
    ) -> None:
        t_flat = t.flatten().detach()
        edges = torch.linspace(0.0, 1.0, self.num_t_bins + 1, device=t.device)
        for idx in range(self.num_t_bins):
            if idx == self.num_t_bins - 1:
                mask = (t_flat >= edges[idx]) & (t_flat <= edges[idx + 1])
            else:
                mask = (t_flat >= edges[idx]) & (t_flat < edges[idx + 1])
            if not mask.any():
                continue
            prefix = f"t_bin/{idx}_{edges[idx].item():.1f}_{edges[idx + 1].item():.1f}"
            metrics[f"{prefix}/flow_loss"] = flow_loss[mask].mean().detach()
            metrics[f"{prefix}/scmr_loss"] = scmr_loss[mask].mean().detach()
            metrics[f"{prefix}/x_pred_mse"] = x_pred_mse[mask].mean().detach()

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t, z, _e = self._sample_view(x)
        v = (x - z) / (1.0 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), labels_dropped)
        v_pred = (x_pred - z) / (1.0 - t).clamp_min(self.t_eps)

        flow_loss_per_sample = ((v - v_pred) ** 2).mean(dim=(1, 2, 3))
        flow_loss = flow_loss_per_sample.mean()
        x_pred_mse_per_sample = ((x_pred - x) ** 2).mean(dim=(1, 2, 3))
        x_pred_mse = x_pred_mse_per_sample.mean()

        lambda_now = self.current_lambda if self.training else 0.0
        if self.training and self.scmr_lambda > 0.0:
            t2, z2, _e2 = self._sample_view(x)
            x_pred2 = self.net(z2, t2.flatten(), labels_dropped)
            target = x_pred.detach() if self.scmr_stopgrad else x_pred
            scmr_loss_per_sample = ((target - x_pred2) ** 2).mean(dim=(1, 2, 3))
            scmr_loss = scmr_loss_per_sample.mean()
        else:
            scmr_loss_per_sample = torch.zeros_like(flow_loss_per_sample)
            scmr_loss = flow_loss.new_zeros(())

        loss = flow_loss + lambda_now * scmr_loss
        metrics = {
            "flow_loss": flow_loss.detach(),
            "scmr_loss": scmr_loss.detach(),
            "scmr_lambda": flow_loss.new_tensor(lambda_now),
            "x_pred_mse": x_pred_mse.detach(),
        }
        self._add_t_bin_metrics(
            metrics,
            t,
            flow_loss_per_sample,
            scmr_loss_per_sample,
            x_pred_mse_per_sample,
        )

        return loss, metrics

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

            # TCTM: spatial downsampling for high-noise steps
            z_input = z
            if self.tctm_enable and t.mean().item() < self.tctm_t_threshold:
                _, _, h, w = z.shape
                new_h, new_w = max(2, int(h * self.tctm_merge_ratio)), max(2, int(w * self.tctm_merge_ratio))
                z_input = torch.nn.functional.interpolate(
                    z, size=(new_h, new_w), mode="bilinear", align_corners=False
                )
                z_input = torch.nn.functional.interpolate(
                    z_input, size=(h, w), mode="bilinear", align_corners=False
                )

            z = stepper(
                self.net, z_input, t, t_next, labels,
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
