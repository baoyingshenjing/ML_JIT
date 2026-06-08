"""Configuration dataclasses and CLI argument parsing for JiT reproduction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from .model import JiT_B_16, JiT_B_32, JiT_L_16, JiT_L_32, JiT_H_16, JiT_H_32


@dataclass
class ModelConfig:
    """Architecture hyperparameters for the JiT model."""

    model: str = "JiT-B/16"
    img_size: int = 256
    patch_size: int = 16
    in_channels: int = 3
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    num_classes: int = 1000
    bottleneck_dim: int = 128
    in_context_len: int = 32
    in_context_start: int = 4


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # diffusion
    label_drop_prob: float = 0.1
    P_mean: float = -0.8
    P_std: float = 0.8
    t_eps: float = 5e-2
    noise_scale: float = 1.0
    scmr_lambda: float = 0.0
    scmr_stopgrad: bool = False
    scmr_warmup_epochs: int = 50

    # optimization
    lr: float | None = None
    blr: float = 5e-5
    min_lr: float = 0.0
    lr_schedule: str = "constant"
    weight_decay: float = 0.0
    batch_size: int = 64
    epochs: int = 600
    warmup_epochs: int = 5

    # EMA
    ema_decay1: float = 0.9999
    ema_decay2: float = 0.9996

    # logging / saving
    log_every: int = 50
    save_every: int = 10
    output_dir: str = "./output"
    data_path: str = "./data"
    resume: str = ""
    seed: int = 0

    # online eval
    gen_bsz: int = 256
    eval_freq: int = 40
    online_eval: bool = False

    # hardware
    device: str = "cuda"
    num_workers: int = 4
    fp16: bool = False
    compile: bool = True


@dataclass
class GenConfig:
    """Image generation hyperparameters."""

    checkpoint: str = ""
    sampling_method: str = "heun"
    num_sampling_steps: int = 50
    cfg: float = 1.0
    interval_min: float = 0.0
    interval_max: float = 1.0
    batch_size: int = 256
    num_images: int = 50000
    output_dir: str = "./generated"
    device: str = "cuda"
    evaluate: bool = False


_MODEL_FACTORY_MAP = {
    "JiT-B/16": JiT_B_16,
    "JiT-B/32": JiT_B_32,
    "JiT-L/16": JiT_L_16,
    "JiT-L/32": JiT_L_32,
    "JiT-H/16": JiT_H_16,
    "JiT-H/32": JiT_H_32,
}


def get_model(name: str):
    """Return the factory function for the given model name."""
    if name not in _MODEL_FACTORY_MAP:
        raise ValueError(
            f"Unknown model {name!r}. Available: {list(_MODEL_FACTORY_MAP.keys())}"
        )
    return _MODEL_FACTORY_MAP[name]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for training or generation."""
    parser = argparse.ArgumentParser(description="JiT Reproduction")

    # model
    parser.add_argument("--model", type=str, default="JiT-B/16")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--class-num", type=int, default=1000)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--proj-dropout", type=float, default=0.0)
    parser.add_argument("--in-context-len", type=int, default=32)
    parser.add_argument("--in-context-start", type=int, default=4)

    # diffusion
    parser.add_argument("--label-drop-prob", type=float, default=0.1)
    parser.add_argument("--P-mean", type=float, default=-0.8)
    parser.add_argument("--P-std", type=float, default=0.8)
    parser.add_argument("--t-eps", type=float, default=5e-2)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument(
        "--scmr-lambda",
        type=float,
        default=0.0,
        help="Target SCMR weight. Recommended sweep: 0.001, 0.003, 0.01, 0.03.",
    )
    parser.add_argument(
        "--scmr-stopgrad",
        action="store_true",
        help="Use stop-gradient SCMR target: ||sg(xhat1) - xhat2||^2.",
    )
    parser.add_argument(
        "--scmr-warmup-epochs",
        type=int,
        default=50,
        help="Linearly ramp SCMR lambda from 0 to target over this many epochs.",
    )

    # optimization
    parser.add_argument("--lr", type=float, default=None, help="Absolute learning rate (if set, overrides blr)")
    parser.add_argument("--blr", type=float, default=5e-5, help="Base learning rate: lr = blr * batch_size / 256")
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--lr-schedule", type=str, default="constant")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--warmup-epochs", type=int, default=5)

    # ema
    parser.add_argument("--ema-decay1", type=float, default=0.9999)
    parser.add_argument("--ema-decay2", type=float, default=0.9996)

    # generation / sampling
    parser.add_argument("--sampling-method", type=str, default="heun")
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--interval-min", type=float, default=0.0)
    parser.add_argument("--interval-max", type=float, default=1.0)
    parser.add_argument("--gen-bsz", type=int, default=256, help="Generation batch size")

    # online evaluation
    parser.add_argument("--eval-freq", type=int, default=40, help="Frequency (epochs) for online evaluation")
    parser.add_argument("--online-eval", action="store_true", help="Enable online FID evaluation during training")

    # i/o
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--num-images", type=int, default=50000)

    # hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-compile", action="store_true")

    # misc
    parser.add_argument("--generate", action="store_true", help="Run in generation mode")
    parser.add_argument("--evaluate", action="store_true", help="Compute FID after generation")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=10)

    return parser.parse_args(argv)
