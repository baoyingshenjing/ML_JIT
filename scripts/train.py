#!/usr/bin/env python3
"""Single-GPU training loop for JiT reproduction."""

from __future__ import annotations

import copy
import math
import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from jit_repro.config import parse_args
from jit_repro.denoiser import Denoiser

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def center_crop_arr(pil_image, image_size):
    """Center crop a PIL image to a square, matching reference behavior."""
    W, H = pil_image.size
    s = min(W, H)
    left = (W - s) // 2
    top = (H - s) // 2
    pil_image = pil_image.crop((left, top, left + s, top + s))
    return pil_image.resize((image_size, image_size))


def build_dataloader(
    data_path: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    train: bool = True,
) -> DataLoader:
    """Build dataloader with reference-style transforms (no Normalize)."""
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor(),
    ])
    dataset = datasets.ImageFolder(data_path, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def adjust_learning_rate(optimizer, epoch, args):
    """Constant learning rate schedule with linear warmup."""
    warmup_epochs = args.warmup_epochs
    if epoch < warmup_epochs:
        lr = args.lr * (epoch + 1) / warmup_epochs
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def online_evaluate(
    denoiser: Denoiser,
    args,
    epoch: int,
    writer: SummaryWriter | None = None,
):
    """Generate images and compute FID using torch-fidelity."""
    import shutil

    import cv2
    import torch_fidelity

    model_without_ddp = denoiser
    model_without_ddp.eval()

    # Construct save folder name matching reference pattern
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1],
            args.num_images, args.img_size,
        ),
    )
    os.makedirs(save_folder, exist_ok=True)
    logger.info("Online eval: saving to %s", save_folder)

    # Switch to EMA params (ema_decay1)
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        ema_state_dict[name] = model_without_ddp.ema.ema_params1[i]
    logger.info("Switched to EMA for evaluation")
    model_without_ddp.load_state_dict(ema_state_dict)

    class_num = args.class_num
    assert args.num_images % class_num == 0, "num_images must be divisible by class_num"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    # Pad extra labels to ensure we have enough (reference pads to 50000)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    batch_size = args.gen_bsz
    num_steps = args.num_images // batch_size + 1

    for i in range(num_steps):
        start_idx = batch_size * i
        end_idx = min(start_idx + batch_size, args.num_images)
        labels_gen = class_label_gen_world[start_idx:end_idx]
        if len(labels_gen) == 0:
            break
        labels_gen = torch.tensor(labels_gen, dtype=torch.long).cuda()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen)

        # Denormalize: [-1, 1] -> [0, 1] -> [0, 255]
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        for b_id in range(sampled_images.size(0)):
            img_id = i * batch_size + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(
                sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255
            ))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(
                os.path.join(save_folder, "{}.png".format(str(img_id).zfill(5))),
                gen_img,
            )

        if i % 10 == 0:
            logger.info("Online eval generation step %d/%d", i, num_steps)

    # Switch back from EMA
    logger.info("Switching back from EMA")
    model_without_ddp.load_state_dict(model_state_dict)

    # Compute FID
    if writer is not None:
        if args.img_size == 256:
            fid_statistics_file = "fid_stats/jit_in256_stats.npz"
        elif args.img_size == 512:
            fid_statistics_file = "fid_stats/jit_in512_stats.npz"
        else:
            raise NotImplementedError(f"Unsupported img_size: {args.img_size}")

        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict["frechet_inception_distance"]
        inception_score = metrics_dict["inception_score_mean"]
        postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
        writer.add_scalar("fid{}".format(postfix), fid, epoch)
        writer.add_scalar("is{}".format(postfix), inception_score, epoch)
        logger.info(
            "Epoch %d: FID=%.4f, IS=%.4f", epoch, fid, inception_score,
        )

        # Clean up generated images
        shutil.rmtree(save_folder)

    model_without_ddp.train()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    logger.info("Building model: %s", args.model)
    denoiser = Denoiser(args).to(device)
    if args.compile and not args.no_compile:
        logger.info("Enabling torch.compile on network")
        denoiser.net = torch.compile(denoiser.net)

    logger.info("Building dataloader")
    train_loader = build_dataloader(
        args.data_path, args.img_size, args.batch_size, args.num_workers, train=True
    )

    # Compute LR from blr if lr is not explicitly set
    if args.lr is None:
        args.lr = args.blr * args.batch_size / 256
    logger.info(
        "Base lr: %.2e, Actual lr: %.2e, batch_size: %d",
        args.blr, args.lr, args.batch_size,
    )

    optimizer = torch.optim.AdamW(
        denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    start_epoch = 0
    global_step = 0

    if args.resume:
        logger.info("Resuming from %s", args.resume)
        checkpoint = torch.load(args.resume, map_location=device)
        denoiser.load_state_dict(checkpoint["model"])
        denoiser.ema.load_state_dict(checkpoint["ema"], model_device=device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        logger.info("Resumed from epoch %d", start_epoch)

    logger.info("Training for %d epochs", args.epochs)
    for epoch in range(start_epoch, args.epochs):
        denoiser.train()
        epoch_loss = 0.0
        t_start = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            epoch_progress = epoch + batch_idx / len(train_loader)

            # Warmup LR scheduler (per-iteration)
            lr = adjust_learning_rate(
                optimizer, epoch_progress, args,
            )

            # Image normalization: div 255 then scale to [-1, 1]
            images = images.to(device, non_blocking=True).to(torch.float32).div_(255)
            images = images * 2.0 - 1.0
            labels = labels.to(device, non_blocking=True)

            # bf16 autocast (no grad scaler needed for bf16)
            denoiser.set_train_progress(epoch_progress)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = denoiser(images, labels)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                logger.error("Loss is %f, stopping training", loss_value)
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            denoiser.update_ema()

            epoch_loss += loss_value

            if global_step % args.log_every == 0:
                writer.add_scalar("train/loss", loss_value, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                for name, value in metrics.items():
                    writer.add_scalar(f"train/{name}", value.item(), global_step)
                logger.info(
                    "epoch=%d step=%d loss=%.6f flow=%.6f scmr=%.6f lambda=%.5f x_mse=%.6f lr=%.2e",
                    epoch, global_step, loss_value,
                    metrics["flow_loss"].item(),
                    metrics["scmr_loss"].item(),
                    metrics["scmr_lambda"].item(),
                    metrics["x_pred_mse"].item(),
                    lr,
                )

            global_step += 1

        epoch_loss /= len(train_loader)
        elapsed = time.time() - t_start
        logger.info(
            "epoch=%d avg_loss=%.6f time=%.1fs",
            epoch, epoch_loss, elapsed,
        )
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch)

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            save_path = os.path.join(args.output_dir, f"checkpoint-{epoch:04d}.pt")
            torch.save(
                {
                    "model": denoiser.state_dict(),
                    "model_ema1": {
                        name: param.cpu()
                        for (name, _), param in zip(
                            denoiser.named_parameters(), denoiser.ema.ema_params1,
                        )
                    },
                    "model_ema2": {
                        name: param.cpu()
                        for (name, _), param in zip(
                            denoiser.named_parameters(), denoiser.ema.ema_params2,
                        )
                    },
                    "ema": denoiser.ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                },
                save_path,
            )
            logger.info("Saved checkpoint to %s", save_path)

        # Online evaluation with FID
        if args.online_eval and (epoch % args.eval_freq == 0 or (epoch + 1) == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                online_evaluate(denoiser, args, epoch, writer)
            torch.cuda.empty_cache()

        writer.flush()

    writer.close()
    logger.info("Training complete")


if __name__ == "__main__":
    main()
