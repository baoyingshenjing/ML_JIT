#!/usr/bin/env python3
"""Image generation from a JiT checkpoint."""

from __future__ import annotations

import os
import shutil
import logging

import numpy as np
import torch
import torch_fidelity

from jit_repro.config import parse_args
from jit_repro.denoiser import Denoiser

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    args = parse_args()

    if not args.checkpoint:
        raise ValueError("--checkpoint is required for generation")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info("Loading model from %s", args.checkpoint)
    denoiser = Denoiser(args).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "model" in checkpoint:
        denoiser.load_state_dict(checkpoint["model"])

        # Load EMA (use decay1 by default, matching reference paper)
        if "model_ema1" in checkpoint:
            logger.info("Using EMA parameters (decay1)")
            for i, (name, _) in enumerate(denoiser.named_parameters()):
                denoiser.ema.ema_params1[i] = checkpoint["model_ema1"][name].to(device)
            # Copy ema_params1 into the model for generation
            for targ, src in zip(denoiser.parameters(), denoiser.ema.ema_params1):
                targ.data.copy_(src.data)
        elif "ema" in checkpoint:
            logger.info("Using EMA parameters (legacy format)")
            denoiser.ema.load_state_dict(checkpoint["ema"], model_device=device)
            denoiser.ema.copy_to(denoiser.net, which=1)
    else:
        denoiser.load_state_dict(checkpoint)

    denoiser.eval()
    denoiser.method = args.sampling_method
    denoiser.steps = args.num_sampling_steps
    denoiser.cfg_scale = args.cfg
    denoiser.cfg_interval = (args.interval_min, args.interval_max)

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(
        "Generating %d images with method=%s steps=%d cfg=%.1f",
        args.num_images, args.sampling_method,
        args.num_sampling_steps, args.cfg,
    )

    # Use class-balanced labels (same as reference eval)
    class_num = args.class_num
    assert args.num_images % class_num == 0, "num_images must be divisible by class_num"
    class_label_gen = np.arange(0, class_num).repeat(args.num_images // class_num)

    num_generated = 0
    bsz = args.gen_bsz if args.gen_bsz else args.batch_size

    while num_generated < args.num_images:
        batch_end = min(num_generated + bsz, args.num_images)
        actual_bsz = batch_end - num_generated
        labels = torch.tensor(
            class_label_gen[num_generated:num_generated + actual_bsz],
            dtype=torch.long,
        ).to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                images = denoiser.generate(labels)

        images = (images + 1) / 2.0
        images = images.clamp(0, 1).detach().cpu()

        for j in range(actual_bsz):
            img_id = num_generated + j
            gen_img = np.round(np.clip(
                images[j].numpy().transpose([1, 2, 0]) * 255, 0, 255
            )).astype(np.uint8)[:, :, ::-1]
            save_path = os.path.join(args.output_dir, "{}.png".format(str(img_id).zfill(5)))
            import cv2
            cv2.imwrite(save_path, gen_img)

        num_generated += actual_bsz

        if num_generated % 5000 == 0:
            logger.info("Generated %d/%d images", num_generated, args.num_images)

    logger.info("Generation complete — %d images saved to %s", args.num_images, args.output_dir)

    # FID evaluation
    if args.evaluate:
        logger.info("Computing FID (evaluate mode)")
        if args.img_size == 256:
            fid_statistics_file = "fid_stats/jit_in256_stats.npz"
        elif args.img_size == 512:
            fid_statistics_file = "fid_stats/jit_in512_stats.npz"
        else:
            raise NotImplementedError(f"Unsupported img_size: {args.img_size}")

        metrics_dict = torch_fidelity.calculate_metrics(
            input1=args.output_dir,
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
        logger.info("FID: %.4f, Inception Score: %.4f", fid, inception_score)


if __name__ == "__main__":
    main()
