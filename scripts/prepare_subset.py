#!/usr/bin/env python3
"""Create an ImageNet-N subset (a balanced subset of ImageNet).

Expects the standard ImageNet directory layout:
    data_path/
        train/n01440764/*.JPEG
        train/n01443537/*.JPEG
        ...
        val/n01440764/*.JPEG
        val/n01443537/*.JPEG
        ...

Creates a subset under output_path/ with the same structure.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def create_subset(
    data_path: str,
    output_path: str,
    num_classes: int = 1000,
    samples_per_class: int | None = None,
    seed: int = 42,
) -> None:
    random.seed(seed)

    src_train = Path(data_path) / "train"
    src_val = Path(data_path) / "val"

    dst_train = Path(output_path) / "train"
    dst_val = Path(output_path) / "val"

    if not src_train.exists():
        raise FileNotFoundError(f"Train directory not found: {src_train}")

    # Gather all class directories from train
    class_dirs = sorted(
        [p for p in src_train.iterdir() if p.is_dir()]
    )[:num_classes]
    logger.info("Found %d class directories", len(class_dirs))

    for split, src_dir, dst_dir in [
        ("train", src_train, dst_train),
        ("val", src_val, dst_val),
    ]:
        dst_dir.mkdir(parents=True, exist_ok=True)
        count = 0

        for class_dir in class_dirs:
            class_name = class_dir.name
            src_class = src_dir / class_name
            if not src_class.exists():
                logger.warning("Skipping %s (not found in %s)", class_name, split)
                continue

            images = sorted([p for p in src_class.iterdir() if p.suffix.lower() in (".jpeg", ".jpg", ".png")])
            if samples_per_class is not None and len(images) > samples_per_class:
                images = random.sample(images, samples_per_class)

            dst_class = dst_dir / class_name
            dst_class.mkdir(parents=True, exist_ok=True)

            for img_path in images:
                shutil.copy2(img_path, dst_class / img_path.name)
                count += 1

        logger.info("%s: copied %d images", split, count)

    logger.info("Subset created at %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ImageNet-N subset")
    parser.add_argument("--data-path", type=str, required=True, help="Path to full ImageNet dataset")
    parser.add_argument("--output-path", type=str, required=True, help="Output path for the subset")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes")
    parser.add_argument("--samples-per-class", type=int, default=None, help="Max samples per class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_subset(
        data_path=args.data_path,
        output_path=args.output_path,
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
