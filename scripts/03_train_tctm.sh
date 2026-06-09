#!/usr/bin/env bash
# =============================================================================
# 03_train_tctm.sh
# TCTM (Time-Conditioned Token Merging) — 采样时加速
#
# TCTM 是 pure sampling-time 方法，不需要额外训练。
# 本脚本:
#   1. 训练 baseline (如果 checkpoint 不存在)
#   2. 用 TCTM 做 evaluation (高噪声步合并 token 加速)
#
# 用法:
#   bash 03_train_tctm.sh                           # 默认配置
#   CHECKPOINT=./output/baseline_JiT-B_32/checkpoint-0099.pt bash 03_train_tctm.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ---- Config ----
MODEL="${MODEL:-JiT-B/32}"
IMG_SIZE="${IMG_SIZE:-256}"
DATA_PATH="${DATA_PATH:-./data/imagenet_2000}"
BASELINE_DIR="${BASELINE_DIR:-./output/baseline_${MODEL//\//_}}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/tctm_${MODEL//\//_}}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_FREQ="${EVAL_FREQ:-20}"

# TCTM 超参
TCTM_T_THRESHOLD="${TCTM_T_THRESHOLD:-0.3}"     # t < 此值时启用合并
TCTM_MERGE_RATIO="${TCTM_MERGE_RATIO:-0.5}"     # 合并比例 (0.5 = 降采样到50%再回)

# ---- 检查数据 ----
if [ ! -d "$DATA_PATH/train" ]; then
    echo "[ERROR] 数据未找到: $DATA_PATH"
    exit 1
fi

pip install -q torch torchvision torch-fidelity tensorboard opencv-python 2>/dev/null || true

echo "========================================"
echo "  JiT + TCTM (Time-Conditioned Token Merging)"
echo "========================================"
echo "  MODEL:          $MODEL"
echo "  TCTM threshold: t < $TCTM_T_THRESHOLD"
echo "  TCTM merge:     ${TCTM_MERGE_RATIO}x"
echo "  (TCTM 只影响采样，训练与 baseline 相同)"
echo "========================================"

cd "$REPO_ROOT"

# ---- Step 1: Train baseline (if no checkpoint) ----
LATEST_CKPT=$(ls -t "$BASELINE_DIR"/checkpoint-*.pt 2>/dev/null | head -1 || echo "")

if [ -z "$LATEST_CKPT" ]; then
    echo "[INFO] 无 baseline checkpoint，先训练 baseline..."
    mkdir -p "$BASELINE_DIR"

    python3 scripts/train.py \
        --model "$MODEL" \
        --img-size "$IMG_SIZE" \
        --data-path "$DATA_PATH" \
        --output-dir "$BASELINE_DIR" \
        --batch-size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --seed "$SEED" \
        --num-workers "$NUM_WORKERS" \
        --eval-freq "$EVAL_FREQ" \
        --online-eval \
        --class-num "$(ls -d "$DATA_PATH"/train/*/ 2>/dev/null | wc -l)" \
        ${EXTRA_ARGS:-}

    LATEST_CKPT=$(ls -t "$BASELINE_DIR"/checkpoint-*.pt 2>/dev/null | head -1)
fi

echo "[INFO] Using checkpoint: $LATEST_CKPT"

# ---- Step 2: 用 TCTM 做 generation + FID 评估 ----
echo ""
echo "========================================"
echo "  Running TCTM evaluation"
echo "========================================"

mkdir -p "$OUTPUT_DIR"

python3 -c "
import torch
import sys
sys.path.insert(0, '$REPO_ROOT/src')
from jit_repro.denoiser import Denoiser
import argparse

# Load checkpoint
ckpt = torch.load('$LATEST_CKPT', map_location='cuda')
args = argparse.Namespace(
    model='$MODEL', img_size=$IMG_SIZE, class_num=1000,
    attn_dropout=0.0, proj_dropout=0.0,
    label_drop_prob=0.1, P_mean=-0.8, P_std=0.8, t_eps=5e-2, noise_scale=1.0,
    scmr_lambda=0.0, scmr_stopgrad=False, scmr_warmup_epochs=50,
    snp_enable=False,
    tctm_enable=True, tctm_t_threshold=$TCTM_T_THRESHOLD, tctm_merge_ratio=$TCTM_MERGE_RATIO,
    ema_decay1=0.9999, ema_decay2=0.9996,
    sampling_method='heun', num_sampling_steps=50, cfg=1.0,
    interval_min=0.0, interval_max=1.0,
)

model = Denoiser(args).cuda()
# Load EMA params
ema_state = {}
for (name, _), param in zip(model.named_parameters(), ckpt['model_ema1'].values()):
    ema_state[name] = param.cuda()
model.load_state_dict(ema_state)
model.eval()

print(f'Generating with TCTM (t<{model.tctm_t_threshold}, merge={model.tctm_merge_ratio})...')

import os, numpy as np, cv2
num_images = 500
class_num = args.class_num
labels = torch.arange(class_num).repeat(num_images // class_num + 1)[:num_images].cuda()

save_dir = '$OUTPUT_DIR/generated_tctm'
os.makedirs(save_dir, exist_ok=True)

with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    for i in range(0, num_images, 64):
        batch_labels = labels[i:i+64]
        gen = model.generate(batch_labels)
        gen = (gen + 1) / 2
        gen = gen.clamp(0, 1).cpu()
        for j in range(gen.size(0)):
            img = (gen[j].permute(1,2,0).numpy() * 255).astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_dir, f'{i+j:05d}.png'), img)
        if i % 128 == 0:
            print(f'  Generated {i+len(batch_labels)}/{num_images}')

print(f'Generated {num_images} images to {save_dir}')
print('Done! (FID evaluation requires torch-fidelity and reference stats)')
" 2>&1

echo ""
echo "========================================"
echo "  TCTM evaluation complete!"
echo "  Generated images: $OUTPUT_DIR/generated_tctm/"
echo ""
echo "  Compare with baseline sampling:"
echo "    Baseline + TCTM: $OUTPUT_DIR/"
echo "========================================"
