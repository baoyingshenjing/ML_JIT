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
MODEL="${MODEL:-JiT-B/16}"
IMG_SIZE="${IMG_SIZE:-256}"
DATA_PATH="${DATA_PATH:-./data/imagenet_2000}"
BASELINE_DIR="${BASELINE_DIR:-./output/baseline_${MODEL//\//_}}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/tctm_${MODEL//\//_}}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_FREQ="${EVAL_FREQ:-20}"
PRETRAINED="${PRETRAINED:-./pretrained_weights/jit_b16_256_pretrained.pt}"

# TCTM 超参
TCTM_T_THRESHOLD="${TCTM_T_THRESHOLD:-0.7}"     # t < 此值时跳过步骤
TCTM_MERGE_RATIO="${TCTM_MERGE_RATIO:-0.33}"     # 步长倍数 (0.33 = 3x步长)

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

# ---- Step 1: Use pretrained checkpoint directly ----
CKPT="$PRETRAINED"
echo "[INFO] Using pretrained checkpoint: $CKPT"

# ---- Step 2: 用 TCTM 做 generation + FID 评估 ----
echo ""
echo "========================================"
echo "  Running TCTM evaluation"
echo "========================================"

mkdir -p "$OUTPUT_DIR"

python3 << PYEOF
import torch, os, numpy as np, cv2, argparse, sys
sys.path.insert(0, 'src')
from jit_repro.denoiser import Denoiser

ckpt = torch.load('$CKPT', map_location='cuda')
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
# Load using checkpoint's 'model' key (our format)
model.load_state_dict(ckpt['model'])
model.eval()

print(f'Generating with TCTM (t<{model.tctm_t_threshold}, merge={model.tctm_merge_ratio})...')

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

# Also generate without TCTM for comparison
args2 = argparse.Namespace(
    model='$MODEL', img_size=$IMG_SIZE, class_num=1000,
    attn_dropout=0.0, proj_dropout=0.0,
    label_drop_prob=0.1, P_mean=-0.8, P_std=0.8, t_eps=5e-2, noise_scale=1.0,
    scmr_lambda=0.0, scmr_stopgrad=False, scmr_warmup_epochs=50,
    snp_enable=False,
    tctm_enable=False, tctm_t_threshold=$TCTM_T_THRESHOLD, tctm_merge_ratio=$TCTM_MERGE_RATIO,
    ema_decay1=0.9999, ema_decay2=0.9996,
    sampling_method='heun', num_sampling_steps=50, cfg=1.0,
    interval_min=0.0, interval_max=1.0,
)
model_no_tctm = Denoiser(args2).cuda()
model_no_tctm.load_state_dict(ckpt['model'])
model_no_tctm.eval()

save_dir_no = '$OUTPUT_DIR/generated_no_tctm'
os.makedirs(save_dir_no, exist_ok=True)
print(f'Generating without TCTM for comparison...')

torch.manual_seed(42)
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    for i in range(0, min(100, num_images), 64):
        batch_labels = labels[i:i+64]
        gen = model_no_tctm.generate(batch_labels)
        gen = (gen + 1) / 2
        gen = gen.clamp(0, 1).cpu()
        for j in range(gen.size(0)):
            img = (gen[j].permute(1,2,0).numpy() * 255).astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_dir_no, f'{i+j:05d}.png'), img)

print(f'Generated baseline images to {save_dir_no}')
print('Done!')
PYEOF

echo ""
echo "========================================"
echo "  TCTM evaluation complete!"
echo "  Generated images: $OUTPUT_DIR/generated_tctm/"
echo ""
echo "  Compare with baseline sampling:"
echo "    Baseline + TCTM: $OUTPUT_DIR/"
echo "========================================"
