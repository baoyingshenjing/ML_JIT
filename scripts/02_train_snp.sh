#!/usr/bin/env bash
# =============================================================================
# 02_train_snp.sh
# 训练 JiT-B/32 + Structured Noise Process (SNP)
#
# SNP 原理: 对每张图随机选1-3个矩形区域，
#   区域内用弱噪声 (snp_noise_in), 区域外用强噪声 (snp_noise_out)
#   让模型学会在不同噪声强度下保持结构一致性
#
# 用法:
#   bash 02_train_snp.sh                            # 默认 SNP 超参
#   SNP_NOISE_IN=0.3 SNP_NOISE_OUT=0.8 bash 02_train_snp.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ---- Config ----
MODEL="${MODEL:-JiT-B/16}"
IMG_SIZE="${IMG_SIZE:-256}"
DATA_PATH="${DATA_PATH:-./data/imagenet_2000}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/snp_${MODEL//\//_}}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_FREQ="${EVAL_FREQ:-20}"
PRETRAINED="${PRETRAINED:-./pretrained_weights/jit_b16_256_pretrained.pt}"

# SNP 超参
SNP_MASK_MIN="${SNP_MASK_MIN:-32}"       # 矩形最小边长 (px)
SNP_MASK_MAX="${SNP_MASK_MAX:-128}"      # 矩形最大边长 (px)
SNP_NOISE_IN="${SNP_NOISE_IN:-0.5}"      # 区域内噪声强度 (越低越保留结构)
SNP_NOISE_OUT="${SNP_NOISE_OUT:-1.0}"    # 区域外噪声强度 (正常噪声)

# ---- 检查数据 ----
if [ ! -d "$DATA_PATH/train" ]; then
    echo "[ERROR] 数据未找到: $DATA_PATH"
    exit 1
fi

pip install -q torch torchvision torch-fidelity tensorboard opencv-python 2>/dev/null || true

echo "========================================"
echo "  JiT + SNP (Structured Noise Process)"
echo "========================================"
echo "  MODEL:       $MODEL"
echo "  DATA:        $DATA_PATH"
echo "  OUTPUT:      $OUTPUT_DIR"
echo "  SNP mask:    ${SNP_MASK_MIN}-${SNP_MASK_MAX}px"
echo "  SNP noise:   in=$SNP_NOISE_IN  out=$SNP_NOISE_OUT"
echo "  BATCH_SIZE:  $BATCH_SIZE"
echo "  EPOCHS:      $EPOCHS"
echo "========================================"

cd "$REPO_ROOT"

python3 scripts/train.py \
    --model "$MODEL" \
    --img-size "$IMG_SIZE" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --num-workers "$NUM_WORKERS" \
    --eval-freq "$EVAL_FREQ" \
    ${ONLINE_EVAL:+--online-eval} \
    --pretrained "$PRETRAINED" \
    --class-num "$(ls -d "$DATA_PATH"/train/*/ 2>/dev/null | wc -l)" \
    --snp-enable \
    --snp-mask-min "$SNP_MASK_MIN" \
    --snp-mask-max "$SNP_MASK_MAX" \
    --snp-noise-in "$SNP_NOISE_IN" \
    --snp-noise-out "$SNP_NOISE_OUT" \
    ${EXTRA_ARGS:-}

echo ""
echo "========================================"
echo "  SNP Training complete!"
echo "  Compare with baseline:"
echo "    Baseline: $PWD/output/baseline_${MODEL//\//_}/"
echo "    SNP:      $OUTPUT_DIR/"
echo "========================================"
