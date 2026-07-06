#!/usr/bin/env bash
# =============================================================================
# 01_train_baseline.sh
# 训练 JiT-B/32 baseline on ImageNet 子集
#
# 用法:
#   bash 01_train_baseline.sh                          # 默认配置
#   MODEL=JiT-B/16 EPOCHS=100 bash 01_train_baseline.sh # 自定义
#   bash 01_train_baseline.sh --help                    # 透传额外参数
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ---- Config (可环境变量覆盖) ----
MODEL="${MODEL:-JiT-B/16}"                    # JiT-B/16, JiT-B/32
IMG_SIZE="${IMG_SIZE:-256}"
DATA_PATH="${DATA_PATH:-./data/imagenet_2000}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/baseline_${MODEL//\//_}}"
BATCH_SIZE="${BATCH_SIZE:-32}"                # 子集小，batch也小
EPOCHS="${EPOCHS:-20}"                        # Fine-tuning epochs
LR="${LR:-}"                                  # 留空 → 自动 blr*bsz/256
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_FREQ="${EVAL_FREQ:-20}"                  # 每20 epoch做一次online eval
EXTRA_ARGS="${EXTRA_ARGS:-}"                  # 透传额外参数
PRETRAINED="${PRETRAINED:-./pretrained_weights/jit_b16_256_pretrained.pt}"

# ---- 检查数据 ----
if [ ! -d "$DATA_PATH/train" ]; then
    echo "[ERROR] 数据未找到: $DATA_PATH"
    echo "  先运行: bash scripts/00_download_imagenet_subset.sh"
    exit 1
fi

# ---- 安装依赖 ----
pip install -q torch torchvision torch-fidelity tensorboard opencv-python 2>/dev/null || true

# ---- 训练 ----
echo "========================================"
echo "  JiT Baseline Training"
echo "========================================"
echo "  MODEL:       $MODEL"
echo "  DATA:        $DATA_PATH"
echo "  OUTPUT:      $OUTPUT_DIR"
echo "  BATCH_SIZE:  $BATCH_SIZE"
echo "  EPOCHS:      $EPOCHS"
echo "  SEED:        $SEED"
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
    ${LR:+--lr "$LR"} \
    ${EXTRA_ARGS}

echo ""
echo "========================================"
echo "  Training complete!"
echo "  Checkpoints: $OUTPUT_DIR/"
echo "  TensorBoard: tensorboard --logdir $OUTPUT_DIR/logs"
echo "========================================"
