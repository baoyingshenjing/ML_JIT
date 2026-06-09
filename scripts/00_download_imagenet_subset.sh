#!/usr/bin/env bash
# =============================================================================
# 00_download_imagenet_subset.sh
# 拉取 ImageNet 子集 (2000张: 2张/类 × 1000类) 用于快速实验
#
# 用法:
#   bash 00_download_imagenet_subset.sh                          # 使用默认路径
#   DATA_PATH=/mnt/data/imagenet bash 00_download_imagenet_subset.sh
#   SAMPLES=5 bash 00_download_imagenet_subset.sh                # 5张/类
# =============================================================================
set -euo pipefail

# ---- Config (可环境变量覆盖) ----
DATA_PATH="${DATA_PATH:-./data/imagenet}"          # 完整 ImageNet 路径
OUTPUT_PATH="${OUTPUT_PATH:-./data/imagenet_2000}" # 子集输出路径
NUM_CLASSES="${NUM_CLASSES:-1000}"                  # 类别数
SAMPLES="${SAMPLES:-2}"                             # 每类取几张
SEED="${SEED:-42}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "  ImageNet Subset Preparation"
echo "========================================"
echo "  DATA_PATH:    $DATA_PATH"
echo "  OUTPUT_PATH:  $OUTPUT_PATH"
echo "  NUM_CLASSES:  $NUM_CLASSES"
echo "  SAMPLES:      $SAMPLES/class"
echo "  Expected:     $((NUM_CLASSES * SAMPLES * 2)) images (train+val)"
echo "========================================"

# ---- Step 1: 检查/安装依赖 ----
pip install -q torch torchvision 2>/dev/null || true

# ---- Step 2: 使用 ImageNette 作为备选 (如果无完整 ImageNet) ----
if [ ! -d "$DATA_PATH/train" ]; then
    echo ""
    echo "[WARN] 完整 ImageNet 未找到: $DATA_PATH"
    echo "[INFO] 方案A: 设置 DATA_PATH 指向你的 ImageNet 目录后重新运行"
    echo "[INFO] 方案B: 自动下载 ImageNette (10类, ~1.5GB) 作为轻量替代"
    echo ""
    echo "选择方案:"
    echo "  1) 我有 ImageNet, 请告诉我怎么设置路径"
    echo "  2) 下载 ImageNette 替代 (输入 'imagenette' 或 2)"
    echo "  3) 从 HuggingFace 下载 ImageNet-1k 子集 (输入 'hf' 或 3)"
    echo ""
    read -rp "选择 [2]: " choice
    choice="${choice:-2}"

    if [ "$choice" = "1" ]; then
        echo "请设置 DATA_PATH 后重新运行:"
        echo "  DATA_PATH=/your/imagenet/path bash $0"
        exit 0
    elif [ "$choice" = "3" ] || [ "$choice" = "hf" ]; then
        echo "[INFO] 从 HuggingFace 下载 imagenet-1k 子集..."
        pip install -q huggingface_hub datasets
        python3 -c "
import os
from datasets import load_dataset
ds = load_dataset('ILSVRC/imagenet-1k', split='train', streaming=True, trust_remote_code=True)
out = '$OUTPUT_PATH/train'
os.makedirs(out, exist_ok=True)
count = 0
target = $NUM_CLASSES * $SAMPLES
seen = {}
for item in ds:
    label = str(item['label'])
    if seen.get(label, 0) >= $SAMPLES:
        continue
    seen[label] = seen.get(label, 0) + 1
    cls_dir = os.path.join(out, f'n{int(label):08d}')
    os.makedirs(cls_dir, exist_ok=True)
    item['image'].save(os.path.join(cls_dir, f'{seen[label]:04d}.JPEG'))
    count += 1
    if count % 100 == 0:
        print(f'  Downloaded {count}/{target}...', flush=True)
    if count >= target:
        break
print(f'Done: {count} images')
# val: same as train for quick subset
import shutil
shutil.copytree(out, '$OUTPUT_PATH/val', dirs_exist_ok=True)
print('val symlinked from train')
"
        # Override paths for later steps
        DATA_PATH="$OUTPUT_PATH"
    else
        # Download ImageNette (10 classes, ~1300 images each)
        echo "[INFO] 下载 ImageNette (320px)..."
        pip install -q gdown
        mkdir -p "$OUTPUT_PATH"
        cd "$OUTPUT_PATH"
        if [ ! -f imagenette2-320.tgz ]; then
            wget -q --show-progress https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
        fi
        tar xzf imagenette2-320.tgz
        # ImageNette has train/ val/ with 10 class dirs
        DATA_PATH="$OUTPUT_PATH/imagenette2-320"
        echo "[INFO] ImageNette 就绪: $DATA_PATH (10类)"
    fi
else
    echo "[INFO] ImageNet 找到: $DATA_PATH"
fi

# ---- Step 3: 创建子集 ----
TOTAL=$((NUM_CLASSES * SAMPLES))
echo ""
echo "[INFO] 从 $DATA_PATH 创建子集: $TOTAL 张/ split"

python3 "$REPO_ROOT/scripts/prepare_subset.py" \
    --data-path "$DATA_PATH" \
    --output-path "$OUTPUT_PATH" \
    --num-classes "$NUM_CLASSES" \
    --samples-per-class "$SAMPLES" \
    --seed "$SEED"

echo ""
echo "========================================"
echo "  Done! Subset at: $OUTPUT_PATH"
echo "  Train: $(find "$OUTPUT_PATH/train" -name '*.JPEG' -o -name '*.jpg' -o -name '*.png' 2>/dev/null | wc -l) images"
echo "  Val:   $(find "$OUTPUT_PATH/val" -name '*.JPEG' -o -name '*.jpg' -o -name '*.png' 2>/dev/null | wc -l) images"
echo "========================================"
echo ""
echo "  下一步:"
echo "    bash 01_train_baseline.sh"
