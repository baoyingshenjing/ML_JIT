# JiT 复现历程、实验结果与改进算法总结

## 1. 项目背景

本项目围绕 JiT（Just image Transformer）图像生成模型展开复现和改进实验。JiT 的核心思路是在 flow matching 框架下直接预测干净图像 `x0`，并使用一个尽量简洁的 ViT 式图像生成骨干网络完成类条件图像生成。

我们复现的基础配置是 JiT-B/16，输入分辨率为 256 x 256，模型主干包含 bottleneck patch embedding、12 层 Transformer block、RoPE 位置编码、adaLN 条件调制和 in-context class tokens。训练目标使用 logit-normal 采样的时间步 `t`，构造

```text
z_t = t * x + (1 - t) * e
```

模型输出 `x_pred`，再由 `x_pred` 推出 velocity，与真实 flow velocity 计算均方误差。采样阶段使用 Euler 或 Heun ODE solver，并支持 classifier-free guidance。

## 2. 复现历程

### 2.1 代码与工程搭建

我们首先完成了 JiT 复现工程的基本模块：

| 模块 | 文件 | 作用 |
|---|---|---|
| 模型结构 | `src/jit_repro/model.py` | 实现 JiT-B/L/H 系列、patch embedding、RoPE、in-context tokens、Transformer blocks 和 final layer |
| 扩散/flow 工具 | `src/jit_repro/diffusion.py` | 实现 logit-normal 时间采样和 EMA 参数维护 |
| 训练与采样封装 | `src/jit_repro/denoiser.py` | 统一处理 `x0` 预测、flow loss、SNP、SCMR 预留逻辑和采样 |
| ODE 采样器 | `src/jit_repro/sampling.py` | 实现 Euler、Heun 和 classifier-free guidance |
| 训练脚本 | `scripts/train.py` | 单卡 fine-tuning、日志、checkpoint、可选在线 FID |
| 实验脚本 | `scripts/01_train_baseline.sh`、`02_train_snp.sh`、`03_train_tctm.sh` | 分别运行 baseline、SNP fine-tuning 和 TCTM 采样评估 |

### 2.2 数据与训练设置

由于完整 ImageNet-1K 训练成本较高，本轮实验使用轻量子集进行快速验证：

| 项目 | 设置 |
|---|---|
| 数据集 | ImageNet-1K 子集，约 2000 张图像，每类 2 张 |
| 模型 | JiT-B/16，256 x 256 |
| 初始化 | 官方或已转换的 pretrained JiT-B/16 checkpoint |
| fine-tuning | 20 epochs |
| batch size | 32 |
| 学习率 | `blr=5e-5`，实际 `lr=blr * batch_size / 256 = 6.25e-6` |
| 优化器 | AdamW，betas=(0.9, 0.95) |
| 采样 | Heun，50 steps |
| 随机种子 | 42 |

本轮评估的重点是复现链路、收敛行为和方法趋势，因此主要使用训练 loss、`x0` MSE、采样速度、像素多样性 `sigma`、图像对比度和均值稳定性等轻量指标。完整 50k ImageNet FID 尚未作为最终结论指标。

### 2.3 改进方向筛选

项目早期提出并评审了多组改进方向，包括 SCMR、BMP、MATC、结构化噪声、token merging、disagreement 诊断等。经过交叉评审后，优先保留了两条最适合快速验证的方向：

| 方向 | 选择原因 |
|---|---|
| SNP：Structured Noise Process | 训练端改动小，只需要改变噪声注入方式；不依赖外部模型；目标是提高样本多样性和结构鲁棒性 |
| TCTM：Time-Conditioned Sampling Acceleration | 采样端改动小，不需要重新训练；目标是提高生成速度，并观察速度-质量折中 |

SCMR 已在代码中预留实现，但由于其与原监督目标存在较强重叠，本轮没有把它作为主要结果展开。BMP 和 MATC 被降级为后续消融或长期方向。

## 3. Baseline 复现结果

baseline fine-tuning 能够稳定收敛。训练曲线显示，20 epoch 后：

| 指标 | Baseline FT |
|---|---:|
| final flow loss | 0.0518 |
| final `x0` MSE | 0.0247 |
| 采样时间 | 2.03 s |
| TCTM sweep 中的像素多样性 `sigma` | 56.4 |
| 采样均值 | 103.9 |
| SNP 对比实验中的像素多样性 `sigma` | 58.3 |
| SNP 对比实验中的图像对比度 | 251.2 |

对应图表见：

```text
docs/plots/02_training_curves.png
docs/plots/05_dashboard.png
```

从曲线看，baseline loss 在训练中存在小幅波动，但总体稳定，没有出现发散。由于数据子集很小，训练 loss 和 proxy 指标只能说明复现链路有效，不能直接替代完整 ImageNet FID。

## 4. 改进算法一：SNP 结构化噪声过程

### 4.1 方法动机

标准 flow matching 使用全图均匀强度的高斯噪声。我们希望让模型在训练时看到更复杂的局部退化模式：同一张图中部分区域保留更多结构，其他区域仍使用完整噪声。这样可以迫使模型在不均匀噪声条件下恢复图像，从而提高结构鲁棒性和生成多样性。

### 4.2 具体实现

SNP 在训练阶段替换普通噪声 `e` 的生成方式。对 batch 中每张图像：

1. 随机采样 1 到 3 个矩形区域。
2. 矩形边长在 32 到 128 像素之间。
3. 矩形区域内使用较低噪声强度 `sigma_in=0.5`。
4. 区域外使用正常噪声强度 `sigma_out=1.0`。
5. 仍然按原公式构造 `z_t = t * x + (1 - t) * e_structured`。

核心代码位于 `src/jit_repro/denoiser.py` 的 `_structured_noise()`。该方法不改变模型结构，只改变训练时的 forward noise process。

### 4.3 SNP 实验结果

| 指标 | Baseline FT | SNP FT | 变化 |
|---|---:|---:|---:|
| final flow loss | 0.0518 | 0.0481 | 更低 |
| final `x0` MSE | 0.0247 | 0.0259 | 接近 |
| 像素多样性 `sigma` | 58.3 | 63.9 | +9.6% |
| 图像对比度 | 251.2 | 253.4 | +0.9% |

SNP 的主要收益体现在生成样本统计上：像素多样性从 58.3 提升到 63.9，增幅约 9.6%；图像对比度也有轻微提升。`x0` MSE 与 baseline 基本接近，说明结构化噪声没有明显破坏干净图像预测能力。

需要注意的是，SNP 并不是免费加速方法，它需要额外 fine-tuning；它更适合用于追求样本多样性、数据增强或创意生成的场景。

对应图表见：

```text
docs/plots/04_snp_bars.png
docs/plots/02_training_curves.png
```

## 5. 改进算法二：TCTM 时间条件采样加速

### 5.1 方法动机

JiT 的高质量采样通常使用 50 步 Heun solver。早期时间步处于高噪声阶段，图像细节尚未形成，score 或 velocity 的变化相对粗粒度；后期时间步接近干净图像，需要更细的步长恢复细节。因此可以在高噪声阶段使用更大的时间步，以减少采样次数。

### 5.2 当前实现说明

文档和脚本中沿用了 TCTM（Time-Conditioned Token Merging）这一名称，但当前仓库中的实现本质上是时间条件的 ODE step skipping，而不是真正的 ToMe-style token merging。

实现逻辑如下：

```text
for each sampling step i:
    if TCTM enabled and t < threshold:
        skip = int(1 / merge_ratio)
        next_i = min(i + skip, last_step)
        用 Heun 从 t_i 直接走到 t_next_i
    else:
        正常走到 t_{i+1}

最后一步始终使用 Euler step 到 t=1
```

该方法只影响采样阶段，不改变训练过程，也不改变模型参数。

### 5.3 TCTM 速度-质量结果

| 配置 | 时间 | 加速比 | 像素多样性 `sigma` | 均值 |
|---|---:|---:|---:|---:|
| Baseline，无 TCTM | 2.03 s | 1.00x | 56.4 | 103.9 |
| threshold=0.3，skip=2 | 1.68 s | 1.21x | 51.1 | 103.5 |
| threshold=0.5，skip=2 | 1.74 s | 1.17x | 51.1 | 103.5 |
| threshold=0.5，skip=3 | 1.19 s | 1.70x | 47.2 | 103.4 |
| threshold=0.7，skip=3 | 1.05 s | 1.93x | 47.2 | 103.4 |
| threshold=0.7，skip=4 | 1.05 s | 1.94x | 44.5 | 103.5 |

推荐配置是 `threshold=0.7, skip=3`，它在本轮实验中达到 1.93x 采样加速，同时均值基本稳定，质量代价主要表现为多样性指标从 56.4 降到 47.2。

更激进的 sweep 显示，当 threshold 提高到 0.8 且 skip=4 时，加速比可以进一步接近 2.5x，但多样性下降更明显。因此它更适合作为低延迟场景的可选配置，而不是默认推荐配置。

对应图表见：

```text
docs/plots/01_tctm_pareto.png
docs/plots/03_tctm_sweep.png
```

## 6. 综合结论

| 方法 | 主要目标 | 结果 | 代价 | 建议用途 |
|---|---|---|---|---|
| Baseline FT | 复现 JiT fine-tuning 链路 | 稳定收敛，final loss 0.0518 | 无额外改动 | 作为所有实验基准 |
| SNP | 提升生成多样性 | `sigma` 从 58.3 到 63.9，提升 9.6% | 需要额外 fine-tuning | 多样性、数据增强、创意生成 |
| TCTM step skipping | 提升采样速度 | 推荐配置 1.93x 加速 | 多样性下降约 16% | 推理加速、资源受限部署 |

本轮实验说明：SNP 和 TCTM 改善的是两个不同维度。SNP 是训练端正则化，主要提高多样性；TCTM 是采样端加速策略，主要降低推理时间。二者理论上可以叠加，但当前实验尚未完成 SNP checkpoint 上的系统性 TCTM sweep。

## 7. 局限性

1. 数据规模有限：当前使用的是 ImageNet 子集，不是完整 ImageNet-1K。
2. 指标仍是轻量 proxy：本报告没有把完整 50k FID 作为最终结论。
3. TCTM 名称需要谨慎：当前实现是 step skipping，不是真正 token merging。
4. 随机性尚未充分评估：当前结果主要来自固定 seed，缺少多 seed 置信区间。
5. 组合实验未完成：SNP + TCTM 的联合收益和 trade-off 仍需系统验证。

## 8. 复现实验命令

```bash
# 1. 准备 ImageNet 子集
bash scripts/00_download_imagenet_subset.sh

# 2. 训练 baseline
bash scripts/01_train_baseline.sh

# 3. 训练 SNP
bash scripts/02_train_snp.sh

# 4. 评估 TCTM 采样加速
bash scripts/03_train_tctm.sh
```

常用可调环境变量：

```bash
MODEL=JiT-B/16
EPOCHS=20
BATCH_SIZE=32
SNP_NOISE_IN=0.5
SNP_NOISE_OUT=1.0
TCTM_T_THRESHOLD=0.7
TCTM_MERGE_RATIO=0.33
```

## 9. 后续工作

下一步建议按以下顺序推进：

1. 在完整或更大规模 ImageNet 子集上计算 FID、IS、precision/recall。
2. 对 SNP 做超参消融：矩形数量、mask size、`sigma_in`、`sigma_out`。
3. 在 SNP checkpoint 上重新跑 TCTM sweep，确认二者是否互补。
4. 将当前 TCTM step skipping 升级为真正的 token merging + restore，再比较两者速度和质量。
5. 对所有关键结果做至少 3 个 seed 的重复实验。
6. 若继续研究 SCMR，需要加入 compute-matched baseline，避免把额外 forward 的收益误判为一致性正则的收益。
