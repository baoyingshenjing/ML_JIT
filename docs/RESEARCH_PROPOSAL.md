# JiT 改进方案研究提案

## 背景

JiT (Li & He, 2025) "Back to Basics: Let Denoising Generative Models Denoise"
arxiv:2511.13720

核心主张：在流匹配扩散框架中，直接预测干净图像(x-prediction)优于预测噪声(ε-prediction)
或速度(v-prediction)。原因：干净数据位于低维流形上，而噪声/速度不在。
这个"流形假设"允许**欠容量网络**在高维像素空间有效工作。

JiT 仅用简单的 patch-wise ViT (无tokenizer、无预训练、无额外loss)就在ImageNet达到有竞争力FID。

关键架构参数：
- JiT-B/16: depth=12, hidden=768, heads=12, bottleneck=128, patch=16
- 训练: logit-normal t采样 (μ=-0.8, σ=0.8), 600 epochs, batch 1024, AdamW, bf16

---

## 三个改进方向

### 方案A: Self-Consistency Manifold Regularization (SCMR) ★首选

**核心思想**:
JiT 存在一个隐含矛盾——它调用流形假设论证 x-prediction 的优势,
但**从未在训练中显式约束模型服从流形假设**。
如果干净数据确实在低维流形上,那么从同一张干净图像 x 出发,
用不同噪声路径 (t1,ε1) 和 (t2,ε2) 生成的两个噪声版本 z_t1 和 z_t2,
它们各自预测的干净图像 x̂_t1 和 x̂_t2 应该收敛到流形上的**同一点**。

**具体实现**:
```python
# 标准 forward: z_t1 = t1*x + (1-t1)*ε1
# 额外 SCMR forward: z_t2 = t2*x + (1-t2)*ε2  (新采样 t2, ε2)
# x̂1 = f(z_t1, t1), x̂2 = f(z_t2, t2)
# L_total = L_flow + λ * ||x̂1 - x̂2||²
```

这样做的好处：
- 显式强制"所有去噪路径收敛到流形上同一点"
- t1 和 t2 不同，迫使模型学习一致的流形表示
- 计算开销仅增加一次前向传播（可共享时间嵌入计算）
- 可用很小的 λ (0.01-0.1) 作为辅助正则项

**创新性判断**:
- 与标准 consistency models (Song et al.) 不同：SCMR 在**同一干净图像的不同噪声版本之间**施加一致性，而非跨时间去噪步
- 与 BYOL/SimSiam 等自监督方法有精神相似性：同一数据的不同视角应产生一致表示
- 在扩散模型中未见类似做法（据我们所知）

**预期效果**:
- 降低 FID（尤其在少步采样时，因为模型有更一致的流形估计）
- 对 JiT-B/16 可能将 FID 从 ~8.6 降至 7.x
- 典型 λ ∈ [0.01, 0.5]

### 方案B: Bottleneck Manifold Projection (BMP)

**核心思想**:
JiT 已使用 bottleneck patch embedding (3×16×16→128→768),
论文 Fig.4 显示 bottleneck 有帮助。但 bottleneck 仅在**输入端**。
我们提出在 Transformer 的**中间层**也引入流形投影瓶颈，
显式逐层压缩和重建流形表示。

**具体实现**:
在每 k 层之后插入一个 bottleneck projector:
`x → Linear(hidden→bottleneck) → SiLU → Linear(bottleneck→hidden) + x`
这类似于在 Transformer 内部创建一系列"流形检查点"。

**创新性判断**:
- 与标准 residual MLP 不同：bottleneck 迫使中间表示压缩到低维
- 理论动机：如果数据在低维流形上，中间表示也应该可以压缩
- 类似于 autoencoder 的 bottleneck，但嵌入在 Transformer 内部

**预期效果**:
- 可能提升训练效率（中间表示的维度更低）
- 对 JiT-B 来说 FID 改善可能有限（模型本身已小）
- 对 JiT-H 更可能有显著改善（大模型容易过拟合）

### 方案C: Manifold-Aware Timestep Curriculum (MATC)

**核心思想**:
JiT 使用固定的 logit-normal 分布采样 t。但 x-prediction 难点在低 t 区域
（高噪声，此时流形几乎不可见）。我们提出从"易到难"的课程学习策略：
- 训练早期：t 偏向 1（低噪声，流形可见）
- 训练后期：均匀采样（包括困难的高噪声区）
- 同时用 SCMR loss 监控流形一致性作为课程进展的信号

**具体实现**:
t 分布的 μ 从正值逐渐退火到 -0.8:
- Epoch 0-100: μ = 0.5 → 0.0
- Epoch 100-300: μ = 0.0 → -0.5
- Epoch 300-600: μ = -0.8 (标准)

或者用 learnable μ parameterized by a small network conditioned on training progress.

**创新性判断**:
- 课程学习在扩散模型中研究较少
- 结合了"流形首先被学习"的直觉：低噪声时流形更容易估计
- 与 SCMR 可以组合使用

**预期效果**:
- 训练更稳定，初期 loss 下降更快
- 最终 FID 可能持平或略好

---

## 方案比较

| 维度 | SCMR (A) | BMP (B) | MATC (C) |
|------|----------|---------|----------|
| 创新性 | ★★★★★ | ★★★ | ★★★ |
| 实现难度 | 低 (~50行) | 中 (~150行) | 低-中 (~80行) |
| 额外计算 | 1×forward | 0（结构改变） | 0 |
| 理论动机 | 强（直接流形约束） | 中 | 中 |
| 预期FID改善 | +0.5-1.5 | +0-0.5 | +0-0.8 |
| 与JiT兼容性 | 完美（仅加loss） | 好（结构修改） | 完美（调度修改） |
| 论文写作价值 | 高（有理论故事） | 中 | 中 |

## 推荐

**首选 SCMR (方案A)**，理由：
1. 直接解决 JiT 论文的"missing piece"——调用流形但未显式约束
2. 最简单的实现，最小计算开销
3. 最强的理论动机，最清晰的论文叙事
4. 可以自然地与 方案C 组合

**备选组合**：SCMR + MATC（方案A + 方案C）:
- SCMR 提供流形一致性约束
- MATC 提供从易到难的课程
- 两个方案正交，可以叠加

---

## 具体实施计划

1. 在现有的 denoiser.py 中添加 SCMR loss
2. 修改 train.py 支持新 loss 和 λ 超参
3. 在 ImageNet-2000 子集上做消融实验：
   - λ = 0（baseline）
   - λ = 0.01, 0.05, 0.1, 0.5
4. 分析 FID、loss 曲线、生成质量
5. 如效果好，进一步做全量 ImageNet 实验
