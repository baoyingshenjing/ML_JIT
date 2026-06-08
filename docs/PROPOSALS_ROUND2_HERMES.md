# 第二轮 JiT 改进提案 — Hermes 独立提出

## 方法论

不重复第一轮。本轮聚焦于 JiT 论文 AND 代码中**真正没有被触碰的维度**：
架构层面、训练范式层面、理论框架层面。每个方案必须回答："为什么 JiT 作者没想到/没做？"

---

## 方案 D: Reciprocal Manifold Learning (RML)

**直觉**：
JiT 只训练单向映射 f: z_t → x。但如果数据在流形 M 上，那么存在一个"流形保持"的加噪方式：
普通的加噪 z_t = t·x + (1-t)·ε 是**脱离流形**的（ε 不在 M 上）。
如果反过来，训练一个生成器 g: x → ε_M，其中 ε_M 是"流形保持"的噪声
（即 z_t = t·x + (1-t)·ε_M 仍然在 M 附近），那么：
- denoiser f 需要区分"在流形上的噪声"和"脱离流形的噪声"
- generator g 被迫学习流形的局部几何

**具体实现**：
```python
# 标准 JiT forward (noise → clean)
z = t*x + (1-t)*e          # e ~ N(0,I), off-manifold
x_pred = f(z, t)

# 新增 Reciprocal forward (clean → manifold-noise)
e_M = g(x, t)                # learned manifold noise
z_M = t*x + (1-t)*e_M        # on-manifold noisy version

# Losses:
L_flow = ||f(z) - x||² / (1-t)²     # standard
L_recip = ||f(z_M) - x||²           # manifold-consistent recovery
L_adv = -||g(x) - e||²              # push g away from trivial e
```

**为什么 JiT 没做**：
JiT 的核心主张是"不需要学流形，只需要预测干净数据"。RML 突破了这一范式，
显式学习流形的局部结构，是对 JiT 哲学的有意义延伸。

**创新性**：★★★★★
- 双向训练在扩散中未见（据我所知）
- 与 CycleGAN 精神相似但动机完全不同
- 理论 story：「要证明你理解流形，就要能在流形上行走」

**风险**：g 可能坍缩到恒等映射（e_M ≈ ε），需要对抗性训练来防止

---

## 方案 E: Time-Conditioned Token Pruning (TCTP)

**直觉**：
JiT 的 patch token 数量在整个去噪过程中保持不变。但在 t→0（高噪声）时，
输入几乎是纯噪声——单个 token 无法从噪声中提取有意义的 patch 信息。
此时保留全部 256 个 token（对于 256px/16patch）是浪费计算，
且在低 SNR 下注意力可能被噪声主导。

反过来，随着 t→1（低噪声），逐 patch 的细节信息越来越多，
需要全分辨率。

**具体实现**：
在每个 Transformer block 之后，根据当前 t 计算一个重要性分数，
剪掉分数最低的 k(t) 个 token。k(t) 随 t 单调递减：
```
k(t) = round(K_max * (1 - t)^γ)
```
- t=0（纯噪声）：保留最少 token（如 16 个）
- t=1（干净）：保留所有 token（256 个）

剪枝策略：用 attention scores 或 token norms 作为重要性指标。

**优势**：
- 训练加速：高噪声区的 token 更少，前向/反向更快
- 采样加速：同样受益
- 正则化效果：强迫模型在高噪声区学习紧凑表示
- 理论一致性：流形是低维的 → 不需要完整 token 集

**为什么 JiT 没做**：
JiT 强调 simplicity。动态 token pruning 增加了复杂度，
但收益（速度+质量）值得这个代价。

**创新性**：★★★★
- Token pruning 在 ViT 分类中存在（DynamicViT, EViT, ToMe）
- 但在扩散 Transformer 中做 time-conditioned pruning 是新的
- 关键区别：pruning ratio 由 t 决定，不是可学习的

**风险**：剪枝策略不稳定可能导致训练崩溃；需要梯度估计（straight-through）

---

## 方案 F: Manifold Coordinate Prediction (MCP)

**直觉**：
JiT 在 768 维（ViT 隐藏维度）空间中操作，但数据流形 M 的内在维度远低于 768。
为什么不直接预测流形坐标？

如果存在一个流形参数化 φ: R^d → R^D（d << D），那么：
- x_pred 可以从流形坐标恢复：x = φ(c) + residual
- 模型只需要预测 d 维坐标 c（加上一个小 residual）

**具体实现**：
修改 JiT 的输出头：不是直接 Linear(hidden → patch*patch*3)，
而是 Linear(hidden → d) + Linear(hidden → patch*patch*3) 但后者从小初始化。
```
c = manifold_head(token)          # d-dim manifold coordinate
r = residual_head(token) * α      # α << 1, small residual
x_pred = manifold_decoder(c) + r   # decoder is shared across tokens
```
其中 manifold_decoder 是一个小型网络或可学习的基函数集合。

**为什么 JiT 没做**：
JiT 坚持"无额外设计"。MCP 增加了结构复杂性（decoder），
但明确地将流形学习嵌入架构中。

**创新性**：★★★★
- 在生成模型中显式预测流形坐标是新颖的
- 与 NeRF 的坐标网络有相似性，但用于不同目的
- 与 VAE 的 latent space 不同：MCP 的坐标是结构化的（跨 patch 共享解码器）

**风险**：d 的选择是关键超参；如果 d 太小，质量下降；太大则退化到原始 JiT

---

## 方案 G: Cross-Noise-Level Distillation (CNLD)

**直觉**：
JiT 在 t=1 附近的预测非常准确（输入已经接近干净），
在 t=0 附近的预测非常困难（输入几乎是纯噪声）。
如果我们让"容易的 t"的预测指导"困难的 t"的预测，效果如何？

这与知识蒸馏不同——教师和学生是**同一个模型**，只是输入的噪声水平不同。

**具体实现**：
采样 t_easy ~ [0.7, 1.0], t_hard ~ [0.0, 0.3]
```
x_pred_easy = f(z_t_easy, t_easy)     # high-quality prediction
x_pred_hard = f(z_t_hard, t_hard)     # low-quality prediction
L_distill = ||x_pred_hard - sg(x_pred_easy)||²
```
关键：easy 预测不需要完美——它只需要比 hard 预测更接近真相。

**为什么 JiT 没做**：
这是一种 self-distillation 形式，JiT 没有探索。

**创新性**：★★★
- 与 self-distillation / BYOL 相似但用于跨噪声水平
- 不是全新的范式，但在扩散 specific 的应用是新的

**风险**：如果 x_pred_easy 本身不够好，蒸馏会传播错误

---

## 方案 H: Schrödinger Bridge Manifold Transport (SBMT)

**直觉**（最深的理论方案）：
JiT 使用线性插值 z_t = t·x + (1-t)·ε。但这是"任意"的传输路径。
如果数据的真实动态是在流形上演化的呢？

Schrödinger Bridge 问题寻找在两个分布之间的最优传输，
受布朗运动约束。如果我们把 JiT 重新解释为求解一个 Schrödinger Bridge：
- 从 t=0（噪声分布）到 t=1（数据流形 M）
- 最优传输路径应该在流形上或接近流形

**具体实现**：
不采样固定的线性插值，而是学习一个随机的 forward process：
```
dz_t = v_θ(z_t, t)·dt + σ·dB_t
```
其中 v_θ 就是 JiT 预测的速度场（即 x_pred 推出的 v_pred）。
但加入布朗运动项 σ·dB_t（随机性）使得过程成为真正的 SDE。

训练时使用 Schrödinger Bridge 的 IPF（Iterative Proportional Fitting）算法：
1. 训练 forward SDE（从数据到噪声）
2. 训练 backward SDE（从噪声到数据）
3. 交替迭代

**为什么 JiT 没做**：
JiT 故意简化。SBMT 在理论上更深，但实现复杂得多。
论文中提及 ODE 但没有讨论 SDE 或最优传输。

**创新性**：★★★★★
- Schrödinger Bridge + 扩散 Transformer 的组合非常新
- 2024-2025 年 SB 在生成模型中开始出现但远未成熟
- 如果成功，这是一个重要的理论贡献

**风险**：极高。IPF 需要交替训练，收敛慢。可能计算成本过大。

---

## 方案 I: Hallucination-Aware Manifold Projection (HAMP)

**直觉**：
JiT（以及所有 x-prediction 扩散模型）在高噪声区域有一种特殊的失败模式：
模型会"幻想"出训练数据中常见的模式，而不是从噪声中提取真实信号。
这是因为在高噪声下，贝叶斯最优估计趋近于类条件均值。

**具体实现**：
对于每个生成的 x_pred，计算一个"可信度"分数：
```
confidence = exp(-β * ||x_pred - μ_class||²)
```
在低可信度区域，模型回溯并使用更保守的预测（类条件均值）。
在采样时，对于低可信度的预测，施加更强的 CFG 或使用更小的步长。

更激进的版本：训练时给"幻想"样本更高的 loss 权重，
迫使模型在不确定时说"我不知道"而不是编造。

**为什么 JiT 没做**：
JiT 没有讨论 failure mode 或可信度。

**创新性**：★★★★
- 扩散模型的 hallucination 问题最近开始被讨论
- 可信度引导的生成是新的（在扩散中）

**风险**：可信度度量的定义可能本身就不准确

---

## 方案比较矩阵

| 方案 | 创新性 | 实现难度 | 理论深度 | 预期FID改善 | 计算开销 | 首选 |
|------|--------|---------|---------|------------|---------|------|
| RML (D) | ★★★★★ | 中高 | 深 | 0.2-0.8 | 1.5x | ★★ |
| TCTP (E) | ★★★★ | 中 | 中 | 0.1-0.4 | 0.6-0.8x | ★★★ |
| MCP (F) | ★★★★ | 中 | 中深 | 0.1-0.5 | 1.0x | ★★ |
| CNLD (G) | ★★★ | 低 | 浅 | 0.1-0.3 | 1.3x | ★ |
| SBMT (H) | ★★★★★ | 极高 | 极深 | ? | 2-3x | ★ (理论研究) |
| HAMP (I) | ★★★★ | 中低 | 中 | 0.2-0.5 | 1.1x | ★★★ |

## 推荐

**实用首选**: TCTP (E) + HAMP (I) 组合
- TCTP 提供速度提升 + 理论一致性
- HAMP 解决 hallucination 问题
- 两者都是"低风险、中等收益"
- 与 SCMR (第一轮) 正交，可叠加

**理论首选**: RML (D) 如果做理论研究
**高风险高回报**: SBMT (H) 如果追求顶会 best paper
