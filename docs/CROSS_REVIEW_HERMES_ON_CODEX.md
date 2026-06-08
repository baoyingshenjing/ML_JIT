# Codex 第二轮提案 — Hermes 交叉评审

## 总体评价

Codex 的 6 个提案在"大胆程度"上胜过我。其中 #4（语义噪声过程）和 #5（In-Context Programs）是真正有原创性的方向。但多数提案存在过度工程化的问题——与 JiT "back to basics" 哲学冲突。

---

## C1: 自适应频带去噪路由器 ★★☆☆☆

**技术判定**: 部分成立，但过度复杂

频带分解 + MoE 路由的组合非常重。Wavelet/MoE 组合在图像恢复中有前作，
但在 diffusion 中应用的新颖性有限。更大的问题是：JiT 的核心论点是
"欠容量就够了"——如果加 MoE 扩展容量，等于背离了这个论点。

**与 JiT 兼容性**: 低。MoE 增加参数量，破坏"极简"叙事。
**与 SCMR 兼容性**: 中。正交但不互补。
**建议**: 不追。如果要做频带感知，简化为 frequency-aware loss weighting 即可。

---

## C2: 曲率感知时间采样 ★★★★☆

**技术判定**: 成立，值得探索

这个方向很好。logit-normal 是固定先验，但训练中不同 t 的难度确实在变化。
二阶差分曲率估计是合理的在线难度指标。

**问题**: 
1. 需要 3 次前向（t-δ, t, t+δ），训练成本 ×3
2. EMA histogram 收敛可能很慢
3. "曲率"指标的噪声可能很大（特别是在早期训练）

**与 JiT 兼容性**: 高。不改变架构，只改采样分布。
**与 SCMR 兼容性**: 高。SCMR 需要在不同 t 做 consistency，曲率感知可以指导 SCMR pair 的选择。

**建议**: 简化版本可以追。不做 3 点曲率估计，只用 velocity MSE 的 per-t 统计做 importance sampling。更简单的 adaptive t-sampling 可参考 Prioritized Experience Replay 的思路。

---

## C3: 自蒸馏式双视角 x-prediction ★★☆☆☆

**技术判定**: 可行但不够新颖

双头 + 跨尺度一致性约束在超分辨率和多尺度生成中大量存在。
Coarse-to-fine diffusion 有 MDT, DiffuseVAE 等前作。
与 SCMR 有概念重叠（都是跨视角 consistency），但 C3 是跨尺度，SCMR 是跨噪声。

**与 JiT 兼容性**: 低。双头 + 卷积分支破坏 ViT-only 设计。
**与 SCMR 兼容性**: 低。两个 consistency loss 可能互相干扰。

**建议**: 不追。如果关心局部细节，TCTP（我的 E 方案）或更小的 patch 是更干净的解决方案。

---

## C4: 语义噪声过程 ★★★★★ 本轮最佳之一

**技术判定**: 成立，非常原创

这是 Codex 最好的提案。核心洞察非常深刻：标准扩散假设所有像素以同样速率退化，
但现实中对象的语义结构比纹理更稳定。结构化噪声过程让模型学习语义层次的恢复。

**优点**:
- 不需要外部 tokenizer（关键！与 JiT 哲学兼容）
- 弱对象代理（超像素/颜色连通/随机矩形）是无监督的
- 区域感知的噪声 schedule 是理论上合理的方向

**问题**:
- 弱对象代理可能非常 noisy，early training 时质量极低
- 如何确保代理"弱"得够但又"有用"——这是个 Goldilocks 问题
- 需要为每个样本动态构造区域噪声，batch 内可能有很大方差

**与 JiT 兼容性**: 中。增加了噪声过程的复杂度，但可以用最简单的随机矩形代理起步。
**与 SCMR 兼容性**: 中。SCMR 的 pair 可以用同一区域划分但不同噪声强度。

**建议**: 强烈建议追。先做最简版本：随机矩形 mask，mask 内外不同噪声强度。
如果这个最简版本能在 2000-image subset 上显示改善，再考虑更强的代理。

---

## C5: In-Context Generation Programs ★★★★☆

**技术判定**: 成立，方向正确但评估难

将 in-context tokens 从 class-only 扩展为 multi-token program 是个自然但强大的延伸。
JiT 已经在用 in-context tokens，扩展它们是"不增加架构，只改变训练数据"的最干净方式。

**问题**:
1. 评估极其困难——ImageNet FID 不能衡量"可控性"
2. Program dropout 和 permutation 是必要的，但增加了训练复杂度
3. "style prototype" 如何从 ImageNet 自动构造？

**与 JiT 兼容性**: 极高。只改变 in-context tokens 的内容和训练方式，不改变架构。
**与 SCMR 兼容性**: 高。program tokens 在 SCMR 的两个分支间保持一致。

**建议**: 值得追，但需要先定义评估协议。可以先用简单的扩展——class token + count token + random crop box——看看模型是否能学会组合这些信号。

---

## C6: 一致性校正的少步 JiT Solver ★★☆☆☆

**技术判定**: 可行但已有大量相关工作

这个方向本质上是 Consistency Models / Consistency Distillation 的变体。
Song et al. (2023) 的 Consistency Models 和后续的 CTM, LCM 等已经充分探索。
在 JiT 上做少步一致性训练不是新的学术贡献。

**与 JiT 兼容性**: 中。需要修改训练目标。
**与 SCMR 兼容性**: 中。两个都是 consistency 约束，但维度不同。

**建议**: 不追。除非能证明 JiT 的 x-prediction 使得少步一致性训练比其他扩散模型显著更有效。

---

## Head-to-Head 最终排行

| 排名 | 谁的 | 方案 | 评分 | 建议 |
|------|------|------|------|------|
| 1 | Codex | C4: 语义噪声过程 | ★★★★★ | 强烈追 |
| 2 | Codex | C5: In-Context Programs | ★★★★☆ | 值得追 |
| 3 | Hermes | E: TCTP | ★★★★☆ | 值得追 |
| 4 | Codex | C2: 曲率感知采样 | ★★★★☆ | 简化后追 |
| 5 | Hermes | D: RML | ★★★★★ | 理论优先 |
| 6 | Hermes | I: HAMP | ★★★★☆ | 值得追 |
| 7 | Hermes | F: MCP | ★★★★☆ | 中等 |
| 8 | Codex | C1: 频带路由器 | ★★☆☆☆ | 不追 |
| 9 | Codex | C3: 双视角蒸馏 | ★★☆☆☆ | 不追 |
| 10 | Codex | C6: 少步 Solver | ★★☆☆☆ | 不追 |
| 11 | Hermes | G: CNLD | ★★★☆☆ | 不追 |
| 12 | Hermes | H: SBMT | ★★★★★ | 长期理论 |
