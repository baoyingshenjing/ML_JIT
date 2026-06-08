# 第二轮 JiT 改进 — 交叉评审最终综合报告

## 评审统计

| 来源 | 提案数 | 存活 | 死亡 | 需修改 | 
|------|--------|------|------|--------|
| Hermes | 6 (D-I) | 1 | 3 | 2 |
| Codex | 6 (C1-C6) | 2 | 2 | 2 |
| 合计 | 12 | 3 | 5 | 4 |

## 最终存活 & 排行

### Tier 1: 立即实现（高信心）

| 排名 | 来源 | 方案 | 修改后名称 | 理由 |
|------|------|------|-----------|------|
| 1 | Codex | C4 | **结构化噪声过程 (SNP)** | 最原创。随机区域代理，区域内外不同噪声强度。不需要外部模型。Codex评审：5星 |
| 2 | Hermes | E | **时间条件化 Token Merging (TCTM)** | 从pruning改为ToMe-style merging+restore。只在高噪声早期启用。Hermes评审：4星→保留 |

### Tier 2: 值得探索（中信心）

| 排名 | 来源 | 方案 | 修改后名称 | 理由 |
|------|------|------|-----------|------|
| 3 | Codex | C5 | **In-Context Programs (ICP)** | 需先定义评估协议。从class token扩展到multi-token program。 |
| 4 | Hermes+Codex | G+I | **Disagreement-Guided Diffusion (DGD)** | 合并CNLD和HAMP。用SCMR disagreement做uncertainty，调采样步长。 |

### Tier 3: 长期/理论（低优先级）

| 排名 | 来源 | 方案 | 理由 |
|------|------|------|------|
| 5 | Hermes | H (SBMT) | 太重型。适合独立论文 |
| 6 | Codex | C2 | 曲率感知采样 → 简化为per-t importance sampling |

### 死亡名单

| 方案 | 死亡原因 |
|------|---------|
| D (RML) | L_adv无下界，manifold noise无定义 |
| F (MCP) | 只是低秩输出头，不是流形坐标 |
| G (CNLD) | 已被SCMR覆盖，只是采样策略变体 |
| C1 (频带路由器) | 过度工程化，背离JiT简约哲学 |
| C3 (双视角蒸馏) | 不够新颖，与SCMR概念重叠 |
| C6 (少步Solver) | Consistency Models已充分探索 |

## 双方共识

1. **class mean 做 confidence 是错误指标** — HAMP 原方案必须改为 disagreement-based
2. **hard token pruning 不适用 JiT** — TCTP 必须改为 token merging + restore
3. **CNLD = SCMR 的子集** — 不值得独立追
4. **SBMT 不是 JiT 改进** — 是独立研究方向
5. **SCMR 实现正确但叙事需要降级** — 这是双方一致结论

## 双方分歧

| 议题 | Hermes 立场 | Codex 立场 | 裁决 |
|------|-----------|-----------|------|
| RML 可行性 | 5星，理论价值高 | 数学不成立 | Codex 对。L_adv 发散的批评致命 |
| TCTP 难度 | "中" | "中偏难" | Codex 对。RoPE+compile+grid问题 |
| MCP 新颖性 | 4星 | 过度包装 | Codex 对。"流形坐标"无数学约束 |
| C4 实现复杂度 | 中 | 未评审(我评Codex时) | 同意。随机区域代理简单 |

## 推荐第一组实验（3个并行）

### 实验A: 结构化噪声 (SNP)
- 最简版本：随机矩形 mask (size=32-128px)，mask内外不同noise_scale
- mask_in: noise_scale=0.5, mask_out: noise_scale=1.0
- 简单到可以在 denoiser.py 里加 30 行代码
- 对比 baseline no-SNP vs SNP
- 指标：FID, per-t-bin x_pred_mse

### 实验B: Token Merging (TCTM)
- ToMe-style：高噪声时(t<0.3)合并相似patch token
- 保留 source map, final前 restore
- 只在采样时启用（训练不变）
- 指标：wall-clock speed, FID, consistency error

### 实验C: Disagreement 诊断
- 不实现任何新方法
- 用现有 SCMR 代码计算 cross-t x_pred disagreement
- 验证 disagreement 与生成质量(人工评分)的相关性
- 如果相关，DGD 值得追；如果不相关，放弃

## 文件索引

```
docs/
├── RESEARCH_PROPOSAL.md              # Round 1: SCMR/BMP/MATC
├── CODEX_REVIEW.md                   # Codex 对 Round 1 的评审
├── PROPOSALS_ROUND2_HERMES.md        # Hermes Round 2: D-I
├── PROPOSALS_ROUND2_CODEX.md         # Codex Round 2: C1-C6
├── CROSS_REVIEW_HERMES_ON_CODEX.md   # Hermes 评 Codex
├── CROSS_REVIEW_CODEX_ON_HERMES.md   # Codex 评 Hermes
└── CROSS_REVIEW_FINAL.md             # 本文件 — 最终综合
```
