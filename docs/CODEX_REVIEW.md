# JiT 改进方案评审

评审身份：按 senior ML researcher 的标准审 proposal，不按“能不能写成故事”来放水。结论先说：三项里只有 SCMR 值得优先做，但不能按原 proposal 的强 novelty 和 0.5-1.5 FID 改善预期来推进；MATC 可作为低成本辅助消融；BMP 不建议作为主线。

## 代码与目标的关键事实

参考实现中，`Denoiser.forward` 采样 logit-normal `t`，构造 `z = t * x + (1 - t) * e`，网络输出 `x_pred`，但训练 loss 是由 `x_pred` 推出的 velocity MSE：`.ref/denoiser.py:45-65`。这等价于对 `x_pred` 的误差加权：

```text
v - v_pred = (x - x_pred) / (1 - t)
L = ||x - x_pred||^2 / (1 - t)^2
```

因此高 `t` 处的 `x` 预测误差权重大，且被 `t_eps=5e-2` 截断。任何额外的 x-space 正则项都必须考虑这个尺度，否则 proposal 中的 `lambda` 范围没有可解释性。

模型结构方面，bottleneck 只在 patch embedding：`Conv2d(3 -> bottleneck_dim, patch, stride=patch)` 再 `Conv2d(bottleneck_dim -> hidden, 1)`，见 `.ref/model_jit.py:17-37` 和 `.ref/model_jit.py:241`。Transformer block 是 adaLN 调制的 attention + SwiGLU FFN：`.ref/model_jit.py:183-202`。in-context class tokens 在第 `in_context_start` 层插入，forward loop 见 `.ref/model_jit.py:346-352`。训练 loop 目前只接受标量 loss，不记录子 loss：`.ref/engine_jit.py:37-64`。超参入口在 `.ref/main_jit.py:53-57`。

## 方案 A：SCMR

### 1. 技术正确性

核心想法“同一干净图像的不同噪声版本应该预测到同一个 `x`”作为正则项是技术上可尝试的，但 proposal 的数学叙事过强。

最大问题是：标准 JiT 训练已经对每个 noisy view 使用同一个监督目标 `x`。如果对两个 view 分别最小化 `||x_pred_i - x||^2`，那么 `||x_pred_1 - x_pred_2||^2` 在全局最优处自然为 0。SCMR 不引入新的标签信息，它主要改变优化路径和函数平滑性，而不是补上一个“原 loss 缺失的流形约束”。这点必须在论文里讲清楚，否则会被 reviewer 直接质疑为 redundant loss。

更细一点，当前实现的主 loss 不是裸 `x` MSE，而是 `||x - x_pred||^2 / (1 - t)^2`。proposal 的 SCMR 是未加权的 `||xhat1 - xhat2||^2`，这会相对强化低/中 `t` 区域，而不是和主目标同尺度。若 `lambda=0.1`，在高 `t` 处它可能很弱；在低 `t` 处它可能反而强于主目标。这个权重行为需要显式设计，不能只扫一个固定 `lambda`。

另一个数学风险是高噪声区的不可识别性。低 `t` 时 `z_t` 与具体样本身份的信息很少；给定类别和 noisy input，MSE 最优解更接近后验均值而不一定是原始训练样本。SCMR 把两个独立 noisy view 绑定到同一个训练样本，会降低预测方差，但也可能鼓励网络忽略 view-specific 信息，生成更平均、更平滑的 `x_pred`。基准监督 loss 会防止完全坍塌，但不能防止 recall 或细节下降。

还有一个实现层面的隐蔽问题：classifier-free label dropout。两个 SCMR 分支必须共享同一个 `labels_dropped`。如果两个分支独立 dropout，一个分支 conditional、另一个 unconditional，consistency loss 会把条件和无条件预测错误地拉近，直接损害 CFG。

结论：核心 idea 可行，但不是“显式强制流形同一点”的严格数学改进；更准确的说法是 paired noisy-view x0 consistency / variance regularization。

### 2. Novelty

创新性中等偏低，不能声称“扩散模型中未见类似做法”。

相关工作脉络很强：

- Denoising autoencoder / denoising score matching 本身就是从不同 corruptions 恢复同一 clean sample 或学习 score 的范式。典型参考包括 Vincent et al. 的 denoising autoencoder，以及 Alain & Bengio 关于 DAE 学到 score 的分析。
- Consistency Models 明确学习把同一 probability-flow ODE trajectory 上不同时间点映射到一致输出。SCMR 不完全相同，因为 pair 来自同一 `x` 的独立噪声而不是同一 ODE trajectory，但“跨噪声/时间的一致映射”不是新问题。
- BYOL / SimSiam 这类 SSL 方法把同一样本的不同 view 拉到一致表示。SCMR 把 view consistency 放在像素级 `x_pred` 而不是 latent representation，差异存在，但思想亲缘很近。
- 扩散训练里的 self-conditioning、consistency distillation、teacher-student consistency、multi-view denoising 都会被 reviewer 拿来比较。proposal 当前没有和这些方法划清边界。

可 defend 的 novelty 角度是窄化后的：在 JiT 的 pixel-space x-prediction flow matching 框架中，对同一样本的两个独立 noising path 施加 x0 prediction consistency，并研究它是否改善少步 ODE 采样。这个角度可以作为 empirical contribution，不足以单独支撑强理论 claim。

### 3. 实现可行性

实现难度低到中，但不是 proposal 说的“仅 50 行”那么干净，因为需要可比性和 logging。

需要改的文件：

- `.ref/denoiser.py`
  - `__init__` 增加 `scmr_lambda`、`scmr_t_mode`、`scmr_stopgrad`、`scmr_weighting` 等参数。
  - `forward` 里采样第二组 `t2/e2/z2`，复用同一个 `labels_dropped`。
  - 计算 `x_pred1/x_pred2`，主 loss 保持 baseline；新增 `L_scmr`。
  - 建议返回 `(loss, metrics)` 或把子 loss 写到属性中，便于训练日志记录。
- `.ref/main_jit.py`
  - argparse 增加 SCMR 超参。
- `.ref/engine_jit.py`
  - 如果 `forward` 返回 dict/tuple，需要兼容。
  - TensorBoard 记录 `loss_flow`、`loss_scmr`、`scmr_lambda`。
- 可选 `.ref/model_jit.py`
  - 不需要改模型本体。

推荐实现细节：

- 使用同一个 `labels_dropped`。
- 初始实验不要用 `lambda=0.5`，先用 `0.001, 0.003, 0.01, 0.03`；当前 loss 尺度下 `0.1` 已经可能过强。
- 对 SCMR 做 warmup，例如前 20-50 epochs 从 0 线性升到目标值。
- 加一个 stop-gradient/EMA teacher 版本：`||sg(xhat1) - xhat2||^2` 或 EMA net 作为 target。虽然主 loss 已有 anchor，但 stop-grad 可减少两个分支互相追逐的噪声。
- 限制或重权 `t` 区间，至少按 t-bin 记录 loss。不要默认全区间同权。

### 4. 预期影响

proposal 估计 JiT-B/16 FID 从约 8.6 到 7.x，过于乐观。

理由：

- 主 loss 已经直接监督 `x_pred -> x`，SCMR 的新增信息有限。
- 额外一倍 forward/backward 使训练 compute 接近翻倍。若按相同 wall-clock 比较，SCMR 可能等价于减少训练 epochs；若按相同 epochs 比较，它不是免费提升。
- FID 对 ImageNet-256 的方差和 CFG/EMA/采样设置敏感，0.5 FID 以内的变化需要多 seed 和严格评估。
- 若 SCMR 改善少步采样，最可能体现在 10/25 steps，而不一定改善 50 steps FID。

更合理预期：若调得好，50-step FID 可能改善 0.1-0.4；少步采样可能有 0.3-0.8 的机会。也有相当概率 FID 不变或变差但 consistency metric 变好。

### 5. 最大风险

最大风险是“看似理论漂亮，实际只是 redundant regularization”。如果没有证明它优于“第二个 noisy view 也做标准 supervised loss”或“同等 compute 训练更久/更大 batch”，论文贡献会被削弱。

次级风险是高噪声区过度一致导致样本细节平均化，FID 可能小幅改善但 precision/recall 或视觉多样性下降。

## 方案 B：BMP

### 1. 技术正确性

这个方案的核心数学 claim 最弱。`x -> Linear(hidden -> bottleneck) -> SiLU -> Linear(bottleneck -> hidden) + x` 带 residual bypass，并不会“迫使中间表示压缩到低维”。主信息可以完全走 identity residual，bottleneck 分支只是一个低秩残差 adapter。除非去掉 residual、惩罚 residual、或让主通路也经过瓶颈，否则它不是投影，也不是逐层流形检查点。

如果去掉 residual，它会强行降低 Transformer 表示容量，极可能破坏 JiT 已经很紧的欠容量假设。JiT-B/16 本身只有 12 层、hidden 768，输入 bottleneck 128 已经是强约束；再在中间层压缩，未必符合“中间表示也低维”的假设。中间 representation 通常需要展开条件、位置、噪声水平和局部结构，不等同于数据流形坐标。

proposal 还声称额外计算为 0，这是不对的。每个 projector 至少增加两次 dense projection 和激活，虽然相对 attention/FFN 较小，但参数、FLOPs、activation memory 都非零。

结论：作为 adapter-style capacity tweak 可行；作为“manifold projection”理论不成立。

### 2. Novelty

创新性低。

这个结构和 Houlsby-style Transformer adapters 非常接近：down-projection、nonlinearity、up-projection、residual add。LoRA/低秩 residual 更新也是同一类参数化思想。T2I-Adapter、ControlNet、diffusion adapter 方向也已经大量使用额外分支/adapter 给 diffusion backbone 注入能力。BMP 的区别主要是给 adapter 一个“流形”解释，但结构本身没有体现流形约束。

如果要写论文，必须承认它是 bottleneck adapter baseline，而不是 novel module。

### 3. 实现可行性

实现中等简单，但会碰到 initialization 和 in-context token 的细节。

需要改的文件：

- `.ref/model_jit.py`
  - 新增 `BottleneckProjector` 或 `BottleneckAdapter` 类。
  - `JiT.__init__` 增加 `mid_bottleneck_dim`、`mid_bottleneck_every`、`mid_bottleneck_zero_init`。
  - 在 `self.blocks` 旁边创建 `self.mid_projectors`，或在 forward loop 中按层插入。
  - forward loop `.ref/model_jit.py:346-352` 中要注意第 `in_context_start` 层之后 token 序列包含 class tokens；projector 若作用于所有 tokens，会改变 class in-context tokens；若只作用 image tokens，实现更复杂。
  - `initialize_weights` 中最好 zero-init up projection 或 learnable scalar gate，使初始模型等价 baseline。
- `.ref/main_jit.py`
  - 增加结构超参。
- `.ref/denoiser.py`
  - 创建 `JiT_models[args.model]` 时传入这些结构超参。

强烈建议 zero-init residual 分支，否则会破坏 JiT 当前的零初始化输出路径：`.ref/model_jit.py:305-315`。

### 4. 预期影响

FID 改善 `0-0.5` 也偏乐观，但比 SCMR 的估计保守。真实结果大概率是：

- JiT-B：0 或负收益。adapter 增加少量容量，但没有明显改善采样 ODE 或 x-prediction bias 的机制。
- JiT-H：不一定更有用。proposal 说大模型容易过拟合，但 ImageNet diffusion 大模型常常仍受优化/compute/数据增强影响，不能默认 overfit。若 H 使用 `proj_dropout=0.2`，增加 adapter 可能反而需要重新调 dropout 和 LR。

合理预期：作为 appendix ablation，若 zero-init/gated 设计，可能在一些设置有 0.0-0.2 FID 波动；不应作为主贡献押注。

### 5. 最大风险

最大风险是理论叙事站不住。reviewer 会指出 residual adapter 不压缩表示，因此“流形投影瓶颈”的名字和数学行为不一致。若改成真正投影，又容易损害容量和稳定性。

## 方案 C：MATC

### 1. 技术正确性

课程学习的直觉合理，但 proposal 里有几个技术矛盾。

第一，proposal 说“后期均匀采样”，但具体 schedule 是 `mu -> -0.8`，这不是均匀分布，而是回到 JiT baseline 的 logit-normal。需要统一定义：到底是 curriculum to baseline，还是 curriculum to uniform。

第二，`t` 的难度假设过于简单。JiT 的 `t=1` 接近 clean、`t=0` 接近 noise。虽然低 `t` 的样本身份难恢复，但当前 velocity MSE 等价于 `x` 误差除以 `(1-t)^2`，所以高 `t` 的错误权重更大。训练早期把 `mu` 推到 0.5 会采更多高 `t`，未必更容易，甚至可能放大初始 `x_pred=0` 时的 loss 和梯度。

第三，改变 `t` 采样分布会改变训练目标。若只是早期 curriculum 再回到 baseline，最后可能被长时间 baseline 训练覆盖；若持续非 baseline，则需要 importance weighting 或重新解释目标分布。proposal 没有说明这一点。

第四，用 SCMR loss “监控流形一致性作为课程进展信号”不够严谨。SCMR 变小可能代表模型更一致，也可能代表输出更平均或低方差。它不能单独作为 curriculum readiness 指标。

结论：MATC 作为工程 schedule 可试；作为 manifold-aware 理论贡献较弱。

### 2. Novelty

创新性低到中。

扩散训练已有多类 timestep/noise-level 采样和权重策略：P2 weighting、Min-SNR weighting、loss-aware/adaptive timestep sampling、课程式 diffusion 训练、perception-prioritized 或 difficulty-based schedule。MATC 的“从易到难调噪声分布”是自然变体，不是明显新方向。

能 defend 的角度是：针对 JiT 的 x-prediction + logit-normal t 采样，系统研究 `P_mean` schedule 与少步采样/FID 的关系。但这更像 empirical training recipe。

### 3. 实现可行性

实现低到中，主要取决于是否要 per-iteration schedule。

需要改的文件：

- `.ref/denoiser.py`
  - `sample_t` 支持传入当前 `P_mean` 或 progress。
  - 或增加 `set_train_progress(epoch_float)`，由训练 loop 更新内部状态。
- `.ref/engine_jit.py`
  - 在 `train_one_epoch` 中已有 `data_iter_step / len(data_loader) + epoch`，可把这个 progress 传给 model：`.ref/engine_jit.py:29-38`。
  - 记录当前 `P_mean` / effective mean(t)。
- `.ref/main_jit.py`
  - 增加 `matc_enable`、`matc_mu_start`、`matc_mu_mid`、`matc_mu_end`、`matc_schedule_epochs` 等参数。

不建议第一版做 learnable `mu` network。它会让训练分布和模型参数耦合，难以复现，也会增加 checkpoint/分布式同步/日志解释问题。

### 4. 预期影响

proposal 的 `0-0.8` FID 改善偏乐观。固定 logit-normal `P_mean=-0.8, P_std=0.8` 很可能是 JiT 论文或实现已经调过的强 baseline。课程调度更可能影响早期 loss 曲线，而不是最终 FID。

合理预期：最终 50-step FID 约 -0.2 到 +0.2 波动；若结合 SCMR，可能帮助避免早期高噪声 consistency 过强，但也可能降低最终 hard-noise 覆盖。少步采样可能比 full 50-step 更敏感。

### 5. 最大风险

最大风险是训练分布 mismatch。模型早期欠训练低 `t`，后期如果 baseline 分布训练不够久，会影响从纯噪声出发的 ODE 采样；如果后期训练很久，课程效果又可能消失。

## Final Verdict

### 哪些值得做

1. 优先做 SCMR，但要降级为“paired noisy-view x0 consistency regularization”，不要宣传为未被探索的显式流形约束。
2. MATC 只作为 SCMR 的辅助消融或训练稳定性实验，不建议单独作为论文主贡献。
3. BMP 暂不追。除非主线失败且需要结构类 appendix，否则不要投入大规模 ImageNet 训练。

### 建议修改或组合

SCMR 修改版：

- 共享 label dropout。
- `lambda` warmup。
- 先扫小权重：`0.001, 0.003, 0.01, 0.03`，不要从 `0.1/0.5` 开始。
- 加 stop-gradient 或 EMA-teacher variant。
- 对 `t` 分桶记录 `flow_loss`、`scmr_loss`、`x_pred MSE`。
- 做一个“第二 noisy view 只加标准 supervised loss，不加 consistency”的 compute-matched baseline。这个基线极关键。

SCMR + MATC 组合：

- 不要用 learnable `mu`。
- 可试简单 schedule：前 50-100 epochs 从 `P_mean=0.0` 或 `-0.2` 退火到 `-0.8`，之后完全 baseline。
- 若 SCMR 在低 `t` 过强，可只对 SCMR 分支采中等 `t`，而不是全局改变主训练分布。

BMP 若非做不可：

- 改名为 bottleneck adapter，不叫 manifold projection。
- residual 分支必须 zero-init 或 gate-init 0。
- 必须加 same-param adapter baseline，证明不是单纯参数量收益。
- 比较 projector 作用于 all tokens vs image tokens only。

### 第一批关键实验

先不要上全量 600 epoch。建议先做 ImageNet-2000 或 100-class subset，但评估设计要能排除假阳性。

1. Baseline JiT-B/16 subset，固定 seed、CFG、EMA、50/25/10 sampling steps。
2. SCMR 小扫：`lambda = 0.001, 0.003, 0.01, 0.03`，带 warmup。
3. Compute-matched baseline：每 step 两个 noisy views，各自对 `x` 做标准 loss，训练 compute 与 SCMR 相同。
4. Stop-grad SCMR vs symmetric SCMR。
5. SCMR t 区间 ablation：all t、low t only、mid t only、high t only。
6. CFG label dropout ablation：共享 dropout 是默认；独立 dropout 只作为 sanity check，预期应变差。
7. MATC 固定 schedule ablation：baseline `P_mean=-0.8`、固定 `P_mean=-0.2/0.0`、curriculum to `-0.8`。
8. 指标不要只看 FID：同时看 IS、precision/recall、per-t x_pred MSE、SCMR consistency metric、生成多样性和少步采样质量。

### 缺失 baselines / ablations

- 同等 compute baseline：非常关键，否则 SCMR 的收益可能只是多看了一个 noisy augmentation。
- 同等 wall-clock 或同等 forward 数 baseline。
- 固定 `P_mean` sweep，证明 MATC 优于简单重调 t 分布。
- `lambda=0` 且代码路径开启的 sanity baseline，排除实现差异。
- 随机错配 pair 的负对照：把 `xhat1` 和另一个样本的 `xhat2` 拉近应明显伤害质量，用来证明 pair identity 重要。
- 只加 `||xhat - x||^2` x-space auxiliary loss 的 baseline，因为当前主 loss 是 velocity-weighted x MSE，SCMR 的一部分收益可能来自改变 x-space 权重。
- 多 seed 小规模实验。少于 3 seeds 的 0.2 FID 以内差异不应写成结论。

## 参考文献与相关工作线索

- JiT / Back to Basics: Let Denoising Generative Models Denoise: https://arxiv.org/abs/2511.13720
- Consistency Models: https://arxiv.org/abs/2303.01469
- Denoising Diffusion Probabilistic Models: https://arxiv.org/abs/2006.11239
- Extracting and Composing Robust Features with Denoising Autoencoders: https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf
- What regularized auto-encoders learn from the data-generating distribution: https://arxiv.org/abs/1211.4246
- BYOL: Bootstrap Your Own Latent: https://arxiv.org/abs/2006.07733
- SimSiam: Exploring Simple Siamese Representation Learning: https://arxiv.org/abs/2011.10566
- Parameter-Efficient Transfer Learning for NLP / Adapters: https://arxiv.org/abs/1902.00751
- LoRA: Low-Rank Adaptation of Large Language Models: https://arxiv.org/abs/2106.09685
- P2 weighting for diffusion training: https://arxiv.org/abs/2204.00227
- Min-SNR diffusion training strategy: https://arxiv.org/abs/2303.09556
- Denoising Task Difficulty-based Curriculum for Training Diffusion Models: https://arxiv.org/abs/2403.10348
