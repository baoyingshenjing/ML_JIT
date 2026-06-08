# Codex 对 Hermes Round 2 的交叉评审

评审立场：按严苛 ML reviewer 标准看，不按“创意 brainstorm”标准看。很多提案的问题不是“还需要调参”，而是核心定义没有闭合、和当前 JiT 代码路径不匹配，或者把已有文献重新命名成新概念。

当前代码事实先钉死：

- JiT 训练是 `x0` prediction 的 flow matching：`z = t*x + (1-t)*e`，网络输出 `x_pred`，再转成 `v_pred = (x_pred-z)/(1-t)`。见 `src/jit_repro/denoiser.py`。
- 采样是 ODE Euler/Heun，状态始终是完整像素张量。见 `src/jit_repro/sampling.py`。
- 模型输出头是每个 patch token 直接线性投到 `patch_size*patch_size*3`，再 `unpatchify` 成图像。见 `src/jit_repro/model.py`。
- 代码里已经有 `SCMR`：同一样本两个噪声视角的 `x_pred` 一致性损失，且支持 stop-grad。Hermes 的 CNLD 不是从零出现的新方向，而是当前仓库已有机制的一个窄化版本。

## D. RML: Reciprocal Manifold Learning

### 1. 技术 soundness

这个方案数学上最危险，甚至比它看起来更不成立。

最大问题：`g(x,t)` 被称为 “manifold noise”，但没有任何约束能保证 `e_M` 在数据流形上，也没有保证 `z_M = t*x + (1-t)*e_M` 靠近数据流形。自然图像流形不是凸集。两个“看起来像图像”的点线性插值也经常离开图像流形；更不用说 `e_M` 如果要同时像噪声、又在流形上，本身就是未定义对象。

`L_adv = -||g(x)-e||^2` 更是直接错误。这个 loss 对 `g` 是无下界的：最优解是把输出范数推到无穷大，而不是学“非平凡流形噪声”。如果加 norm clamp 或 discriminator，方案才有定义；但那已经不是文中给出的方案。

还有一个致命 mismatch：采样从高斯噪声开始，而 RML 训练额外喂给 denoiser 的是 `z_M` 分布。若 `z_M` 的边缘分布不同于采样轨迹上的 `z_t`，训练只是在制造 OOD 辅助样本。除非同时改 forward process 和 sampler，否则它不一定提升主 FID。

“JiT 只训练单向映射 f: z_t -> x”这个表述也过粗。代码里网络输出 `x_pred`，但训练 loss 实际通过 `v_pred` 匹配 flow velocity。把它说成单向回归会掩盖采样路径一致性问题。

### 2. 新颖性

不应给五星。最接近的已有工作包括：

- Denoising Diffusion GANs：每个 denoising step 用生成器建模多模态条件分布，已经把 GAN/对抗思想引入 denoising 过程。
- Schrödinger Bridge / diffusion bridge 系列：直接学习两个分布之间的 stochastic bridge，而不是固定线性插值。
- manifold-constrained / Riemannian diffusion：讨论在约束流形或非欧空间上的 diffusion。
- CycleGAN / cycle consistency 只是最弱的相似点；RML 真正相近的是“学习 forward/noising process”和“bridge matching”，不是简单双向训练。

所以 RML 更像把 learned noising + adversarial regularization + manifold story 混在一起，并没有给出一个新的、可检验的目标。

### 3. 在 JiT 代码中的实现可行性

会破坏当前代码的简洁假设：

- 需要新增 `g` 网络、优化器参数组、EMA 策略和 checkpoint schema。
- `Denoiser.forward` 目前只采样一个高斯 view；RML 至少要多一次生成器 forward、一次 denoiser forward，显存和 wall time 约 1.8-2.2x，而不是轻微开销。
- `g` 输出像素空间 3xHxW，若用同等级 Transformer 成本不可接受；若用小 CNN，又很难相信它学到所谓“流形局部几何”。
- `L_adv` 若真按文中写，会数值发散；必须重写成 bounded objective。
- sampler 不改时训练分布和采样分布不一致；sampler 改时就不再是当前 JiT ODE。

### 4. FID 改善估计

Hermes 估计 `0.2-0.8` 过于乐观。我会估：

- 按原方案实现：大概率 FID 变差或训练发散。
- 修成 bounded auxiliary noising：可能 `-0.2 到 +0.1`，需要很强 ablation 才能证明不是正则化噪声。
- 想拿 `0.5+` FID 改善，必须证明 learned forward process 也用于采样，并且训练稳定；这已经是另一个项目。

### 5. 最大风险

生成器学会制造 denoiser 的 adversarial/OOD 输入，主模型花容量修补辅助分布，最终采样轨迹上更差。这个失败模式非常现实。

## E. TCTP: Time-Conditioned Token Pruning

### 1. 技术 soundness

这是六个里最可救的一个，但原提案把 “token pruning 能省算” 说得太轻松。

高噪声时 token 信息少，不等于可以随便删空间 token。JiT 的输出是完整图像，`unpatchify` 要求 token 数是完整平方网格。删 token 后必须记录 spatial index，并在输出前 scatter/restore；否则模型根本无法生成完整图像。

用 attention scores 做重要性也不现实：当前 attention 用 PyTorch SDPA，默认不返回 attention map。要拿 score 必须改 attention 实现，牺牲 fused kernel，速度收益可能被吃光。用 token norm 则是廉价但粗糙的启发式，未必和生成质量相关。

还有 RoPE 和 in-context token 问题。模型在中间层插入 32 个 in-context class tokens，后续 `feat_rope_incontext` 假设固定 token 布局。动态删除 patch token 会让 RoPE 的空间位置、in-context offset、最后 `x[:, self.in_context_len:]` 的切片语义都变脆。

更根本的是：扩散/flow 采样是连续 ODE。某一步删掉空间自由度，下一步又恢复，等价于引入非平滑、非可逆的投影噪声。若不做 token merging 或 latent cache，只做 hard pruning，质量风险很大。

### 2. 新颖性

不新到四星。closest known work 很多：

- DynamicViT：动态 token sparsification。
- EViT：token reorganization / inattentive token fusion。
- ToMe：token merging，避免直接丢信息。
- DiffRate / diffusion Transformer token pruning 或 merging 方向：已经有人把 token reduction 用在 diffusion/DiT 加速上。

真正相对有价值的点只是 “pruning ratio 由 diffusion time 控制”。这是合理工程 twist，不是新范式。

### 3. 在 JiT 代码中的实现可行性

中等偏难，不是“中”：

- `JiTBlock.forward`、`Attention.forward`、RoPE、final layer、`unpatchify` 都要支持动态 token set。
- `torch.compile` 会被动态 shape 搞得频繁 recompile 或退化；当前训练默认 compile。
- batch 内每个样本 t 不同，token 数不同。要么 pad mask，要么按 t bucket 分组；否则 GPU 利用率很差。
- hard top-k 不可微，straight-through 不是免费午餐；如果只是固定 t schedule，可以不用学 importance，但仍要定义保留哪些空间位置。
- 若要保留 FID，应该先做 ToMe-style merging，而不是直接 pruning。

### 4. FID 改善估计

FID 改善不是这个方案的主卖点。现实估计：

- 原始 hard pruning：`-0.5 到 +0.05`，很可能伤质量。
- token merging + scatter restore + 只在高噪声早期启用：`0.0 到 +0.15` 有可能。
- 速度：Hermes 的 `0.6-0.8x` 也偏乐观。考虑 patch embed、MLP、final head、动态 gather/scatter 和 compile 退化，实际端到端可能只有 `1.05-1.25x` 加速，除非工程做得很细。

### 5. 最大风险

不可逆地丢掉空间信息，导致高噪声早期建立的全局结构偏差在 ODE 采样中被后续步骤放大。看起来省算，实际 FID 和视觉结构一起掉。

## F. MCP: Manifold Coordinate Prediction

### 1. 技术 soundness

这个方案把“数据流形低维”误用到了 patch 输出头上。

自然图像整体可能有低维结构，但 JiT 的 final head 是每个 patch token 输出 `16*16*3=768` 个像素。对每个 patch 独立预测一个 `d` 维坐标，并不能自动变成全图数据流形坐标。共享 `manifold_decoder(c)` 只是一个跨 patch 共享的低秩/字典式输出头，和真正的流形参数化不是一回事。

所谓坐标 `c` 没有 identifiability。任何可逆线性变换都能改变坐标而不改变输出；没有 chart consistency、邻域保持、Jacobian regularization 或全局 atlas，不能宣称学到了 manifold coordinate。

`residual_head` 小初始化也不能保证结构约束。训练几千步后 residual 可以接管所有输出，模型退化回原 JiT；如果强行限制 residual，则 patch 细节会欠拟合。

还有一个位置条件问题：如果 decoder 真的共享，位置相关的纹理和语义必须全部塞进 `c`。这不是 manifold learning，而是把 final linear layer 拆成两层 MLP。

### 2. 新颖性

明显过度包装。closest known work：

- Autoencoder/VAE/latent diffusion：低维 latent/code 再 decoder 回像素。
- Dictionary learning / low-rank output heads / bottleneck decoders：用共享基函数或低秩结构生成高维输出。
- Neural fields / coordinate networks 只是在“坐标 -> 信号”形式上相似，但 MCP 这里不是连续坐标场。
- Patch-wise tokenizer/decoder 结构在 ViT/MAE/生成模型中早就存在。

“显式预测流形坐标”这个说法没有数学约束支撑，所以 novelty 不能按这个 claim 计算。

### 3. 在 JiT 代码中的实现可行性

表面容易，实则容易做出无意义 ablation：

- 只需替换 `FinalLayer.linear` 为 `coord_head + decoder + residual_head`，所以代码改动不大。
- 但 checkpoint 不兼容，zero-init final head 的稳定 trick 需要重新设计。
- decoder 是否接收 timestep/class conditioning？若不接收，它表达力不足；若接收，又变成另一个 full output head。
- `d` 是强超参。小了伤细节，大了退化；论文很难讲清楚为什么选这个值。

### 4. FID 改善估计

Hermes 的 `0.1-0.5` 基本没有依据。我的估计：

- 小 `d` 且 residual 受限：FID 变差概率高。
- 大 `d` 或 residual 不受限：接近 baseline，但 novelty 消失。
- 合理上限：`0.0 到 +0.05`；更可能是负收益。

### 5. 最大风险

输出头瓶颈限制了 patch 高频细节，最后只能靠 residual 逃逸。一旦 residual 逃逸，方法主张失效；不逃逸，FID 受伤。

## G. CNLD: Cross-Noise-Level Distillation

### 1. 技术 soundness

这个想法可实现，但不深。对同一个 clean image，easy/hard 两个噪声视角的监督目标本来都是 `x`。当前 flow loss 已经直接把 hard view 拉向真值；再让 hard view 模仿 easy view，本质上是用一个 noisy teacher 替代真值的一部分。

在 `t_easy` 接近 1 时，`x_pred_easy` 约等于输入 clean image，所以 CNLD 近似额外的 `x_pred_hard -> x` 监督。但代码里 hard view 已经有这个监督，而且 velocity weighting 还会通过 `(1-t)` 改变梯度尺度。CNLD 的增益只能来自 regularization / curriculum，不是新的信息源。

更要命：当前仓库已经实现了 `SCMR`，形式就是两个 sampled views 的 `x_pred` 一致性，可选 stop-grad。CNLD 只是把两个 view 的 t 区间手工限定为 easy/hard。把已有代码功能重新命名为三星创新，已经偏宽容。

### 2. 新颖性

低。closest known work：

- Consistency Models / Consistency Training：不同噪声水平上的同一 ODE 轨迹端点一致。
- Progressive Distillation / Consistency Distillation / LCM / CTM：跨时间步或少步轨迹蒸馏。
- BYOL/self-distillation：同模型不同 view 的 stop-grad teacher-student。
- 本仓库的 SCMR：直接同构。

CNLD 的唯一差异是 easy/hard 分桶；这不是独立方法，更像 SCMR 的 sampling policy。

### 3. 在 JiT 代码中的实现可行性

六个里最容易：

- 在 `Denoiser.forward` 中固定采样 `t_easy in [0.7,1.0]`、`t_hard in [0.0,0.3]`。
- 复用现有 `scmr_lambda`、`scmr_stopgrad` 和 metrics。
- 代价是至少两次 net forward，训练成本约 `1.7-2.0x`，不是表中 `1.3x`。

但实现容易不等于值得做。它需要非常小的 lambda、warmup 和 per-t bin 监控，否则会把 hard region 学成 easy teacher 的偏差。

### 4. FID 改善估计

`0.1-0.3` 有可能但偏乐观，尤其是在已有 SCMR 的情况下。更现实：

- 作为 SCMR 分桶策略：`0.0 到 +0.15`。
- 权重大或 teacher 未收敛：负收益。
- 若没有完整 50k FID，只看小样本视觉效果，很容易自欺欺人。

### 5. 最大风险

teacher 不是 teacher，只是同一个未收敛网络在另一个 t 上的输出。错误会自我强化，尤其会把高噪声区域推向类平均或过平滑预测。

## H. SBMT: Schrödinger Bridge Manifold Transport

### 1. 技术 soundness

这是六个里最像“理论名词堆叠”的方案。Schrödinger Bridge 是严肃问题，但文中描述没有形成可训练算法。

SB 要定义两个端点分布、reference process、forward/backward drift 或 potentials，并通过 IPF/IMF 等方法匹配边缘分布。提案写 `dz_t = v_theta(z_t,t)dt + sigma dB_t`，然后说 `v_theta` 就是 JiT 的速度场，这不构成 SB。加一个 Brownian term 不会自动变成 Schrödinger Bridge。

“最优传输路径应该在流形上或接近流形”也不对。Brownian reference 在像素欧氏空间里会把样本推离低维图像流形；SB 是熵正则的分布桥，不是 manifold geodesic。若想让路径贴近流形，需要额外 state cost、constraint 或 learned reference，不是 SB 自带性质。

IPF 的 forward/backward 交替训练也不是一句话能塞进 JiT。当前 JiT 是单个 ODE denoiser；SBMT 至少需要两个方向模型、路径采样器、边缘重加权或 replay 机制。

### 2. 新颖性

“SB + diffusion/生成模型”早已不是空白。closest known work：

- I2SB: Image-to-Image Schrödinger Bridge。
- Diffusion Schrödinger Bridge Matching (DSBM)。
- Diffusion Bridge Mixture Transports。
- Generalized Schrödinger Bridge Matching。
- Score-based Schrödinger Bridge / IPF variants。

把 backbone 换成 JiT/Transformer 不足以构成五星创新。除非提出新的 SB objective 专门利用 x-prediction，否则只是“把现有 SB 套到另一个架构”。

### 3. 在 JiT 代码中的实现可行性

短期几乎不可行：

- `Denoiser.forward` 需要整体重写，不再是单步 `z=t*x+(1-t)*e` supervised regression。
- `sampling.py` 的 Euler/Heun ODE 要换成 SDE sampler，并处理 stochastic path。
- checkpoint/EMA/online eval 都要支持多模型或多 drift。
- IPF/IMF 要存路径样本或反复生成中间边缘，训练成本远超 `2-3x`；实际可能是 `5x+` 和大量工程风险。
- 当前单卡训练脚本不适合这种交替过程。

### 4. FID 改善估计

不能给 `?` 然后把它包装成高风险高回报。以当前代码和算力假设，我会给：

- 近期可复现实验：无可用 FID 改善，项目大概率无法跑到公平对比。
- 若完整实现 SB framework：结果未知，但和 baseline JiT 的 apples-to-apples 对比成本极高。

### 5. 最大风险

理论目标和实现目标脱节。最后会得到一个既不是标准 JiT、也不是正确 SB 的混合系统，无法解释、无法稳定训练、无法说服 reviewer。

## I. HAMP: Hallucination-Aware Manifold Projection

### 1. 技术 soundness

问题意识有一点真实：高噪声下 `x_pred` 会受 class prior 支配。但具体方案的 confidence 完全站不住。

`confidence = exp(-beta * ||x_pred - mu_class||^2)` 等价于认为“越接近类均值越可信”。这在 ImageNet 上基本反着来。类均值是模糊、低频、无实例细节的 prototype；高质量、多样化样本通常远离类均值。用它当可信度会惩罚多样性，鼓励均值化。

“低可信度时回溯并使用类条件均值”几乎必然伤 FID。FID 不只看 precision，也看 generated distribution 的 covariance。把难样本往均值吸会降低 recall/diversity，可能让图像更平滑但统计更差。

“低可信度施加更强 CFG”也没有方向性保证。CFG 本来就会放大条件/无条件差异；在不确定区域加大 CFG 往往放大伪影和过饱和，而不是减少 hallucination。

训练时让模型“不知道”更是未定义。当前监督目标是确定的 clean `x`；没有 uncertainty head、没有 likelihood calibration、没有 abstention label。只说“不要编造”不是 loss function。

### 2. 新颖性

中低。closest known work：

- Classifier guidance / classifier-free guidance：按条件信号调整采样方向。
- Dynamic thresholding / guidance rescale：抑制过强 guidance 的伪影。
- Diffusion uncertainty / ensemble disagreement / self-consistency filtering：用预测方差或跨视角一致性评估可靠性。
- Precision-recall tradeoff 和 mode collapse 文献：向 class mean 投影是经典的多样性损失风险。

HAMP 的“hallucination-aware”命名新，但 metric 和 intervention 都很粗糙。

### 3. 在 JiT 代码中的实现可行性

如果按文中做，代码不难但科学性差：

- 需要预计算每个 ImageNet class 的像素均值，注意当前训练图像归一化到 `[-1,1]`，且数据增强会改变均值。
- `sampling.py` 可以在 `cfg_forward` 或 stepper 后加入 confidence gate。
- 若要“回溯”，当前 Euler/Heun 没保存完整历史，需要改 sampler。
- 类均值 tensor 是 `1000 x 3 x 256 x 256`，存储不大，但实际语义很弱。

更合理的版本应该用跨噪声/多噪声 prediction disagreement、EMA vs online disagreement、或 MC dropout/augmentation variance，而不是 class mean 距离。

### 4. FID 改善估计

Hermes 的 `0.2-0.5` 明显 inflated。我的估计：

- class-mean projection：大概率 FID 变差，可能视觉上更“安全”但更无聊。
- confidence-gated step size：若只在极端异常值触发，可能 `0.0 到 +0.05`。
- 用 CNLD/SCMR disagreement 做 uncertainty，再调 step size：有机会 `+0.05 到 +0.15`，但这已经不是原 HAMP。

### 5. 最大风险

把真实多样性误判成 hallucination，导致 mode collapse/prototype collapse。FID、recall 和视觉多样性都会受伤。

## 六方案总排名

| 排名 | 方案 | 判断 | 理由 |
|---:|---|---|---|
| 1 | E: TCTP | 最可救，但要改成 merging/restore | 有明确工程收益；原 hard pruning 会伤质量 |
| 2 | G: CNLD | 低新颖但可快速验证 | 本质是 SCMR 分桶策略，容易做出干净 ablation |
| 3 | I: HAMP | 问题真实，方案错误 | class mean confidence 很差；可改成 disagreement-based uncertainty |
| 4 | F: MCP | 易实现但主张空 | 更像低秩输出头，不是 manifold coordinate |
| 5 | D: RML | 数学目标不闭合 | `L_adv` 无下界，manifold noise 没定义，采样分布 mismatch |
| 6 | H: SBMT | 不适合作为 JiT 改进提案 | 是另一个研究项目，不是当前代码的可落地方向 |

## 单个最强 idea

最强的是 **E: TCTP 的时间条件化算力分配**，但不是 Hermes 写的 hard pruning 版本。

正确版本应该是：

- 高噪声阶段做 token merging 或 pooled latent tokens，不直接丢 spatial token。
- 保留 source map，最后 scatter/restore 到完整 patch grid。
- 只在采样前半段或训练高噪声 t bucket 开启。
- 先把目标定义为速度/显存收益，不要先承诺 FID 提升。

它的价值在“JiT 的计算是否应该随 t 变化”，这个问题是清楚且可实验的。其他方案多数在概念定义上就不稳。

## 应该组合的 idea

1. **G + I：用跨噪声一致性误差做 uncertainty**

   不要用 class mean 距离。对同一样本采多个 `t` 或同一采样轨迹相邻点，计算 `x_pred` disagreement。高 disagreement 才表示不可靠。这样 HAMP 的 confidence 至少来自模型行为，而不是错误的 prototype 假设。

2. **E + G：剪枝/合并策略必须受一致性约束**

   如果 TCTP 改变 token set，就要求 full-token 和 reduced-token 的 `x_pred` 一致。这样可以直接测 token reduction 是否破坏 denoising trajectory。

3. **不要组合 D + H**

   RML 和 SBMT 都想改 forward process，但一个没有合法分布定义，一个需要完整 SB framework。把它们组合只会把不可识别性和工程复杂度叠加。

## 建议优先实验

### 实验 1：CNLD 是否只是 SCMR 的分桶版本

目的：快速杀死或保留 G。

设置：

- baseline：`scmr_lambda=0`。
- 当前 SCMR：随机两个 t，lambda sweep `{0.001, 0.003, 0.01}`，含 stop-grad/无 stop-grad。
- CNLD：`t_easy in [0.7,1.0]`，`t_hard in [0.0,0.3]`，同样 lambda sweep。

观测：

- per-t-bin `x_pred_mse` 和 `flow_loss`，代码已经有 bins。
- 小规模 FID 只能做趋势，最终必须 50k FID。
- 如果 CNLD 不显著优于 SCMR，G 直接降级为采样策略，不应作为新方法。

### 实验 2：TCTP 先做 non-destructive token merging，不做 hard pruning

目的：验证 E 的核心问题“高噪声是否需要完整 token 计算”，避免一开始就破坏输出网格。

设置：

- 在高噪声 `t < 0.3` 的中后层合并相邻 2x2 patch token 或 ToMe-style 相似 token。
- 保留 source map，final 前 scatter/average restore 到原 patch 数。
- 对比只在 inference 启用、train+inference 都启用两种。

观测：

- wall-clock speed、峰值显存、compile 是否 recompile。
- FID/IS，以及 high-noise bin 的 `x_pred_mse` 是否恶化。
- reduced-token 输出和 full-token 输出的一致性误差。

### 实验 3：HAMP 先做诊断，不做干预

目的：验证 “hallucination/confidence” 指标是否和真实错误相关。

设置：

- 记录高噪声区域的三类指标：class mean distance、跨 t `x_pred` disagreement、EMA vs online prediction disagreement。
- 与最终样本质量 proxy 相关：FID contribution、Inception confidence、CLIP/ImageNet classifier confidence、precision/recall 或人工抽样评分。

判定：

- 如果 class mean distance 和错误不相关或负相关，HAMP 原方案直接淘汰。
- 只有 disagreement 指标有效时，才进入 step-size gating 或 CFG rescale 实验。

## 参考的最近邻工作

- DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification, NeurIPS 2021.
- EViT: Expediting Vision Transformers via Token Reorganizations, ICLR 2022.
- ToMe: Token Merging: Your ViT But Faster, ICLR 2023.
- Consistency Models, ICML 2023: https://proceedings.mlr.press/v202/song23a.html
- Latent Consistency Models, arXiv 2023: https://arxiv.org/abs/2310.04378
- I2SB: Image-to-Image Schrödinger Bridge, ICML 2023: https://arxiv.org/abs/2302.05872
- Diffusion Schrödinger Bridge Matching, NeurIPS 2023: https://arxiv.org/abs/2303.16852
- Diffusion Bridge Mixture Transports, JMLR 2023: https://jmlr.org/papers/v24/23-0527.html
- Generalized Schrödinger Bridge Matching, ICLR 2024: https://arxiv.org/abs/2310.02233
- Denoising Diffusion GANs, ICLR 2022: https://arxiv.org/abs/2112.07804
