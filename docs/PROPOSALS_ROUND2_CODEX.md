# JiT 改进方向创意提案 Round 2

## 1. 自适应频带去噪路由器

**一句话总结：** 让 JiT 在不同时间步自动把低频结构、中频布局和高频纹理交给不同容量的专家路径处理，而不是用同一套 ViT 块均匀 denoise 所有频带。

**核心直觉：**  
扩散或 flow matching 的去噪过程并不是频谱均匀的：早期更像恢复全局能量和轮廓，后期更像补纹理、边缘和颜色细节。JiT 的大 patch ViT 很适合全局建模，但可能把高频细节压进同一个 token 流里，导致容量使用不经济。若显式引入频带分解和时间步相关路由，可以让模型在 x-prediction 下更像真正的多尺度信号恢复器。

**具体实现草图：**
- 在输入图像和 noisy sample 上做轻量可逆频带分解，例如 Haar wavelet、learned Laplacian pyramid 或 DCT block 分解。
- 每个 patch token 附带频带标识：low / mid / high，或把同一空间 patch 拆成多个频带子 token。
- ViT 主干中加入时间步条件化的 MoE 路由：低 t 偏向 high-frequency expert，高 t 偏向 low-frequency expert，中间共享布局 expert。
- velocity MSE 保持不变，但额外加入频带重加权损失：早期增强低频 x 预测，后期增强高频残差预测。
- 推理时可以根据 t 动态关闭部分高频专家，获得可调速度和质量。

**为什么 JiT 作者可能没有做：**  
JiT 的贡献重点是证明“简单 x-prediction + 大 patch ViT + 无 tokenizer”已经足够强。频带路由会显著增加系统复杂度，并且破坏“back to basics”的叙事；同时 MoE 和 wavelet token 的工程调参成本较高，不利于首篇主方法展示。

**新颖性评估：** ★★★★☆

**预期影响与最大风险：**  
预期可提升纹理锐度、边缘一致性和计算效率，尤其适合高分辨率生成。最大风险是频带拆分带来的归纳偏置过强，导致模型在复杂语义纹理上出现 ringing artifact 或频带间不一致。

## 2. 曲率感知时间采样与局部路径重参数化

**一句话总结：** 不再固定使用 logit-normal 时间采样，而是在线估计生成路径的局部曲率，把训练预算集中到 flow 轨迹最弯、最难学的时间区域。

**核心直觉：**  
flow matching 的速度场难度不只由 t 决定，还取决于数据分布、模型阶段和当前样本。固定的 logit-normal 采样是全局先验，但真实训练中某些 t 区域的 x-prediction 误差、Jacobian 变化和速度方向变化会更剧烈。若用曲率来动态调整时间采样，模型会把容量花在路径几何最复杂的位置。

**具体实现草图：**
- 对同一 batch 样本采样相邻时间点 `t-delta, t, t+delta`，用模型预测的 x 或 velocity 估计二阶差分曲率。
- 维护一个 EMA 时间难度直方图，指标可包含 velocity 误差、预测方向夹角、二阶差分范数。
- 将原 logit-normal 分布与 learned histogram 混合：`p(t)=alpha*p_logitnormal(t)+(1-alpha)*p_curvature(t)`。
- 在高曲率区间引入局部时间重参数化，让 ODE solver 在这些区间使用更密集步长。
- 为避免训练分布漂移，前 20% steps 使用原始采样，之后逐步提高 adaptive 采样权重。

**为什么 JiT 作者可能没有做：**  
固定 logit-normal 足够稳定，且容易复现。在线时间采样会引入额外状态、分布反馈环和可复现性问题；如果主论文目标是简化训练 recipe，动态采样反而会使实验解释变难。

**新颖性评估：** ★★★★☆

**预期影响与最大风险：**  
预期能减少同等 FID 下的训练 token 数，并提高少步采样质量。最大风险是自适应采样被模型早期错误误导，过度关注噪声很大的时间段，导致覆盖不足或训练不稳定。

## 3. 自蒸馏式双视角 x-prediction

**一句话总结：** 同一 noisy sample 同时训练“全局粗视角”和“局部细视角”两个预测头，并用互相一致性约束让 JiT 学会跨尺度自校正。

**核心直觉：**  
JiT 的大 patch 设计提升效率和全局建模，但大 patch 天然牺牲局部像素几何。可以不引入外部 tokenizer 或预训练，而是在网络内部构造两个视角：低分辨率全局 x 预测负责语义结构，高分辨率局部残差预测负责细节。二者共享主干，通过自蒸馏保持一致，相当于让模型在训练时拥有一个内生的 coarse-to-fine teacher。

**具体实现草图：**
- BottleneckPatchEmbed 输出后保留两个分支：主 ViT token 流和一个轻量局部卷积 / window attention residual 流。
- 主头预测完整 x；局部头预测 `x - upsample(downsample(x))` 或 patch 内高频残差。
- 对主头输出做 downsample，与真实低分辨率 x 做 MSE；对局部头做 residual MSE。
- 增加一致性损失：`main_x ≈ coarse_x + local_residual`，且两个分支在随机 crop / resize augment 下保持等变。
- 推理时可只使用主头，或在最后若干步启用局部头作为 refinement。

**为什么 JiT 作者可能没有做：**  
JiT 的路线强调纯 ViT 和单一 x-prediction 目标。双头、多尺度残差和自蒸馏会让架构看起来更接近 hybrid diffusion U-Net，不利于突出“ViT-only generative backbone”的简洁性。

**新颖性评估：** ★★★☆☆

**预期影响与最大风险：**  
预期改善大 patch 下的局部细节、文字边缘和小物体形状，并可能降低对 patch size 的敏感性。最大风险是两个头互相妥协，主头学习变弱，局部头产生过锐或不一致的纹理补偿。

## 4. 语义噪声过程：从像素高斯流到对象保持流

**一句话总结：** 把 JiT 的噪声过程从纯像素插值扩展为“对象身份慢变、纹理快变”的结构化流，让 denoising 过程更符合图像语义层级。

**核心直觉：**  
标准 flow matching 通常把图像看成连续像素向量，但自然图像的语义不是均匀退化的。对象存在性、姿态、材质、背景纹理在去噪中的恢复节奏不同。如果构造一个保留对象级低维结构更久、较早破坏局部纹理的噪声过程，x-prediction 可能更容易学习稳定语义。

**具体实现草图：**
- 不使用外部 tokenizer 或 pretrained encoder，而是从图像自身构造弱对象代理：颜色连通区域、边缘超像素、随机矩形区域或 learned slot tokens。
- 噪声注入分成两部分：区域均值 / 低频 slot 表示使用较低噪声强度，区域内残差使用较高噪声强度。
- 时间步条件中加入区域噪声日程参数，让模型知道当前是对象级恢复还是纹理级恢复。
- loss 分为对象均值 x-prediction、区域 mask 边界一致性、像素残差 velocity MSE。
- 可用随机区域划分 ensemble，避免模型过拟合某一种 segmentation。

**为什么 JiT 作者可能没有做：**  
这会挑战 JiT 最朴素的“无 tokenizer、无预训练、无复杂先验”设定。即使对象代理不依赖外部模型，也会引入很多关于区域划分和噪声 schedule 的设计选择，论文实验空间会急剧扩大。

**新颖性评估：** ★★★★★

**预期影响与最大风险：**  
预期增强对象完整性、减少肢体和边界断裂，并改善复杂场景的组合一致性。最大风险是弱对象代理质量不足，错误区域会把不相关像素绑定在一起，反而制造结构伪影。

## 5. In-Context Class Tokens 到 In-Context Generation Programs

**一句话总结：** 把 JiT 的 in-context class tokens 扩展成可组合的生成程序 token，让类别、风格、布局、负约束和采样策略都通过上下文序列表达。

**核心直觉：**  
JiT 已经用 in-context class tokens 证明条件可以作为 token 注入，而不是依赖传统分类嵌入。这个方向可以进一步把“条件”泛化成一个小型程序：多个 token 表达对象、关系、风格、局部区域、禁止项和求解偏好。这样模型不需要文本预训练，也能获得比单 class label 更丰富的可控生成接口。

**具体实现草图：**
- 设计若干无语言依赖的 program token 类型：class、count、region box、style prototype、negative class、composition relation。
- 训练时从 ImageNet 标签和图像增强自动生成 program：例如 class token + random crop box + color style token + negative sampled class。
- 对 patch tokens 加 cross-attention 或 prefix attention，使 program tokens 在所有层作为上下文参与。
- 加入 program dropout 和 program permutation，使模型学会组合而不是死记 token 顺序。
- 推理时用户可以拼接多个 class / region / style token，形成无需文本编码器的可控生成。

**为什么 JiT 作者可能没有做：**  
原始 JiT 可能更关注 class-conditional 生成基准，复杂 program token 很难在标准 ImageNet FID 中体现价值。它还需要定义新的评估指标，例如区域可控性、负约束成功率和组合泛化能力。

**新颖性评估：** ★★★★☆

**预期影响与最大风险：**  
预期把 JiT 从分类条件生成推向结构化可控生成，并保持无文本预训练的纯净设定。最大风险是自动构造的 program 信号太弱，模型只利用 class token，忽略更复杂的控制 token。

## 6. 一致性校正的少步 JiT Solver

**一句话总结：** 在训练阶段显式约束不同 ODE 步长下的 x-prediction 终点一致，使 JiT 原生支持 4-8 步高质量采样。

**核心直觉：**  
JiT 的 x-prediction 很适合被解释为“当前 noisy state 指向 clean x 的投影”。如果不同时间步的 clean projection 足够一致，采样 solver 可以更大胆地跨步。与其训练完再蒸馏少步模型，不如在主训练中直接让模型学会跨时间自一致。

**具体实现草图：**
- 对同一样本采样 `t_high > t_mid > t_low`。
- 从 `z_t_high` 用模型预测一步 Euler 到 `z_t_mid`，再预测 clean x；同时直接在真实 `z_t_mid` 上预测 clean x。
- 加入 consistency loss：两条路径的 `x_pred` 和 implied velocity 要一致。
- 随训练推进逐渐扩大时间跨度，模拟少步 solver 的大步长误差。
- 推理时使用固定 4、6、8 步 schedule，并用最后一步 x-prediction 做轻量校正。

**为什么 JiT 作者可能没有做：**  
少步一致性训练会增加显存和训练时间，也可能干扰主 FID 训练目标。若论文重点是展示 JiT 架构和 x-prediction 的基础能力，作者可能选择把采样加速留给后续蒸馏或 solver 研究。

**新颖性评估：** ★★★☆☆

**预期影响与最大风险：**  
预期显著降低推理步数，提升实际部署价值。最大风险是 consistency loss 过强会让模型学到过平滑的平均 clean projection，牺牲多样性和细节。
