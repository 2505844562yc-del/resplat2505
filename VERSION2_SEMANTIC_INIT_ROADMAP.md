# 第二版开发路线：Semantic-Conditioned Gaussian Initialization and Recurrent Refinement

## 1. 项目目标与边界

第二版研究的核心问题是：语义结构能否在初始高斯生成阶段减少物体边界处的深度混合、尺度跨界和错误覆盖，并与第一版的语义边界循环精炼形成互补。

第二版主模型暂定为：

```text
多视角 RGB + 相机参数
        ├── 原始 ReSplat/DepthSplat 初始特征与深度路径
        └── 离线语义结构输入（边界、置信度、边界距离）
                           ↓
                Semantic Geometry Adapter
                           ↓
               语义条件化的初始高斯
                           ↓
                    RGB / Depth 渲染
                           ↓
      第一版 Stage 4：Boundary Loss + Local Alignment Feedback
                           ↓
                  Recurrent Gaussian Refinement
                           ↓
                       最终高斯
```

第一阶段仍保持固定高斯数量，不增加语义类别预测，不修改 CUDA 渲染器，不在线训练 SAM2。只有语义初始化被多场景实验验证后，才考虑给高斯增加低维语义属性或进行固定预算的边界候选分配。

## 2. 可检验研究假设

### H1：语义初始化改善初始高斯

在循环精炼前（refine step 0），语义条件化初始化应改善至少一类结构指标，并且不明显损害 RGB：

- 初始渲染 Boundary F1 / Boundary L1；
- 边界带 PSNR / LPIPS；
- 可用时的深度边界误差；
- 初始高斯尺度跨越语义边界的比例。

### H2：语义初始化与第一版循环反馈互补

完整模型应满足：

```text
Semantic Init + Stage 4 Refinement
优于 Semantic Init Only
且优于 Stage 4 Only
```

如果只优于原始 ReSplat、但不优于 Stage 4，说明两者主要在修复同一种错误，最终论文不应强行同时保留。

### H3：收益不是由更多参数或更多高斯造成

第一主版本保持：

- 高斯数量相同；
- recurrent iteration 数相同；
- 输入视角数量相同；
- 训练步数相同；
- 大部分预训练主干冻结策略相同。

## 3. 推荐主架构

### 3.1 语义输入

第一阶段只使用已经存在的 SAM2 sidecar，构造轻量结构输入：

```text
S = [boundary, confidence, boundary_distance]
```

- `boundary`：类别无关语义边界；
- `confidence`：边界可靠度；
- `boundary_distance`：到最近可信边界的归一化距离或软边界带。

只能使用 context views 的语义输入。target view 的边界只能用于训练损失或评测，绝不能进入初始高斯生成器，防止测试视角信息泄漏。

第一阶段不直接输入高维 SAM/DINO 特征，避免显存、存储和语义噪声同时增加。高维语义特征属于验证后的增强实验。

### 3.2 Semantic Geometry Adapter

语义结构图先经过小型卷积编码器，并下采样到原始初始高斯特征的空间尺度：

```text
S -> small CNN -> F_sem
```

融合采用零初始化残差，而不是直接拼接后完全重写主干：

```text
F_fused = F_base + gate(F_base, F_sem) * ZeroProj(F_sem)
```

要求：

- `ZeroProj` 最后一层零初始化；
- 模块开启但未训练时输出严格等于官方初始高斯；
- 语义缺失或置信度为零时退化到原始路径；
- 门控逐空间 token 计算，不使用单个全局标量；
- 记录门控均值、分布和边界/非边界差异，避免门控塌缩。

### 3.3 语义条件化的初始几何残差

第一版实现不直接替换原始 Gaussian head。保留官方预测：

```text
G_base^0 = H_base(F_base)
```

新增小型、零初始化的几何残差头：

```text
[delta_depth_or_mean, delta_scale] = H_sem(F_fused, F_sem)
```

只修正最直接相关的属性：

- 三维位置或其上游深度参数；
- 三轴尺度。

暂时不修正：

- 旋转；
- 不透明度；
- SH 颜色。

最终初始值：

```text
mean_0  = mean_base  + boundary_gate * delta_mean
scale_0 = scale_base + boundary_gate * delta_scale
```

残差必须有界，并与原始参数单位匹配。优先修正上游深度/射线位置参数；如果原始代码结构不允许安全接入，再对 world-space mean 做相对尺度残差。

### 3.4 第一版循环精炼的继承

初始高斯生成后，默认继续使用第一版 Stage 4：

```text
Boundary Loss
+ confidence gating
+ Local Alignment [signed residual, dx, dy]
+ original recurrent updater
```

Stage 6–10 默认关闭，不能叠加到第二版主实验：

- multi-view scalar consensus；
- multi-view displacement；
- multiplicative parameter routing；
- selective output residual。

## 4. 必须实现的四种模型模式

所有模块必须有独立配置开关，至少支持：

| 模式 | Semantic Init | Stage 4 Refinement | 用途 |
|---|---:|---:|---|
| ReSplat baseline | 否 | 否 | 官方基线 |
| V1 only | 否 | 是 | 第一版主方法 |
| Init only | 是 | 否 | 验证语义初始化本身 |
| Init + V1 | 是 | 是 | 验证互补性，候选第二版主模型 |

禁止只比较 `ReSplat` 和 `Init + V1`，否则无法判断提升来自初始化还是第一版循环反馈。

## 5. 分阶段开发路线与晋级门槛

### 阶段 V2-0：冻结第一版与建立基准

目标：建立不可变化的第二版起点。

工作：

- 给当前 `boundary-resplat` 的 Stage 4 主配置建立版本标签；
- 保存正式配置、数据清单、测试索引和当前 commit；
- 确认 Stage 4 与官方 baseline 的可重复结果；
- 清点服务器磁盘、预训练权重和现有数据；
- 建立第二版独立分支。

通过条件：工作树干净，36 项现有测试通过，Stage 4 固定评测可复现。

### 阶段 V2-1：语义初始化数据契约

目标：只完成数据和张量路径，不改变模型输出。

工作：

- 从现有 sidecar 生成 boundary distance / soft band；
- 与 crop、resize、flip 保持严格同步；
- 明确 context/target 数据权限；
- 增加缺失语义的安全回退；
- 可视化 RGB、boundary、confidence、distance 对齐。

测试：形状、范围、增强同步、缺失文件、无边界图、target leakage。

### 阶段 V2-2：恒等 Semantic Geometry Adapter

目标：把语义特征接入初始高斯特征，但保持数值恒等。

工作：

- 小型 semantic encoder；
- 空间对齐与下采样；
- zero projection；
- learned spatial gate；
- 配置开关和诊断统计。

硬性测试：开关关闭、开关开启但零初始化时，初始高斯和渲染结果逐值一致或在规定浮点误差内一致。

### 阶段 V2-3：位置/尺度初始残差

目标：让语义只修改初始几何。

工作：

- 确定最安全的深度/mean 接入点；
- 分离 mean 与 scale 残差头；
- 逐轴、有界、零初始化修正；
- 边界空间门控；
- 记录边界与非边界高斯的修正幅度。

验证：单元测试、2-step 前向/反向、checkpoint 保存加载、显存峰值、梯度是否进入新模块。

### 阶段 V2-4：单场景短程筛选

按完全相同的种子和测试集比较四种模式：

```text
2 steps  -> 工程检查
20 steps -> 趋势筛选
50 steps -> 稳定性筛选
```

20-step 晋级要求：

- 初始边界指标有可测改善；
- 最终 RGB 三指标不得形成明显全面退化；
- 新分支不是全零、全开或全关；
- 非边界区域改变显著小于边界区域。

50-step 晋级要求：短期收益不反转，并且 `Init + V1` 至少显示互补趋势。

只允许一次基于诊断的设计修正，不进行无依据的超参数网格搜索。

### 阶段 V2-5：最小多训练场景验证

在与第一版一致的三个训练场景上独立训练，使用固定五场景测试，至少比较：

```text
V1 only
Init only
Init + V1
```

先用 200-step 健康检查。报告 15 个 train-model/test-scene 配对的均值、胜率和逐场景结果，不能只报告总体平均。

晋级要求：核心边界指标在多数配对中获胜，且 RGB/LPIPS 不出现系统性退化。

### 阶段 V2-6：第一轮正式训练

只有 V2-5 通过后才确定正式步数、场景数和随机种子。至少包含：

- 原始 ReSplat；
- V1 Stage 4；
- Semantic Init only；
- Semantic Init + V1；
- 去掉 confidence；
- 去掉 boundary distance；
- 只修正 mean；
- 只修正 scale。

正式报告均值、标准差、显存、训练时间、推理时间和参数量。

## 6. 评价体系

### 常规指标

- PSNR；
- SSIM；
- LPIPS；
- Boundary L1；
- Boundary F1。

### 必须新增的局部指标

- Boundary-band PSNR / LPIPS：只在语义边界附近固定宽度区域计算；
- Interior PSNR / LPIPS：检查边界收益是否以内部区域退化为代价；
- refine-step-0 指标：直接评价初始高斯，而不是只评价精炼后结果；
- refine gain：最终结果减去 step-0，判断初始化与精炼各自贡献；
- gate activation：边界和非边界区域分别统计；
- mean/scale correction magnitude：检查模块是否真正修改了目标属性。

如果数据没有可靠 GT depth，不把伪深度误差包装成真实几何指标；可使用跨视图重投影一致性作为辅助诊断，但不能替代真实深度评价。

## 7. 关键风险与预案

### 风险 A：语义只改善边界图，不改善 RGB

预案：分别检查 step-0 边界带 RGB、mean/scale 修正和最终 RGB。如果只有 Boundary F1 改善且 RGB持续下降，不把模块作为重建贡献。

### 风险 B：语义初始化与 Stage 4 冗余

预案：用四模式矩阵判断。若 `Init + V1` 不优于两者单独版本，最终只保留较强、较简单的一个。

### 风险 C：SAM2 边界造成测试依赖

预案：明确方法在推理时是否要求对 context images 运行 SAM2；统计 SAM2预处理成本。不得使用目标新视角语义作为输入。

### 风险 D：预训练主干被破坏

预案：恒等初始化、冻结主干、只训练 adapter 起步；验证后再局部解冻 Gaussian head 或深度后层。

### 风险 E：单卡显存与存储

预案：预计算低维语义图，不缓存高维 foundation features；batch size 1；保留每个晋级实验的最终 checkpoint，其余只保留日志和 JSON；先短跑后长跑。

## 8. 训练策略

### Phase A：Adapter-only

冻结原始 backbone、深度模块和大部分 Gaussian initializer，只训练：

- semantic encoder；
- fusion projection；
- spatial gate；
- mean/scale residual head。

目的：验证新增语义路径是否有信息，而不是让大模型用额外参数重新拟合训练场景。

### Phase B：局部解冻

若 Phase A 有稳定趋势，再解冻：

- 原 Gaussian parameter head；
- depth/geometry后层；
- 必要的 feature fusion后层。

不立即解冻完整 foundation backbone。

### Phase C：联合训练

只有多场景200-step通过后才进行更长联合训练，并使用相同资源预算比较所有主基线。

## 9. Git 管理规范

### 分支与标签

```text
boundary-resplat                 第一版历史与稳定代码
tag: v1-stage4-frozen            第一版正式主方法起点
v2-semantic-init                 第二版集成分支
```

大功能使用短期分支并合并回 `v2-semantic-init`：

```text
v2/data-contract
v2/semantic-adapter
v2/init-geometry-head
v2/evaluation
```

### 推荐提交粒度

```text
data: add semantic initialization inputs
test: verify semantic input alignment and leakage rules
model: add identity semantic geometry adapter
test: verify zero-init equivalence
model: add boundary-gated initial mean and scale residuals
exp: add v2 short-run ablation matrix
eval: add boundary-band and step-zero metrics
docs: record v2 stage results and promotion decision
```

每个提交只完成一个可解释目标，提交前运行相关测试和 `git diff --check`。

### 实验可复现性

每个实验目录保存：

- commit hash；
- 完整解析后的配置；
- 数据集与场景清单；
- 随机种子；
- checkpoint来源；
- GPU型号和显存峰值；
- 指标 JSON；
- 是否满足晋级门槛。

checkpoint、数据集、SAM2 sidecar和大体积可视化不得提交到 Git。代码提交后及时推送，切换服务器时只从远程分支恢复代码，数据和权重另行同步。

## 10. 第二版主线完成标准

只有同时满足以下条件，才能称为第二版主方法：

1. 语义初始化在 refine step 0 产生可重复的结构改善；
2. 改善在最终精炼结果中仍然存在；
3. `Init + V1` 的互补性通过四模式消融验证；
4. 三训练场景/五测试场景的多数配对获胜；
5. RGB质量没有系统性退化；
6. 增益不是由更多高斯或更多训练预算造成；
7. 单张 RTX 4090D 可以完成训练与推理；
8. 代码、配置、数据清单和实验记录可复现。

如果语义初始化未通过，则第一版 Stage 4 仍保持为独立成果，第二版探索不会破坏第一版代码和结论。

## 11. 验证后的增强路线（暂不开发）

### 11.1 低维语义高斯属性

把高斯扩展为：

```text
G_i = (mean, scale, rotation, opacity, SH, z_i)
```

其中 `z_i` 是低维 semantic embedding，可渲染语义特征图并参与后续精炼。只有当 `z_i` 反过来影响共享几何时，才构成语义引导重建，而不只是并行语义输出。

### 11.2 固定预算边界候选分配

在总高斯数固定的前提下，把更多、更小或前景/背景双深度候选分配给边界区域，从低价值内部区域回收预算。该阶段会改变初始化候选结构和张量形状，风险较高，必须建立在语义初始化已经有效的基础上。

### 11.3 高维 foundation feature

最后才考虑预计算 DINO/SAM低维投影特征，替代纯边界输入。需要单独评估磁盘、I/O、显存、跨场景泛化和是否真的优于轻量边界表示。

## 12. 推荐执行顺序

```text
V2-0 冻结第一版与建分支
  -> V2-1 数据契约
  -> V2-2 恒等语义适配器
  -> V2-3 初始 mean/scale 残差
  -> V2-4 2/20/50-step 四模式筛选
  -> V2-5 三训练场景 200-step
  -> V2-6 正式训练与论文消融
  -> 通过后再考虑 semantic z 或固定预算候选
```

当前下一项实际工作应从 V2-0 开始：冻结第一版 Stage 4 的确切配置和 commit，建立 `v2-semantic-init` 分支，并完成第二版的数据与代码接入审计。
