# 双头GNN架构完整实现说明

## 架构概述

双头GNN架构将数据关联任务分解为两个独立的子任务：

1. **质量头（Quality Head）**：判断测量是否为真实信号（输出 0-1 分数）
2. **关联头（Association Head）**：判断测量属于哪个锚点（输出 M×K 关联概率）

### 核心优势

- **质量头不依赖预测位置**：只使用物理特征（RSS内在一致性）
- **即使BP预测错误，质量头仍能正确识别杂波**
- **两个头互不干扰，但共享底层特征提取**

---

## 实现文件

### 1. 模型架构：[gnn_model.py](bp_slam/core/gnn_model.py)

**新增类**：`JointDualHeadGNN` (lines 163-293)

**关键组件**：
```python
class JointDualHeadGNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2,
                 use_temporal_gru=False, use_layer_gru=False):
        # 共享特征提取主干
        self.input_encoder = nn.Sequential(...)
        self.gnn_layers = nn.ModuleList([...])

        # 质量头：M → M (每个测量一个质量分数)
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )

        # 关联头：M×K → M×K (每个测量对每个锚点的关联分数)
        self.association_head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, hybrid_input, hidden_state=None):
        # 返回: assoc_logits (B, M, K), quality_scores (B, M), new_hidden_state
```

**输入输出**：
- 输入：`hybrid_input` (1, M, K+1, 5) - 混合特征张量
- 输出：
  - `assoc_logits`: (1, M, K) - 关联logits（注意：没有垃圾桶列）
  - `quality_scores`: (1, M) - 质量分数 [0, 1]
  - `h_out`: (1, hidden_dim) - 新的隐藏状态

---

### 2. 训练器：[gnn_trainer_improved.py](bp_slam/core/gnn_trainer_improved.py)

**新增类**：`JointDualHeadTrainer` (lines 366-741)

#### 核心方法

##### 2.1 训练步骤
```python
def step(self, hybrid_tensor, measurements, predicted_measurements, predicted_variances, h_in=None):
    """
    执行一步训练

    返回:
        assoc_probs: (M, K) 关联概率
        quality_scores: (M,) 质量分数
        loss: 标量损失值
    """
```

##### 2.2 联合损失计算
```python
def _compute_joint_loss(self, assoc_logits, quality_scores, measurements,
                       predicted_measurements, predicted_variances):
    """
    双老师自监督：
    1. 物理老师：用RSS内在一致性监督质量头
    2. 几何老师：用匈牙利算法监督关联头

    总损失 = quality_weight × quality_loss + assoc_weight × assoc_loss
    """
```

##### 2.3 质量损失（物理老师）
```python
def _compute_quality_loss(self, quality_scores, measurements):
    """
    物理老师：用RSS内在一致性监督质量头

    原理：
    - 真实测量：RSS与距离符合路径损耗模型（误差 < 6 dB）
    - 杂波：RSS与距离不符合（误差 > 10 dB）

    路径损耗模型：
    RSS_theory = P_tx - 10 × n × log10(distance)
    其中 P_tx = 15.41 dBm, n = 2.0

    伪标签生成：
    - RSS误差 < 6 dB  → 标签 = 1 (好点)
    - RSS误差 > 10 dB → 标签 = 0 (坏点)
    - 6-10 dB         → 不参与训练（不确定）
    """
```

##### 2.4 关联损失（几何老师）
```python
def _compute_association_loss(self, assoc_logits, quality_scores, measurements,
                              predicted_measurements, predicted_variances):
    """
    几何老师：用匈牙利算法监督关联头

    关键：只对质量好的测量进行关联训练
    原理：如果把杂波强行拿去匹配，会教坏关联头

    步骤：
    1. 提取质量好的测量（quality > threshold）
    2. 计算几何代价矩阵（Mahalanobis距离）
    3. 匈牙利算法生成匹配标签
    4. 熔断：过滤掉代价太大的匹配（cost > threshold）
    5. 只对匹配成功的样本计算交叉熵损失
    """
```

---

### 3. SLAM集成：[slam.py](bp_slam/core/slam.py)

#### 3.1 导入双头训练器
```python
from .gnn_trainer_improved import GNNTrainerImproved, JointDualHeadTrainer
```

#### 3.2 初始化（lines 65-141）
```python
use_dual_head = parameters.get('gnn_use_dual_head', False)

if use_dual_head:
    # 双头架构
    gnn_trainer = JointDualHeadTrainer(
        device=device,
        lr=parameters.get('gnn_lr', 1e-3),
        hidden_dim=parameters.get('gnn_hidden_dim', 64),
        checkpoint_path=parameters.get('gnn_checkpoint_path', None),
        seed=42,
        use_ema=parameters.get('gnn_use_ema', True),
        ema_decay=parameters.get('gnn_ema_decay', 0.999),
        use_lr_scheduler=parameters.get('gnn_use_lr_scheduler', True),
        use_temporal_gru=parameters.get('gnn_use_temporal_gru', False),
        use_layer_gru=parameters.get('gnn_use_layer_gru', False),
        quality_threshold=parameters.get('gnn_quality_threshold', 0.5),
        assoc_threshold=parameters.get('gnn_assoc_threshold', 3.0),
        quality_weight=parameters.get('gnn_quality_weight', 1.0),
        assoc_weight=parameters.get('gnn_assoc_weight', 2.0)
    )
else:
    # 单头架构（原有实现）
    gnn_trainer = GNNTrainerImproved(...)
```

#### 3.3 推理逻辑（lines 342-436）

**双头架构推理**：
```python
if use_dual_head:
    # 1. 前向推理
    assoc_probs, quality_scores, loss = gnn_trainer.step(
        hybrid_tensor,
        filtered_measurements,
        predicted_measurements,
        predicted_uncertainties
    )
    # assoc_probs: (M_filtered, K) 关联概率
    # quality_scores: (M_filtered,) 质量分数 [0, 1]

    if step > warmup_steps:
        use_gnn_result = True

        # 2. 质量头输出 → 垃圾桶概率
        gnn_dustbin_filtered = 1.0 - quality_scores

        # 3. 关联头输出 → 关联概率
        gnn_probs = assoc_probs

        # 4. 映射回原始测量索引
        full_gnn_probs = np.zeros((num_measurements, num_anchors))
        full_gnn_dustbin = np.ones(num_measurements)
        valid_indices = np.where(valid_mask)[0]
        full_gnn_probs[valid_indices, :] = gnn_probs
        full_gnn_dustbin[valid_indices] = gnn_dustbin_filtered

        # 5. 转换为BP消息格式
        message_lhf_ratios = full_gnn_probs * (1.0 - full_gnn_dustbin[:, np.newaxis])

        # 6. 新锚点检测：quality_high + assoc_low → 新锚点
        max_assoc_prob = np.max(full_gnn_probs, axis=1)
        messages_new = (1.0 - full_gnn_dustbin) * (1.0 - max_assoc_prob)
```

**关键逻辑**：
- 质量分数 → 垃圾桶概率：`dustbin = 1 - quality`
- 新锚点检测：`new_anchor_prob = (1 - dustbin) × (1 - max_assoc)`
  - 高质量（质量分数高）
  - 低关联（不属于任何现有锚点）
  - → 可能是新锚点

---

### 4. 参数配置：[testbed.py](testbed.py)

#### 4.1 双头架构参数（lines 347-352）
```python
# [新增] 双头架构参数
parameters['gnn_use_dual_head'] = True  # 使用双头架构（质量头+关联头）
parameters['gnn_quality_threshold'] = 0.5  # 质量判断阈值
parameters['gnn_quality_weight'] = 1.0  # 质量损失权重
parameters['gnn_assoc_weight'] = 2.0  # 关联损失权重（更难学，权重更高）
parameters['gnn_assoc_threshold'] = 3.0  # 关联熔断阈值
```

#### 4.2 参数说明

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `gnn_use_dual_head` | True | 是否使用双头架构 |
| `gnn_quality_threshold` | 0.5 | 质量判断阈值（>0.5认为是好点） |
| `gnn_quality_weight` | 1.0 | 质量损失权重 |
| `gnn_assoc_weight` | 2.0 | 关联损失权重（关联更难学，权重更高） |
| `gnn_assoc_threshold` | 3.0 | 关联熔断阈值（Mahalanobis距离） |

---

## 测试方法

### 快速测试

使用提供的测试脚本：
```bash
./test_dual_head.sh
```

该脚本会：
1. 运行双头GNN在失配模式下的测试
2. 运行BP在失配模式下的测试（对比基准）
3. 自动对比分析结果

### 手动测试

#### 测试1：双头GNN + 失配模式
```bash
python testbed.py --mode gnn --steps 900 \
  --load-measurements measurementbadf.mat \
  --add-clutter --mismatch-mode
```

#### 测试2：BP + 失配模式（对比）
```bash
python testbed.py --mode bp --steps 900 \
  --load-measurements measurementbadf.mat \
  --add-clutter --mismatch-mode
```

#### 测试3：对比分析
```bash
python analyze_results.py
```

---

## 预期效果

### 失配模式下的性能对比

| 指标 | BP | 单头GNN | 双头GNN | 目标 |
|-----|----|---------|---------|----|
| 平均误差 | 0.25m | 0.30m ❌ | **0.20m** ✅ | < BP |
| 鲁棒性 | 差 | 差 | **强** | 不受参数失配影响 |

### 为什么双头架构更鲁棒？

1. **质量头独立于预测**
   - 单头GNN：依赖 `predicted_measurements`（失配时不准）
   - 双头GNN：质量头只看RSS内在一致性（不受预测影响）

2. **关联头只训练好点**
   - 单头GNN：杂波也参与训练（污染模型）
   - 双头GNN：只对质量好的测量训练关联（避免污染）

3. **物理约束优先**
   - 单头GNN：几何+物理混合（几何不准时整体失效）
   - 双头GNN：物理优先筛选，再做几何关联（分层鲁棒）

---

## 调试技巧

### 1. 监控质量头性能

在训练过程中，观察质量头的判断准确率：
```python
# 在 _compute_quality_loss 中添加
if hasattr(self, 'step_count') and self.step_count % 100 == 0:
    good_count = good_mask.sum().item()
    bad_count = bad_mask.sum().item()
    print(f"[Step {self.step_count}] 质量判断:")
    print(f"  好点: {good_count}/{M} ({good_count/M*100:.1f}%)")
    print(f"  坏点: {bad_count}/{M} ({bad_count/M*100:.1f}%)")
```

### 2. 监控关联头性能

观察有多少测量参与关联训练：
```python
# 在 _compute_association_loss 中添加
if hasattr(self, 'step_count') and self.step_count % 100 == 0:
    print(f"[Step {self.step_count}] 关联训练:")
    print(f"  质量好的测量: {len(valid_indices)}/{M}")
    print(f"  匹配成功: {match_mask.sum().item()}/{len(valid_indices)}")
```

### 3. 对比不同配置

测试不同的质量阈值：
```bash
# 配置1: 宽松阈值（0.3）
# 修改 testbed.py:349
parameters['gnn_quality_threshold'] = 0.3
python testbed.py --mode gnn --steps 900 --load-measurements measurementbadf.mat --add-clutter --mismatch-mode
mv results/results_gnn.npz results/results_gnn_threshold_0.3.npz

# 配置2: 标准阈值（0.5）
parameters['gnn_quality_threshold'] = 0.5
python testbed.py --mode gnn --steps 900 --load-measurements measurementbadf.mat --add-clutter --mismatch-mode
mv results/results_gnn.npz results/results_gnn_threshold_0.5.npz

# 配置3: 严格阈值（0.7）
parameters['gnn_quality_threshold'] = 0.7
python testbed.py --mode gnn --steps 900 --load-measurements measurementbadf.mat --add-clutter --mismatch-mode
mv results/results_gnn.npz results/results_gnn_threshold_0.7.npz
```

---

## 参数调优指南

### 1. 质量阈值（quality_threshold）

| 阈值 | 效果 | 适用场景 |
|-----|------|---------|
| 0.3 | 宽松，更多测量参与关联训练 | 杂波较少（<2倍） |
| 0.5 | 标准，平衡精确率和召回率 | 杂波中等（2-3倍） |
| 0.7 | 严格，只有高质量测量参与 | 杂波很多（>3倍） |

### 2. 损失权重（quality_weight vs assoc_weight）

| 配置 | quality_weight | assoc_weight | 效果 |
|-----|---------------|--------------|------|
| 质量优先 | 2.0 | 1.0 | 强化杂波识别 |
| 平衡 | 1.0 | 1.0 | 均衡训练 |
| 关联优先 | 1.0 | 2.0 | 强化关联精度（推荐） |

**推荐**：`quality_weight=1.0, assoc_weight=2.0`
- 原因：关联任务更难学，需要更高权重
- 质量任务相对简单（RSS一致性是强信号）

### 3. 关联熔断阈值（assoc_threshold）

| 阈值 | 效果 | 适用场景 |
|-----|------|---------|
| 2.0 | 严格，只接受高置信度匹配 | 预测非常不准 |
| 3.0 | 标准，2-3σ范围内接受 | 预测中等准确 |
| 4.0 | 宽松，接受更多匹配 | 预测较准确 |

---

## 理论依据

### 信息论视角

**互信息分解**：
```
I(Measurement; Anchor) = I_quality + I_association

其中:
I_quality = 物理信息（RSS内在一致性）
I_association = 几何信息（空间位置匹配）
```

失配模式下：
- `I_association` 受损（预测不准）
- `I_quality` 保持（直接测量）
- 双头架构先用 `I_quality` 筛选，再用 `I_association` 关联

### 贝叶斯视角

**后验概率分解**：
```
P(anchor|meas) ∝ P(meas|anchor) × P(anchor)

双头分解:
P(meas|anchor) = P_quality(meas) × P_assoc(anchor|meas, quality=good)
```

优势：
- `P_quality` 不依赖锚点预测
- `P_assoc` 只在质量好的测量上计算
- 避免错误预测污染整个后验

---

## 总结

### 核心创新

1. **任务分解**：质量判断 + 关联决策
2. **双老师自监督**：物理老师（RSS）+ 几何老师（匈牙利）
3. **分层鲁棒**：物理优先筛选，几何精细关联

### 三大优势

1. **不依赖预测**：质量头只看物理特征
2. **避免污染**：关联头只训练好点
3. **自然新锚点检测**：quality_high + assoc_low

### 预期效果

在参数失配环境下，双头GNN应该展现出比BP和单头GNN更强的鲁棒性，因为：
- 物理约束（RSS）不受参数失配影响
- 质量头提供可靠的杂波过滤
- 关联头在干净数据上训练，避免错误累积

**这正是数据驱动方法相比模型驱动方法的核心优势！**
