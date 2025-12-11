# GNN 失配模式优化说明

## 问题诊断

在参数失配模式下（`--mismatch-mode`），GNN性能反而比BP更差，原因如下：

### 1. **预热阶段学习错误模式**
- 前45步（5%）使用BP的关联结果训练GNN
- 失配模式下，BP因参数不匹配给出大量错误关联
- GNN在预热阶段就学到了错误的关联模式
- 后续即使用匈牙利算法自监督，也难以纠正

### 2. **熔断阈值不够严格**
- 原始阈值：`REJECTION_THRESHOLD = 3.0`
- 杂波增加3倍后，很多杂波的代价在2.5-3.5之间
- 导致误判杂波为正样本

### 3. **样本权重失衡加剧**
- 原始权重：正样本 5.0，负样本 1.0
- 失配模式下，负样本（杂波）数量激增（×3）
- 即使加权，网络仍然被大量杂波主导

---

## 优化方案

### 修改文件
1. [testbed.py](testbed.py:333-341) - 添加失配模式参数
2. [gnn_trainer_improved.py](bp_slam/core/gnn_trainer_improved.py:15-19) - 支持可配置参数
3. [slam.py](bp_slam/core/slam.py:84-86) - 传递新参数

### 核心改进

#### 1. **跳过预热阶段**（最关键）
```python
if mismatch_mode:
    parameters['gnn_warmup_steps'] = 0  # 跳过预热，避免学习BP的错误关联
```

**原理**：
- 失配模式下，BP的关联结果不可靠
- 直接从第1步开始使用匈牙利算法自监督
- 避免学习错误的关联模式

#### 2. **更严格的熔断阈值**
```python
if mismatch_mode:
    parameters['gnn_rejection_threshold'] = 2.0  # 从3.0降到2.0
else:
    parameters['gnn_rejection_threshold'] = 3.0  # 标准阈值
```

**原理**：
- 杂波增加3倍后，需要更严格的筛选
- 阈值2.0相当于2-sigma，只接受高置信度匹配
- 宁可漏检，也不误判杂波为正样本

#### 3. **更高的正样本权重**
```python
if mismatch_mode:
    parameters['gnn_positive_weight'] = 10.0  # 从5.0提升到10.0
else:
    parameters['gnn_positive_weight'] = 5.0  # 标准权重
```

**原理**：
- 杂波×3后，正负样本比例从1:2变为1:6
- 提升正样本权重到10.0，强制网络关注真实匹配
- 防止被大量杂波淹没

---

## 使用方法

### 测试命令

```bash
# 1. GNN 失配模式（优化后）
python testbed.py --mode gnn --steps 900 \
  --load-measurements measurementbadf.mat \
  --add-clutter --mismatch-mode

# 2. BP 失配模式（对比）
python testbed.py --mode bp --steps 900 \
  --load-measurements measurementbadf.mat \
  --add-clutter --mismatch-mode

# 3. 对比结果
python analyze_results.py
```

### 预期效果

**优化前**：
- GNN在失配模式下性能下降严重（比BP更差）
- 原因：预热阶段学习了BP的错误关联

**优化后**：
- GNN跳过预热，直接使用匈牙利算法自监督
- 更严格的熔断阈值（2.0）过滤杂波
- 更高的正样本权重（10.0）对抗样本失衡
- **预期**：GNN性能应该接近或优于BP

---

## 参数调优指南

如果优化后效果仍不理想，可以尝试调整以下参数：

### 1. 熔断阈值（[testbed.py:335](testbed.py:335)）
```python
parameters['gnn_rejection_threshold'] = 2.0  # 可调整范围: 1.5 - 3.0
```
- **降低**（1.5）：更严格，减少误判，但可能漏检
- **提高**（2.5）：更宽松，减少漏检，但可能误判杂波

### 2. 正样本权重（[testbed.py:336](testbed.py:336)）
```python
parameters['gnn_positive_weight'] = 10.0  # 可调整范围: 5.0 - 20.0
```
- **降低**（5.0）：减少对正样本的偏好，适合杂波较少的情况
- **提高**（15.0-20.0）：强化对正样本的关注，适合杂波极多的情况

### 3. 预热步数（[testbed.py:337](testbed.py:337)）
```python
parameters['gnn_warmup_steps'] = 0  # 可调整范围: 0 - 50
```
- **0**：完全跳过预热（推荐用于失配模式）
- **10-20**：少量预热，适合轻度失配
- **45**：标准预热，仅用于无失配模式

### 4. 失配程度（[testbed.py:110-112](testbed.py:110)）
```python
actual_mean_clutter = parameters['meanNumberOfClutter'] * 3  # 杂波倍数
actual_detection_prob = max(0.5, parameters['detectionProbability'] - 0.1)  # 检测率差
```

测试不同失配程度：
| 失配程度 | 杂波倍数 | 检测率差 | 阈值建议 | 权重建议 |
|---------|---------|---------|---------|---------|
| 轻度    | 2x      | -0.05   | 2.5     | 7.0     |
| 中度    | 3x      | -0.10   | 2.0     | 10.0    |
| 重度    | 5x      | -0.20   | 1.5     | 15.0    |
| 极端    | 10x     | -0.30   | 1.0     | 20.0    |

---

## 调试技巧

### 1. 查看样本统计
GNN每100步会打印样本统计信息：
```
[Step 100] 样本统计:
  正样本: 5/15 (33.3%)
  负样本: 10/15 (66.7%)
  失衡比例: 1:2.00
```

**健康指标**：
- 正样本比例 > 20%：正常
- 正样本比例 10-20%：轻度失衡，可能需要调整权重
- 正样本比例 < 10%：严重失衡，需要降低阈值或提高权重

### 2. 监控Loss曲线
```python
# 加载结果后查看Loss历史
import numpy as np
data = np.load('results/results_gnn.npz', allow_pickle=True)
# Loss历史保存在checkpoint中
```

**健康指标**：
- Loss稳定下降：训练正常
- Loss震荡：学习率过高或样本失衡严重
- Loss不变：阈值过严，几乎没有正样本

### 3. 对比不同配置
```bash
# 配置1：标准（阈值3.0，权重5.0）
python testbed.py --mode gnn --steps 900 --load-measurements measurementbadf.mat --add-clutter --mismatch-mode
mv results/results_gnn.npz results/results_gnn_config1.npz

# 配置2：严格（阈值2.0，权重10.0）
# 修改 testbed.py:335-336
python testbed.py --mode gnn --steps 900 --load-measurements measurementbadf.mat --add-clutter --mismatch-mode
mv results/results_gnn.npz results/results_gnn_config2.npz

# 对比
python -c "
import numpy as np
c1 = np.load('results/results_gnn_config1.npz', allow_pickle=True)
c2 = np.load('results/results_gnn_config2.npz', allow_pickle=True)
print(f'配置1误差: {c1[\"mean_error\"]:.6f} m')
print(f'配置2误差: {c2[\"mean_error\"]:.6f} m')
print(f'改进: {(c1[\"mean_error\"]-c2[\"mean_error\"])/c1[\"mean_error\"]*100:.2f}%')
"
```

---

## 理论依据

### 为什么跳过预热有效？

**传统机器学习**：需要大量标注数据预训练
**自监督学习**：从数据本身学习模式

匈牙利算法 + 物理约束 = 强自监督信号：
1. **几何约束**：距离残差（Mahalanobis距离）
2. **物理约束**：幅度一致性（RSS模型）
3. **全局最优**：匈牙利算法保证最优匹配

在失配模式下：
- BP的"伪标签"不可靠（因为参数错误）
- 物理约束仍然可靠（几何+幅度不会因参数失配而改变）
- 因此，直接用物理约束自监督比用BP预热更好

### 为什么更严格的阈值有效？

**精确率 vs 召回率权衡**：
- 阈值高（3.0）：召回率高，但精确率低（误判杂波）
- 阈值低（2.0）：精确率高，但召回率低（漏检真值）

在失配模式下：
- 杂波×3，误判代价更高
- 宁可漏检（召回率低），也不误判（精确率高）
- 因为误判会污染训练数据，导致错误累积

---

## 总结

**核心思想**：在参数失配环境下，物理约束比统计参数更可靠。

**三大优化**：
1. 跳过预热（避免学习错误模式）
2. 严格熔断（提高精确率）
3. 高权重（对抗样本失衡）

**预期效果**：GNN在失配模式下应该接近或优于BP，展现数据驱动方法的鲁棒性优势。
