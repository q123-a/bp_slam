# BP 参数失配测试指南

## 问题背景

BP-SLAM 算法在数据关联时依赖两个关键参数：
- `detectionProbability`：检测概率（锚点被检测到的概率）
- `meanNumberOfClutter`：平均杂波数（每帧的平均误报数）

这些参数在 `bp_slam/core/association.py:141,154` 中被使用：
```python
input_bp[0, :] = (1 - detection_probability)  # 未检测概率
factor = ... * detection_probability / clutter_intensity  # 似然因子
```

**模型失衡问题**：当实际环境的参数与BP假设不同时，BP会产生错误的关联概率。

---

## 参数失配的影响

### 场景1：BP 低估杂波（实际杂波更多）
- **BP假设**：`meanNumberOfClutter = 1`
- **实际环境**：`meanNumberOfClutter = 5`
- **后果**：BP过于自信地关联测量，将杂波误认为真实锚点

### 场景2：BP 高估检测概率（实际漏检更多）
- **BP假设**：`detectionProbability = 0.95`
- **实际环境**：`detectionProbability = 0.85`
- **后果**：BP低估漏检概率，强行关联不可靠的测量

### 场景3：双重失配（最严重）
- **BP假设**：`meanNumberOfClutter = 1, detectionProbability = 0.95`
- **实际环境**：`meanNumberOfClutter = 5, detectionProbability = 0.85`
- **后果**：BP同时高估检测率和低估杂波率，导致大量错误关联

---

## 测试命令

### 1. 基准测试（无失配）
```bash
# BP 基准（实际参数 = BP假设）
python testbed.py --mode bp --steps 900 \
  --load-measurements measurementbadf.mat \
  --add-clutter

# GNN 基准
python testbed.py --mode gnn --steps 900 \
  --load-measurements measurementbadf.mat \
  --add-clutter
```

**说明**：
- 实际杂波数 = 1（泊松分布，均值1）
- 实际检测概率 = 0.95
- BP假设参数完全匹配，性能最优

---

### 2. 参数失配测试（测试BP鲁棒性）
```bash
# BP 失配模式（实际参数 ≠ BP假设）
python testbed.py --mode bp --steps 900 \
  --load-measurements measurementbadf.mat \
  --add-clutter --mismatch-mode

# GNN 失配模式（GNN不受影响，因为它不依赖这些参数）
python testbed.py --mode gnn --steps 900 \
  --load-measurements measurementbadf.mat \
  --add-clutter --mismatch-mode
```

**说明**：
- **实际环境**（由 `add_synthetic_clutter_to_measurements` 生成）：
  - 实际杂波数 = 3（BP假设的3倍）
  - 实际检测概率 = 0.85（比BP假设低10%）
- **BP假设**（在 `testbed.py:267,262` 中定义）：
  - BP假设杂波数 = 1
  - BP假设检测概率 = 0.95
- **预期结果**：
  - BP性能下降（因为参数失配）
  - GNN性能不受影响（因为GNN是数据驱动，不依赖这些参数）

---

### 3. 对比分析
```bash
# 运行完测试后，对比结果
python analyze_results.py
```

**预期输出**：
```
对比结果 (Comparison)
======================================================================

位置误差对比:
指标                 BP              GNN             改进
----------------------------------------------------------------------
平均误差 (Mean)      0.XXXXXX        0.YYYYYY        +ZZ.ZZ%
标准差 (Std Dev)     0.XXXXXX        0.YYYYYY        +ZZ.ZZ%
...
```

---

## 失配参数调整

如果您想测试不同程度的失配，可以修改 `testbed.py:110-115`：

```python
if mismatch_mode:
    # 实际杂波率是BP假设的3倍（可调整倍数）
    actual_mean_clutter = parameters['meanNumberOfClutter'] * 3  # 改为 2, 5, 10 等
    # 实际检测概率比BP假设低10%（可调整差值）
    actual_detection_prob = max(0.5, parameters['detectionProbability'] - 0.1)  # 改为 -0.2, -0.3 等
```

### 建议的测试矩阵

| 测试场景 | 杂波倍数 | 检测概率差 | 预期BP性能 |
|---------|---------|-----------|-----------|
| 轻度失配 | 2x      | -0.05     | 轻微下降   |
| 中度失配 | 3x      | -0.10     | 明显下降   |
| 重度失配 | 5x      | -0.20     | 严重下降   |
| 极端失配 | 10x     | -0.30     | 崩溃      |

---

## 为什么GNN不受影响？

GNN完全取代了 `bp_slam/core/association.py` 模块，它：
1. **不使用** `detectionProbability` 和 `clutterIntensity` 参数
2. **直接从数据学习**关联模式（通过几何+物理特征）
3. **使用匈牙利算法**进行全局最优匹配，自动处理杂波

因此，即使在参数失配环境下，GNN仍能保持稳定性能。

---

## 实验建议

### 实验1：验证BP失配敏感性
```bash
# 1. 无失配（基准）
python testbed.py --mode bp --steps 900 --load-measurements measurementbadf.mat --add-clutter
mv results/results_bp.npz results/results_bp_baseline.npz

# 2. 有失配
python testbed.py --mode bp --steps 900 --load-measurements measurementbadf.mat --add-clutter --mismatch-mode
mv results/results_bp.npz results/results_bp_mismatch.npz

# 3. 对比
python -c "
import numpy as np
baseline = np.load('results/results_bp_baseline.npz', allow_pickle=True)
mismatch = np.load('results/results_bp_mismatch.npz', allow_pickle=True)
print(f'基准误差: {baseline[\"mean_error\"]:.6f} m')
print(f'失配误差: {mismatch[\"mean_error\"]:.6f} m')
print(f'性能下降: {(mismatch[\"mean_error\"]/baseline[\"mean_error\"]-1)*100:.2f}%')
"
```

### 实验2：验证GNN鲁棒性
```bash
# 1. 无失配
python testbed.py --mode gnn --steps 900 --load-measurements measurementbadf.mat --add-clutter
mv results/results_gnn.npz results/results_gnn_baseline.npz

# 2. 有失配
python testbed.py --mode gnn --steps 900 --load-measurements measurementbadf.mat --add-clutter --mismatch-mode
mv results/results_gnn.npz results/results_gnn_mismatch.npz

# 3. 对比（预期：GNN性能几乎不变）
python -c "
import numpy as np
baseline = np.load('results/results_gnn_baseline.npz', allow_pickle=True)
mismatch = np.load('results/results_gnn_mismatch.npz', allow_pickle=True)
print(f'基准误差: {baseline[\"mean_error\"]:.6f} m')
print(f'失配误差: {mismatch[\"mean_error\"]:.6f} m')
print(f'性能变化: {(mismatch[\"mean_error\"]/baseline[\"mean_error\"]-1)*100:.2f}%')
"
```

---

## 总结

- **BP的弱点**：依赖准确的统计参数，参数失配会导致性能下降
- **GNN的优势**：数据驱动，不依赖统计参数，对参数失配鲁棒
- **测试方法**：使用 `--mismatch-mode` 标志模拟参数失配环境
- **调试方法**：修改 `testbed.py:110-115` 调整失配程度

这正是GNN相比传统BP的核心优势之一！
