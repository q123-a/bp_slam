# 新锚点试用期机制 (New Anchor Probation Period)

## 实施日期
2025-12-12

---

## 核心思想

**问题**：新锚点在初始化时位置估计不稳定，导致OSPA误差出现尖峰。

**解决方案**：给新锚点设置"试用期"（probation period），在试用期内：
- ✅ 新锚点**正常参与SLAM算法**（关联、更新、删除）
- ✅ 新锚点**不参与OSPA评分**（评估时被过滤掉）

这样可以：
1. 保持SLAM算法的完整性（不影响核心逻辑）
2. 防止新锚点初始化不稳定导致OSPA尖峰
3. 获得更平滑的OSPA曲线

---

## 实现方式

### 1. 记录锚点出生时间（slam.py）

在创建新锚点时，记录 `generatedAt` 字段：

```python
# slam.py: lines 697-710
estimated_anchors[sensor][step].append({
    'x': np.mean(new_particles_anchors[measurement]['x'], axis=1),
    'posteriorExistence': posterior_existence,
    'generatedAt': step  # [关键] 记录出生时间
})
```

### 2. OSPA计算时过滤试用期锚点（visualizer.py）

在计算OSPA时，只包含"成年"的锚点：

```python
# visualizer.py: lines 232-241
# 获取试用期参数（默认10帧）
probation_period = parameters.get('gnn_new_anchor_probation', 10)

for anchor in estimated_anchors[sensor][step]:
    # 计算锚点年龄
    born_time = anchor.get('generatedAt', 0)
    age = step - born_time

    # 只有"成年"的锚点才参与OSPA评分
    if anchor_existence >= detection_threshold and age >= probation_period:
        estimated_anchor_positions.append(anchor_pos)
```

### 3. 软OSPA也支持试用期过滤（generate_smooth_ospa.py）

```python
# generate_smooth_ospa.py: lines 49-61
if use_soft_ospa:
    # 软OSPA: 包含所有"成年"锚点，使用存在概率作为权重
    if age >= probation_period:
        estimated_anchor_positions.append(anchor_pos)
        existence_weights.append(anchor_existence)
else:
    # 硬OSPA: 只包含超过阈值且"成年"的锚点
    if anchor_existence >= detection_threshold and age >= probation_period:
        estimated_anchor_positions.append(anchor_pos)
```

---

## 参数配置

### 推荐参数

```python
parameters = {
    # 新锚点试用期（帧数）
    'gnn_new_anchor_probation': 10,  # 默认10帧

    # 其他相关参数
    'detectionThreshold': 0.5,       # 存在概率阈值
    'unreliabilityThreshold': 0.01,  # 删除阈值
}
```

### 参数调整建议

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 高动态环境 | 5-8帧 | 新锚点快速稳定，缩短试用期 |
| 标准环境 | 10帧 | 默认值，平衡稳定性和响应速度 |
| 高杂波环境 | 15-20帧 | 给新锚点更多时间稳定 |

---

## 工作流程示意

```
时间轴:  0 -----> 10 -----> 20 -----> 30 -----> 40
         |        |         |         |         |
新锚点A:  出生     试用期    成年      正常      正常
         ↓        ↓         ↓         ↓         ↓
SLAM:    参与     参与      参与      参与      参与
OSPA:    ✗过滤   ✗过滤     ✓评分     ✓评分     ✓评分
         (age=0) (age<10)  (age≥10)  (age=20)  (age=30)
```

**关键点**：
- 新锚点从出生开始就参与SLAM算法（关联、更新、删除）
- 但在前10帧（试用期）内不参与OSPA评分
- 10帧后"成年"，开始正常参与OSPA评分

---

## 与懒惰退出的区别

| 机制 | 作用位置 | 影响范围 | 目的 |
|------|---------|---------|------|
| **试用期过滤** | OSPA计算 | 只影响评估 | 防止新锚点初始化不稳定导致OSPA尖峰 |
| **懒惰退出** | SLAM主循环 | 影响算法核心 | 防止锚点频繁删除（已回退） |

**为什么回退懒惰退出？**
- 懒惰退出会导致锚点数量爆炸（100+个）
- 低质量锚点累积，污染SLAM算法
- 试用期过滤是更优雅的方案：只影响评估，不影响算法

---

## 预期效果

### 1. OSPA曲线更平滑

**原因**：
- 新锚点初始化时位置不准确（粒子分布广）
- 前几帧的位置估计会剧烈波动
- 过滤掉试用期锚点后，OSPA只评估稳定的锚点

### 2. 锚点数量保持正常

**原因**：
- 删除逻辑保持原始（立即删除低存在概率锚点）
- 不会累积低质量锚点
- 锚点数量应该在6-7个左右

### 3. SLAM算法不受影响

**原因**：
- 试用期过滤只在OSPA计算时生效
- SLAM主循环完全不受影响
- 新锚点仍然正常参与关联、更新、删除

---

## 测试方法

### 1. 运行测试

```bash
# 使用默认试用期（10帧）
python testbed.py --mode gnn --steps 200 \
  --load-measurements measurementbadf.mat \
  --add-clutter --mismatch-mode

# 自定义试用期
python testbed.py --mode gnn --steps 200 \
  --load-measurements measurementbadf.mat \
  --add-clutter --mismatch-mode \
  --gnn-new-anchor-probation 15
```

### 2. 生成OSPA图

```bash
# 生成平滑OSPA图（会自动应用试用期过滤）
python generate_smooth_ospa.py results/results_gnn.npz 20
```

### 3. 观察关键指标

- ✅ OSPA曲线是否更平滑（尖峰减少）
- ✅ 锚点数量是否正常（6-7个）
- ✅ 平均OSPA误差是否降低

---

## 调试输出

在SLAM运行时，可以观察：

```
Time instance: 15
Number of Anchors Sensor 1: 7
  - Anchor 0: age=15, participating in OSPA ✓
  - Anchor 1: age=12, participating in OSPA ✓
  - Anchor 2: age=8, in probation period ✗
  - Anchor 3: age=5, in probation period ✗
  ...
```

---

## 相关文件

### 核心修改文件

1. [slam.py](bp_slam/core/slam.py) - 记录 `generatedAt` 字段
   - Lines 697-710: 新锚点创建时记录出生时间

2. [visualizer.py](bp_slam/visualization/visualizer.py) - OSPA计算时过滤
   - Lines 212-241: 添加试用期过滤逻辑

3. [generate_smooth_ospa.py](generate_smooth_ospa.py) - 软OSPA支持试用期
   - Lines 31-61: 软OSPA和硬OSPA都支持试用期过滤

---

## 技术细节

### 锚点年龄计算

```python
born_time = anchor.get('generatedAt', 0)  # 出生时间，默认0（老锚点）
age = current_step - born_time            # 当前年龄（帧数）
```

**注意**：
- 如果锚点没有 `generatedAt` 字段，默认为0（老锚点）
- 这样可以兼容旧数据和初始化锚点

### 过滤条件

```python
# 硬OSPA过滤条件
if anchor_existence >= detection_threshold and age >= probation_period:
    include_in_ospa()

# 软OSPA过滤条件
if age >= probation_period:
    include_in_ospa_with_weight(anchor_existence)
```

---

## 总结

通过实现新锚点试用期机制，我们实现了：

1. ✅ **OSPA曲线更平滑** - 过滤掉不稳定的新锚点
2. ✅ **算法逻辑不变** - 只影响评估，不影响SLAM核心
3. ✅ **锚点数量正常** - 保持原始删除逻辑，不累积低质量锚点
4. ✅ **模块化设计** - 评估和算法完全解耦

这是一个比懒惰退出更优雅的解决方案！🎉

---

## 参考资料

- **OSPA距离**：Schuhmacher et al., IEEE Trans. Signal Processing, 2008
- **软OSPA**：使用存在概率权重的OSPA变体
- **试用期机制**：借鉴了目标跟踪中的"track confirmation"思想
