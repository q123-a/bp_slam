"""
使用预加载测量数据进行监督学习
================================

## 问题

measurement1500.mat 文件包含预先生成的物理仿真测量数据，但**没有真实标签**。
这意味着无法直接使用监督学习。

## 数据结构

### measurement1500.mat 包含：
- estimated_measurements_cell: (900, 2) 测量数据
  * 每个元素是 (3, M) 的数组
  * 第0行: 时延 (s)
  * 第1行: 噪声功率
  * 第2行: 信号幅度 (RSS)

### 缺少的信息：
- true_id: 每个测量对应的真实锚点 ID
- is_clutter: 是否是杂波

## 解决方案

### 方案 1: 使用 --add-clutter 生成标签（推荐）

当使用 `--add-clutter` 时，系统会：
1. 加载物理仿真的测量数据（干净数据）
2. 添加合成杂波
3. 生成标签

**命令示例**：
```bash
python testbed.py \
    --mode gnn \
    --steps 900 \
    --particles 100000 \
    --load-measurements measurement1500.mat \
    --add-clutter \
    --use-sparse-graph-v2
```

**数据流**：
```
1. 加载 measurement1500.mat
   ↓
   - 物理仿真的测量数据（干净）
   - 每个测量对应一个真实锚点

2. 添加合成杂波 (--add-clutter)
   ↓
   - 添加泊松分布的杂波
   - 添加漏检
   - **生成标签**:
     * 原始测量: true_id = 锚点ID, is_clutter = False
     * 杂波测量: true_id = -1, is_clutter = True

3. 传入 SLAM 算法
   ↓
   - 使用监督学习训练 GNN
```

**优点**：
- ✅ 简单，只需添加 `--add-clutter` 参数
- ✅ 可以控制杂波强度
- ✅ 自动生成标签

**缺点**：
- ❌ 杂波是合成的，不是物理仿真的

---

### 方案 2: 修改代码为预加载数据生成标签

如果你想为物理仿真的测量数据生成标签（不添加杂波），需要修改代码。

**核心思路**：
- 物理仿真的测量数据是"干净"的（没有杂波）
- 每个测量对应一个真实锚点
- 需要根据测量顺序推断 true_id

**实现步骤**：

#### 步骤 1: 修改 load_measurements_from_mat 函数

在 testbed.py 中修改 `load_measurements_from_mat` 函数，添加标签生成逻辑：

```python
def load_measurements_from_mat(mat_file='measurementbadf.mat', add_synthetic_clutter=False,
                               parameters=None, mismatch_mode=False, return_labels=False):
    """
    从 MAT 文件加载预先生成的检测数据

    新增参数:
        return_labels: bool, 是否生成监督学习标签
    """
    # ... 原有代码 ...

    # 如果需要返回标签
    if return_labels:
        labels = [[None for _ in range(num_sensors)] for _ in range(num_steps)]

        for step in range(num_steps):
            for sensor in range(num_sensors):
                meas = cluttered_measurements[step][sensor]
                if meas.size > 0:
                    M = meas.shape[1]
                    # 假设物理仿真的测量是按锚点顺序排列的
                    # true_id = [0, 1, 2, ..., M-1]
                    true_ids = np.arange(M, dtype=int)
                    is_clutter = np.zeros(M, dtype=bool)  # 全部是真实测量

                    labels[step][sensor] = {
                        'true_id': true_ids,
                        'is_clutter': is_clutter
                    }

        return cluttered_measurements, labels
    else:
        return cluttered_measurements
```

#### 步骤 2: 修改 main 函数

在 testbed.py 的 main 函数中，修改加载逻辑：

```python
if load_measurements is not None:
    if use_gnn:
        # GNN 模式：生成标签
        cluttered_measurements, ground_truth_labels = load_measurements_from_mat(
            load_measurements,
            add_synthetic_clutter=add_synthetic_clutter,
            parameters=parameters,
            mismatch_mode=mismatch_mode,
            return_labels=True  # 生成标签
        )
        print("✓ 已为预加载数据生成监督学习标签")
    else:
        # 纯 BP 模式：不需要标签
        cluttered_measurements = load_measurements_from_mat(
            load_measurements,
            add_synthetic_clutter=add_synthetic_clutter,
            parameters=parameters,
            mismatch_mode=mismatch_mode
        )
        ground_truth_labels = None
```

**优点**：
- ✅ 使用物理仿真的测量数据
- ✅ 可以进行监督学习

**缺点**：
- ❌ 需要假设测量顺序与锚点顺序一致
- ❌ 没有杂波，无法训练杂波检测能力

---

## 推荐方案

### 对于训练 GNN：

**推荐使用方案 1 (--add-clutter)**：
```bash
python testbed.py \
    --mode gnn \
    --steps 900 \
    --particles 100000 \
    --load-measurements measurement1500.mat \
    --add-clutter \
    --use-sparse-graph-v2
```

**原因**：
1. 简单，不需要修改代码
2. 可以训练杂波检测能力（质量头）
3. 可以控制杂波强度，测试鲁棒性

### 对于测试 GNN：

**可以不使用 --add-clutter**：
```bash
python testbed.py \
    --mode gnn \
    --steps 900 \
    --particles 100000 \
    --load-measurements measurement1500.mat \
    --use-sparse-graph-v2 \
    --inference-only \
    --load-checkpoint checkpoints/gnn_model.pth
```

**原因**：
1. 测试时不需要标签
2. 可以评估 GNN 在干净数据上的性能

---

## 总结

| 场景 | 命令 | 是否有标签 | 是否有杂波 |
|------|------|-----------|-----------|
| 生成新数据 (GNN) | `--mode gnn` | ✅ 自动生成 | ✅ 有 |
| 加载数据 (GNN) | `--load-measurements --add-clutter` | ✅ 自动生成 | ✅ 有 |
| 加载数据 (GNN, 无杂波) | `--load-measurements` | ❌ 无 | ❌ 无 |
| 加载数据 (纯BP) | `--load-measurements` | ❌ 不需要 | ❌ 无 |

**关键点**：
- measurement1500.mat 是物理仿真的干净数据（无杂波）
- 如果要使用监督学习，需要添加 `--add-clutter` 生成标签
- 如果只是测试（推理），不需要标签
"""
