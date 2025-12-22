# GNN 版本切换使用指南

本指南介绍如何在 `testbed.py` 中切换使用不同版本的 GNN 模型。

## 可用的 GNN 版本

### V1 - 原始版本 (Hard Max)
- **架构**: 隐式因子图
- **聚合方式**: Hard Max（梯度只流向最大值）
- **特点**:
  - 最简单的实现
  - 梯度稀疏，只有最大值节点接收梯度
  - 适合快速原型验证

### V2 - Softmax 聚合版本
- **架构**: 隐式因子图
- **聚合方式**: Softmax（smooth max，梯度流向所有节点）
- **特点**:
  - 更平滑的梯度流动
  - 边特征增强（支持 `concat` 和 `diff` 模式）
  - 可配置的温度参数 `gamma`
  - 更好的训练稳定性
  - **推荐用于大多数场景**

### V3 - 显式因子节点版本
- **架构**: 显式因子图（两阶段消息传递）
- **聚合方式**: Softmax（可配置）
- **特点**:
  - 真正的因子图结构：测量节点 → 因子节点 → 锚点节点
  - 两阶段消息传递：
    - Stage 1: 变量节点 → 因子节点
    - Stage 2: 因子节点 → 变量节点
  - 更强的表达能力
  - 更符合 Belief Propagation 数学原理
  - 参数量比 V2 增加约 50%
  - **适合复杂关联场景**

## 使用方法

### 基本命令格式

```bash
python testbed.py --mode gnn --gnn-version <版本>
```

### 示例 1: 使用 V1 版本（Hard Max）

```bash
python testbed.py --mode gnn --gnn-version v1 --steps 100
```

### 示例 2: 使用 V2 版本（Softmax，推荐）

```bash
python testbed.py --mode gnn --gnn-version v2 --steps 100
```

### 示例 3: 使用 V3 版本（显式因子节点）

```bash
python testbed.py --mode gnn --gnn-version v3 --steps 100
```

### 示例 4: 完整参数示例

```bash
python testbed.py \
    --mode gnn \
    --gnn-version v3 \
    --steps 900 \
    --particles 100000 \
    --warmup 50 \
    --load-measurements measurementbadf.mat \
    --add-clutter
```

## 版本对比

| 特性 | V1 | V2 | V3 |
|------|----|----|-----|
| 聚合方式 | Hard Max | Softmax | Softmax |
| 因子节点 | 隐式 | 隐式 | **显式** |
| 梯度流动 | 稀疏 | 平滑 | 平滑 |
| 参数量 | ~100K | ~100K | ~150K |
| 表达能力 | 基础 | 增强 | **最强** |
| 训练稳定性 | 一般 | 好 | 好 |
| 推荐场景 | 原型验证 | **通用场景** | 复杂关联 |

## 高级配置

### V2/V3 特有参数

在 `testbed.py` 中，V2 和 V3 版本支持以下额外参数：

```python
parameters['gnn_aggregation'] = 'softmax'  # 聚合方式: 'softmax', 'max', 'mean'
parameters['gnn_gamma'] = 3.0              # Softmax 温度参数
parameters['gnn_edge_mode'] = 'concat'     # 边特征模式: 'diff', 'concat'
```

#### 聚合方式说明

- **`softmax`** (推荐): 平滑最大值，梯度流向所有节点
  - `gamma=3.0`: 平衡模式
  - `gamma→∞`: 退化为 Hard Max
  - `gamma→0`: 退化为 Mean

- **`max`**: Hard Max，梯度只流向最大值

- **`mean`**: 平均池化，全局平均

#### 边特征模式说明

- **`concat`** (推荐): 拼接 `[h, h - agg(h)]`
  - 保留绝对值和相对值信息
  - 更强的表达能力

- **`diff`**: 只用差分 `h - agg(h)`
  - 更简洁，参数量更少

## 性能建议

### 快速测试（≤100 步）
```bash
python testbed.py --mode gnn --gnn-version v2 --steps 100 --particles 50000
```

### 中等测试（≤300 步）
```bash
python testbed.py --mode gnn --gnn-version v2 --steps 300 --particles 100000
```

### 完整测试（900 步）
```bash
python testbed.py --mode gnn --gnn-version v3 --steps 900 --particles 100000
```

## 权重管理

### 保存权重
```bash
python testbed.py --mode gnn --gnn-version v2 --steps 100
# 权重自动保存到 checkpoints/gnn_model.pth
```

### 加载权重继续训练
```bash
python testbed.py --mode gnn --gnn-version v2 --load-checkpoint checkpoints/gnn_model.pth
```

### 仅推理模式（不训练）
```bash
python testbed.py --mode gnn --gnn-version v2 --load-checkpoint checkpoints/gnn_model.pth --inference-only
```

### 不保存权重
```bash
python testbed.py --mode gnn --gnn-version v2 --no-save-checkpoint
```

## 注意事项

1. **版本兼容性**: 不同版本的权重文件不兼容，切换版本时需要重新训练

2. **默认版本**: 如果不指定 `--gnn-version`，默认使用 V2 版本

3. **参数量**: V3 版本参数量比 V1/V2 多约 50%，训练时间会稍长

4. **推荐配置**:
   - 一般场景: V2 + Softmax + concat
   - 复杂场景: V3 + Softmax + concat
   - 快速验证: V1 + Hard Max

## 故障排查

### 问题 1: 导入错误
```
ImportError: cannot import name 'JointDualHeadTrainerV2'
```

**解决方案**: 确保以下文件存在：
- `bp_slam/core/gnn_trainer_improved.py` (V1)
- `bp_slam/core/gnn_trainer_v2.py` (V2)
- `bp_slam/core/gnn_trainer_v3.py` (V3)

### 问题 2: 版本不匹配
```
ValueError: 未知的GNN版本: vX
```

**解决方案**: 使用正确的版本标识：`v1`, `v2`, 或 `v3`

### 问题 3: CUDA 内存不足
```
RuntimeError: CUDA out of memory
```

**解决方案**:
- 减少粒子数: `--particles 50000`
- 减少步数: `--steps 100`
- 使用 CPU: 代码会自动检测并使用 CPU

## 示例脚本

创建一个批量测试脚本 `test_all_versions.sh`:

```bash
#!/bin/bash

# 测试所有 GNN 版本
for version in v1 v2 v3; do
    echo "Testing GNN $version..."
    python testbed.py \
        --mode gnn \
        --gnn-version $version \
        --steps 100 \
        --particles 50000 \
        --load-measurements measurementbadf.mat
done

echo "All tests completed!"
```

运行：
```bash
chmod +x test_all_versions.sh
./test_all_versions.sh
```

## 更多信息

- V1 实现: [bp_slam/core/gnn_model.py](bp_slam/core/gnn_model.py)
- V2 实现: [bp_slam/core/gnn_model_v2.py](bp_slam/core/gnn_model_v2.py)
- V3 实现: [bp_slam/core/gnn_model_v3_explicit_factor.py](bp_slam/core/gnn_model_v3_explicit_factor.py)
- V3 测试: [test_gnn_v3_explicit_factor.py](test_gnn_v3_explicit_factor.py)
