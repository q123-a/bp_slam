"""
损失跟踪功能说明
================================

## 概述

系统现在支持完整的损失跟踪和可视化功能，包括：

1. **实时损失记录**: 训练过程中自动记录总损失、质量损失、关联损失
2. **定期统计打印**: 每 50 步打印一次损失统计信息
3. **CSV 导出**: 训练结束后自动保存损失历史到 CSV 文件
4. **可视化工具**: 提供脚本绘制损失曲线图

---

## 功能特性

### 1. 自动损失记录

训练器会自动记录三种损失：

- **总损失 (Total Loss)**: 质量损失 + 关联损失的加权和
- **质量损失 (Quality Loss)**: 杂波检测损失（BCE loss）
- **关联损失 (Association Loss)**: 数据关联损失（BCE loss）

### 2. 实时统计打印

训练过程中会定期打印损失信息：

- **每 10 步**: 打印当前步的损失值
- **每 50 步**: 打印详细的损失统计（当前值 + 50 步移动平均）

### 3. 自动保存

训练结束后自动保存：

- **Checkpoint**: 包含模型权重、优化器状态、损失历史
- **CSV 文件**: 包含每一步的详细损失数据

---

## 使用方法

### 方法 1: 训练时自动记录

运行训练命令，损失会自动记录：

```bash
python testbed.py \
    --mode gnn \
    --steps 900 \
    --particles 100000 \
    --load-measurements measurement1500.mat \
    --add-clutter \
    --use-sparse-graph-v2
```

**输出示例**:

```
Step 0:
  [稀疏图 GAT V2] Sensor 1, Step 0, Loss: 0.6931
    杂波识别: 8/11 个真实信号, 3 个杂波
    ...

Step 50:
  [稀疏图 GAT V2] Sensor 1, Step 50, Loss: 0.3245

------------------------------------------------------------
  [Loss] 当前: 0.3245 (质量: 0.1523, 关联: 0.1722)
  [Loss] 平均(50步): 0.4123 (质量: 0.2145, 关联: 0.1978)
------------------------------------------------------------
```

### 方法 2: 训练结束后查看统计

训练结束时会自动打印完整统计：

```
================================================================================
GNN 训练损失统计
================================================================================
  [Loss] 当前: 0.2145 (质量: 0.0987, 关联: 0.1158)
  [Loss] 平均(50步): 0.2234 (质量: 0.1023, 关联: 0.1211)
================================================================================

✓ 权重已保存: checkpoints/gnn_model.pth
✓ 损失历史已保存到 CSV: results/loss_history.csv
```

### 方法 3: 可视化损失曲线

使用提供的可视化脚本：

```bash
# 基本用法
python visualize_loss.py

# 指定文件路径
python visualize_loss.py \
    --csv results/loss_history.csv \
    --output results/loss_curves.png \
    --window 10

# 不显示图形窗口（仅保存）
python visualize_loss.py --no-show
```

**输出**:

1. 控制台打印详细统计
2. 保存损失曲线图（4 个子图）

---

## 配置参数

### testbed.py 中的参数

```python
parameters = {
    # 是否保存 checkpoint（包含损失历史）
    'gnn_save_checkpoint': True,
    'gnn_checkpoint_save_path': 'checkpoints/gnn_model.pth',

    # 是否保存损失历史到 CSV（默认 True）
    'gnn_save_loss_csv': True,
    'gnn_loss_csv_path': 'results/loss_history.csv',

    # 是否仅推理模式（不记录损失）
    'gnn_inference_only': False,
}
```

### 可视化脚本参数

```bash
python visualize_loss.py --help

参数:
  --csv CSV          损失历史 CSV 文件路径 (默认: results/loss_history.csv)
  --output OUTPUT    输出图像路径 (默认: results/loss_curves.png)
  --window WINDOW    移动平均窗口大小 (默认: 10)
  --no-show          不显示图形窗口
```

---

## 输出文件格式

### CSV 文件格式

`results/loss_history.csv`:

```csv
Step,Total_Loss,Quality_Loss,Assoc_Loss
0,0.6931,0.3465,0.3466
1,0.6523,0.3201,0.3322
2,0.6145,0.2987,0.3158
...
```

### Checkpoint 文件内容

`checkpoints/gnn_model.pth`:

```python
{
    'model_state_dict': ...,           # 模型权重
    'ema_model_state_dict': ...,       # EMA 模型权重
    'optimizer_state_dict': ...,       # 优化器状态
    'step_count': 900,                 # 训练步数
    'loss_history': [...],             # 总损失历史
    'quality_loss_history': [...],     # 质量损失历史
    'assoc_loss_history': [...]        # 关联损失历史
}
```

---

## 可视化输出

### 损失曲线图

`results/loss_curves.png` 包含 4 个子图：

1. **总损失 (Total Loss)**: 原始值 + 移动平均
2. **质量损失 (Quality Loss)**: 杂波检测损失
3. **关联损失 (Association Loss)**: 数据关联损失
4. **损失对比**: 三种损失的移动平均对比

### 统计摘要

```
================================================================================
损失统计摘要
================================================================================

训练步数: 900

总损失 (Total Loss):
  初始值: 0.6931
  最终值: 0.2145
  最小值: 0.2087 (步骤 856)
  最大值: 0.7123 (步骤 3)
  平均值: 0.3456
  标准差: 0.1234

质量损失 (Quality Loss):
  初始值: 0.3465
  最终值: 0.0987
  ...

关联损失 (Association Loss):
  初始值: 0.3466
  最终值: 0.1158
  ...

最后 50 步平均:
  总损失: 0.2234
  质量损失: 0.1023
  关联损失: 0.1211
================================================================================
```

---

## 代码实现

### GNN Trainer 中的方法

```python
# 保存损失历史到 CSV
gnn_trainer.save_loss_history_to_csv('results/loss_history.csv')

# 打印损失统计
gnn_trainer.print_loss_statistics(window_size=50)

# 保存 checkpoint（包含损失历史）
gnn_trainer.save_checkpoint('checkpoints/gnn_model.pth')
```

### 从 Checkpoint 加载损失历史

```python
import torch

checkpoint = torch.load('checkpoints/gnn_model.pth')
loss_history = checkpoint['loss_history']
quality_loss_history = checkpoint['quality_loss_history']
assoc_loss_history = checkpoint['assoc_loss_history']

print(f"训练步数: {len(loss_history)}")
print(f"最终损失: {loss_history[-1]:.4f}")
```

---

## 常见问题

### Q1: 为什么损失曲线有波动？

**A**: 这是正常现象，原因包括：
- 每步的测量数量和杂波比例不同
- 随机采样导致的方差
- 学习率调度导致的波动

**解决方案**: 使用移动平均平滑曲线（默认窗口大小 10）

### Q2: 如何判断训练是否收敛？

**A**: 观察以下指标：
- 总损失持续下降并趋于稳定
- 最后 50 步的平均损失变化小于 5%
- 质量损失和关联损失都在下降

### Q3: 损失突然上升怎么办？

**A**: 可能原因：
- 学习率过大（检查学习率调度）
- 遇到困难样本（正常现象）
- 梯度爆炸（检查梯度裁剪）

**解决方案**:
- 降低学习率
- 增加梯度裁剪阈值
- 检查数据质量

### Q4: CSV 文件在哪里？

**A**: 默认保存在 `results/loss_history.csv`

如果找不到，检查：
- 是否设置了 `gnn_save_loss_csv=False`
- 是否在仅推理模式 (`gnn_inference_only=True`)
- 训练是否正常结束

---

## 总结

损失跟踪功能提供了完整的训练监控能力：

1. ✅ **自动记录**: 无需手动操作，训练时自动记录
2. ✅ **实时监控**: 每 50 步打印统计，及时发现问题
3. ✅ **持久化存储**: CSV 和 checkpoint 双重保存
4. ✅ **可视化分析**: 一键生成损失曲线图
5. ✅ **详细统计**: 提供完整的统计摘要

**推荐工作流**:

1. 运行训练命令
2. 观察实时损失打印
3. 训练结束后查看统计摘要
4. 运行 `visualize_loss.py` 生成曲线图
5. 分析损失趋势，调整超参数
"""
