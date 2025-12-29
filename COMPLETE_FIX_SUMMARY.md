"""
完整修复方案总结
================================

## 概述

本文档总结了为实现完美的 GNN 监督学习训练效果所做的所有关键修复。

---

## 🎯 核心问题和解决方案

### 问题 1: 类别不平衡导致模型崩溃

**现象**:
- GNN 把所有测量都预测为杂波
- 识别准确率极低（14%-33%）
- Loss 很高且不下降

**原因**:
当数据中杂波比例很高时（如 80% 杂波），使用普通 BCE 损失会导致模型学会"总是预测杂波"。

**解决方案**: 使用加权 BCE 损失

```python
# 质量损失 - 加权版本
def _compute_quality_loss_supervised(self, quality_logits, ground_truth_labels):
    is_clutter = ground_truth_labels['is_clutter']
    target_quality = torch.from_numpy(~is_clutter).float().to(self.device)

    # 计算类别权重
    num_signals = torch.sum(target_quality)
    num_clutter = len(target_quality) - num_signals

    if num_signals > 0 and num_clutter > 0:
        total = len(target_quality)
        pos_weight = total / (2.0 * num_signals)  # 真实信号权重
        neg_weight = total / (2.0 * num_clutter)  # 杂波权重
        weights = torch.where(target_quality == 1.0, pos_weight, neg_weight)
    else:
        weights = torch.ones_like(target_quality)

    # 加权 BCE 损失
    quality_probs = torch.sigmoid(quality_logits)
    loss = F.binary_cross_entropy(quality_probs, target_quality, weight=weights)
    return loss
```

**效果**:
- ✅ 模型不再偏向预测杂波
- ✅ 少数类（真实信号）得到足够重视
- ✅ 训练稳定，准确率提升

---

### 问题 2: 多传感器训练冲突

**现象**:
- 学习率下降过快
- 两个传感器的梯度互相干扰
- 训练不稳定，损失曲线锯齿状

**原因**:
每个时间步对两个传感器分别训练，导致：
1. 学习率每步更新 2 次（下降速度是预期的 2 倍）
2. Sensor 1 的更新可能被 Sensor 2 破坏
3. 损失记录混乱（每步记录 2 次）

**解决方案**: 梯度累积 + 统一更新

```python
# GNN Trainer 修改
def step(..., update_weights=True):
    # 训练循环
    for iter_idx in range(num_iterations):
        loss.backward()  # 梯度累积

        # 只在 update_weights=True 时更新
        if update_weights:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.05)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.ema_model is not None:
                self._update_ema()

    # 只在 update_weights=True 时更新学习率和记录损失
    if update_weights:
        self.scheduler.step()
        self.loss_history.append(avg_loss)
        self.quality_loss_history.append(avg_quality_loss)
        self.assoc_loss_history.append(avg_assoc_loss)
```

```python
# SLAM 循环修改
for step in range(num_steps):
    for sensor in range(num_sensors):
        is_last_sensor = (sensor == num_sensors - 1)

        loss = gnn_trainer.step(
            ...,
            sensor_id=sensor,
            ground_truth_labels=filtered_labels,
            update_weights=is_last_sensor  # 只在最后一个传感器更新
        )
```

**效果**:
- ✅ 学习率每步只更新 1 次
- ✅ 两个传感器的梯度合并后统一更新
- ✅ 训练更稳定，收敛更快

---

### 问题 3: 输出统计信息过时

**现象**:
显示基于 RSS 误差的统计信息，但使用监督学习后这些信息不再准确。

**解决方案**: 显示真实标签统计和准确率

```python
# 打印 GNN 预测统计
num_predicted_good = np.sum(dustbin_probs < 0.5)
num_predicted_bad = np.sum(dustbin_probs >= 0.5)
print(f"    GNN预测: {num_predicted_good}/{len(dustbin_probs)} 个真实信号, {num_predicted_bad} 个杂波")

# 如果有真实标签，显示对比和准确率
if filtered_labels is not None:
    true_signals = np.sum(~filtered_labels['is_clutter'])
    true_clutter = np.sum(filtered_labels['is_clutter'])
    print(f"    真实标签: {true_signals} 个真实信号, {true_clutter} 个杂波")

    # 计算准确率
    predicted_clutter = (dustbin_probs >= 0.5)
    true_clutter_mask = filtered_labels['is_clutter']
    correct = np.sum(predicted_clutter == true_clutter_mask)
    accuracy = correct / len(dustbin_probs) * 100
    print(f"    识别准确率: {accuracy:.1f}% ({correct}/{len(dustbin_probs)})")
```

**效果**:
- ✅ 直观看到 GNN 预测 vs 真实标签
- ✅ 实时监控识别准确率
- ✅ 便于调试和评估

---

## 📊 完整的训练流程

### 1. 数据准备

```bash
# 使用物理仿真数据 + 匈牙利算法推断标签 + 添加合成杂波
python testbed.py \
    --mode gnn \
    --steps 900 \
    --particles 100000 \
    --load-measurements measurement1500.mat \
    --add-clutter \
    --use-sparse-graph-v2
```

### 2. 训练过程

```
每个时间步:
  1. Sensor 0: 前向传播 → 计算损失 → 反向传播 → 梯度累积
  2. Sensor 1: 前向传播 → 计算损失 → 反向传播 → 梯度累积 → 统一更新权重
  3. 每 10 步打印当前损失和准确率
  4. 每 50 步打印损失统计（当前 + 移动平均）
```

### 3. 训练结束

```
1. 打印完整损失统计
2. 保存 checkpoint（包含损失历史）
3. 保存损失历史到 CSV
4. 可视化损失曲线
```

---

## 🎯 预期训练效果

### 训练初期（0-50 步）

```
Step 10:
  [稀疏图 GAT V2] Sensor 1, Step 10, Loss: 1.2xxx
    GNN预测: 3/7 个真实信号, 4 个杂波
    真实标签: 6 个真实信号, 1 个杂波
    识别准确率: 42.9% (3/7)
```

**特点**:
- Loss 较高（1.0-1.5）
- 准确率较低（40%-60%）
- 模型开始学习区分真实信号和杂波

### 训练中期（50-200 步）

```
Step 100:
  [稀疏图 GAT V2] Sensor 1, Step 100, Loss: 0.6xxx
    GNN预测: 5/7 个真实信号, 2 个杂波
    真实标签: 6 个真实信号, 1 个杂波
    识别准确率: 71.4% (5/7)

------------------------------------------------------------
  [Loss] 当前: 0.6245 (质量: 0.3123, 关联: 0.3122)
  [Loss] 平均(50步): 0.7234 (质量: 0.3623, 关联: 0.3611)
------------------------------------------------------------
```

**特点**:
- Loss 稳定下降（0.5-0.8）
- 准确率提升（70%-85%）
- 损失曲线平滑

### 训练后期（200+ 步）

```
Step 300:
  [稀疏图 GAT V2] Sensor 1, Step 300, Loss: 0.3xxx
    GNN预测: 6/7 个真实信号, 1 个杂波
    真实标签: 6 个真实信号, 1 个杂波
    识别准确率: 100.0% (7/7)

------------------------------------------------------------
  [Loss] 当前: 0.3145 (质量: 0.1523, 关联: 0.1622)
  [Loss] 平均(50步): 0.3534 (质量: 0.1723, 关联: 0.1811)
------------------------------------------------------------
```

**特点**:
- Loss 很低（0.2-0.4）
- 准确率很高（90%-100%）
- 模型收敛

---

## 📁 关键文件修改总结

### 1. gnn_trainer_sparse_gat_v2.py

**修改点**:
- ✅ 添加 `update_weights` 参数
- ✅ 质量损失使用加权 BCE
- ✅ 关联损失使用加权 BCE
- ✅ 条件更新权重和学习率
- ✅ 条件记录损失历史
- ✅ 保存/加载所有三种损失历史
- ✅ 新增 `save_loss_history_to_csv()` 方法
- ✅ 新增 `print_loss_statistics()` 方法

### 2. slam.py

**修改点**:
- ✅ 添加 `is_last_sensor` 判断
- ✅ 传递 `update_weights` 参数
- ✅ 更新输出统计信息（显示真实标签和准确率）
- ✅ 每 50 步打印损失统计
- ✅ 训练结束时打印完整统计
- ✅ 自动保存损失历史到 CSV

### 3. 新增文件

- ✅ `visualize_loss.py` - 损失可视化脚本
- ✅ `LOSS_TRACKING.md` - 损失跟踪功能说明
- ✅ `COMPLETE_FIX_SUMMARY.md` - 本文档

---

## ⚙️ 推荐参数设置

### testbed.py 参数

```python
parameters = {
    # GNN 训练参数
    'gnn_lr': 1e-4,                    # 学习率
    'gnn_hidden_dim': 64,              # 隐藏层维度
    'gnn_num_layers': 2,               # GAT 层数
    'gnn_heads': 4,                    # 注意力头数
    'gnn_dropout': 0.1,                # Dropout
    'gnn_use_ema': True,               # 使用 EMA
    'gnn_ema_decay': 0.999,            # EMA 衰减率

    # 损失权重
    'gnn_quality_weight': 1.5,         # 质量损失权重
    'gnn_assoc_weight': 1.0,           # 关联损失权重
    'gnn_adaptive_weighting': True,    # 自适应权重

    # 预热和保存
    'gnn_warmup_steps': 10,            # 预热步数（监督学习）
    'gnn_save_checkpoint': True,       # 保存 checkpoint
    'gnn_save_loss_csv': True,         # 保存损失 CSV

    # 数据参数
    'detectionProbability': 0.95,      # 检测概率
    'meanNumberOfClutter': 1,          # 平均杂波数
}
```

---

## 🚀 使用流程

### 步骤 1: 训练模型

```bash
python testbed.py \
    --mode gnn \
    --steps 900 \
    --particles 100000 \
    --load-measurements measurement1500.mat \
    --add-clutter \
    --use-sparse-graph-v2
```

### 步骤 2: 观察训练过程

监控以下指标:
- ✅ Loss 是否稳定下降
- ✅ 识别准确率是否提升
- ✅ 每 50 步的移动平均是否改善

### 步骤 3: 可视化损失曲线

```bash
python visualize_loss.py
```

### 步骤 4: 分析结果

检查:
- ✅ 损失曲线是否平滑下降
- ✅ 最终准确率是否达到 90%+
- ✅ 模型是否收敛

---

## 🔍 故障排查

### 问题: Loss 不下降

**可能原因**:
1. 学习率过大或过小
2. 数据标签质量差
3. 模型容量不足

**解决方案**:
- 调整学习率（1e-5 到 1e-3）
- 检查标签准确性
- 增加隐藏层维度或层数

### 问题: 准确率很低

**可能原因**:
1. 类别不平衡未解决
2. 预热步数不够
3. 数据质量问题

**解决方案**:
- 确认使用了加权 BCE 损失
- 增加预热步数到 20-30
- 检查匈牙利算法匹配率

### 问题: 训练不稳定

**可能原因**:
1. 学习率过大
2. 梯度爆炸
3. 批量大小问题

**解决方案**:
- 降低学习率
- 检查梯度裁剪（max_norm=0.05）
- 确认使用梯度累积

---

## ✅ 验证清单

训练前检查:
- [ ] 使用加权 BCE 损失
- [ ] 使用梯度累积（多传感器）
- [ ] 设置合适的预热步数
- [ ] 启用损失跟踪和保存

训练中监控:
- [ ] Loss 稳定下降
- [ ] 准确率逐步提升
- [ ] 无异常波动

训练后验证:
- [ ] 最终准确率 > 90%
- [ ] 损失曲线平滑
- [ ] Checkpoint 正确保存
- [ ] CSV 文件生成

---

## 📝 总结

所有关键问题已修复:
1. ✅ 类别不平衡 → 加权 BCE 损失
2. ✅ 多传感器冲突 → 梯度累积
3. ✅ 学习率问题 → 统一更新
4. ✅ 损失跟踪 → 完整实现
5. ✅ 输出统计 → 显示准确率

**现在系统已准备好进行完美的监督学习训练！** 🎉
"""
