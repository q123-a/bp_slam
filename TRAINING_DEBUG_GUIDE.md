"""
GNN 训练问题诊断清单
================================

## 问题：GNN 把所有测量都预测为杂波

现象：
- GNN预测: 0/7 个真实信号, 7 个杂波
- 真实标签: 6 个真实信号, 1 个杂波
- 识别准确率: 14.3%

这说明模型完全崩溃了！

---

## 诊断步骤

### 1. 检查标签是否正确传递

**检查点 1**: 标签是否正确生成？
```bash
# 运行测试脚本验证标签
python test_label_generation.py
```

**预期输出**:
- 匈牙利算法匹配率 > 95%
- 标签数量 = 测量数量
- true_id 和 is_clutter 正确对应

**检查点 2**: 标签是否正确传递到 GNN？

在 slam.py 中添加调试输出：
```python
# 在 GNN 调用前添加
if step == 10 and sensor == 0:
    print(f"\\n[DEBUG] Step {step}, Sensor {sensor}")
    print(f"  测量数: {filtered_measurements.shape[1]}")
    if filtered_labels is not None:
        print(f"  标签 true_id: {filtered_labels['true_id']}")
        print(f"  标签 is_clutter: {filtered_labels['is_clutter']}")
        print(f"  真实信号数: {np.sum(~filtered_labels['is_clutter'])}")
        print(f"  杂波数: {np.sum(filtered_labels['is_clutter'])}")
    else:
        print(f"  ❌ 标签为 None！")
```

---

### 2. 检查损失函数是否正常

**检查点 3**: 加权 BCE 损失是否正确？

在 gnn_trainer_sparse_gat_v2.py 中添加调试：
```python
def _compute_quality_loss_supervised(self, quality_logits, ground_truth_labels):
    is_clutter = ground_truth_labels['is_clutter']
    target_quality = torch.from_numpy(~is_clutter).float().to(self.device)

    num_signals = torch.sum(target_quality)
    num_clutter = len(target_quality) - num_signals

    # 添加调试输出
    if self.step_count % 50 == 0:
        print(f"\\n[DEBUG Quality Loss]")
        print(f"  真实信号数: {num_signals.item()}")
        print(f"  杂波数: {num_clutter.item()}")
        print(f"  质量 logits 范围: [{quality_logits.min().item():.3f}, {quality_logits.max().item():.3f}]")

    # ... 继续原来的代码
```

---

### 3. 检查模型输出

**检查点 4**: 模型输出是否合理？

```python
# 在 step() 方法中添加
if self.step_count % 50 == 0:
    quality_probs = torch.sigmoid(quality_logits)
    print(f"\\n[DEBUG Model Output]")
    print(f"  质量概率范围: [{quality_probs.min().item():.3f}, {quality_probs.max().item():.3f}]")
    print(f"  质量概率均值: {quality_probs.mean().item():.3f}")
    print(f"  预测为真实信号: {torch.sum(quality_probs > 0.5).item()}")
    print(f"  预测为杂波: {torch.sum(quality_probs <= 0.5).item()}")
```

---

### 4. 检查学习率和梯度

**检查点 5**: 学习率是否过大或过小？

```python
# 在 step() 方法中添加
if self.step_count % 50 == 0:
    current_lr = self.optimizer.param_groups[0]['lr']
    print(f"\\n[DEBUG Training]")
    print(f"  当前学习率: {current_lr:.6f}")
    print(f"  训练步数: {self.step_count}")
```

**检查点 6**: 梯度是否正常？

```python
# 在反向传播后添加
if self.step_count % 50 == 0:
    total_norm = 0
    for p in self.model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"  梯度范数: {total_norm:.6f}")
```

---

### 5. 检查数据分布

**检查点 7**: 数据是否严重不平衡？

运行诊断脚本：
```python
# diagnose_data_distribution.py
import numpy as np

# 统计前 100 步的数据分布
total_signals = 0
total_clutter = 0

for step in range(100):
    for sensor in range(2):
        if labels[step][sensor] is not None:
            total_signals += np.sum(~labels[step][sensor]['is_clutter'])
            total_clutter += np.sum(labels[step][sensor]['is_clutter'])

clutter_ratio = total_clutter / (total_signals + total_clutter)
print(f"杂波比例: {clutter_ratio*100:.1f}%")
print(f"真实信号: {total_signals}")
print(f"杂波: {total_clutter}")

if clutter_ratio > 0.8:
    print("⚠️ 警告：杂波比例过高！")
```

---

## 常见问题和解决方案

### 问题 1: 标签没有正确传递

**症状**: filtered_labels 为 None

**原因**:
- 没有使用 --load-measurements
- 没有使用 --add-clutter
- 标签生成失败

**解决方案**:
```bash
# 确保使用正确的命令
python testbed.py \\
    --mode gnn \\
    --load-measurements measurement1500.mat \\
    --add-clutter \\
    --use-sparse-graph-v2
```

---

### 问题 2: 类别权重计算错误

**症状**: 权重全是 NaN 或 Inf

**原因**: num_signals 或 num_clutter 为 0

**解决方案**: 检查权重计算逻辑
```python
if num_signals > 0 and num_clutter > 0:
    # 正常计算权重
else:
    # 使用均匀权重
    weights = torch.ones_like(target_quality)
```

---

### 问题 3: 学习率过大

**症状**: Loss 不下降或震荡

**原因**: 学习率 1e-4 可能过大

**解决方案**: 降低学习率
```python
# 在 testbed.py 中
parameters['gnn_lr'] = 1e-5  # 从 1e-4 降到 1e-5
```

---

### 问题 4: 预热不够

**症状**: 前期准确率极低

**原因**: BP 还没稳定，GNN 接收到的预测不准

**解决方案**: 增加预热步数
```python
parameters['gnn_warmup_steps'] = 50  # 从 10 增加到 50
```

---

### 问题 5: 模型初始化问题

**症状**: 模型输出全是 0.5 附近

**原因**: 权重初始化不当

**解决方案**: 检查模型初始化
```python
# 在模型初始化后添加
for name, param in self.model.named_parameters():
    if 'weight' in name:
        print(f"{name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
```

---

## 快速修复建议

### 立即尝试的修复（按优先级）:

1. **检查标签传递** ✅ 最重要
   ```python
   # 在 slam.py 中添加调试输出
   if filtered_labels is None:
       print(f"❌ Step {step}, Sensor {sensor}: 标签为 None！")
   ```

2. **降低学习率**
   ```python
   parameters['gnn_lr'] = 5e-5  # 降低学习率
   ```

3. **增加预热步数**
   ```python
   parameters['gnn_warmup_steps'] = 30  # 增加预热
   ```

4. **检查数据分布**
   - 如果杂波比例 > 80%，考虑调整 meanNumberOfClutter

5. **减少迭代次数**
   ```python
   num_iterations = 1  # 先改回 1，确保基本功能正常
   ```

---

## 调试命令

### 1. 添加详细日志
```bash
python testbed.py \\
    --mode gnn \\
    --steps 100 \\
    --load-measurements measurement1500.mat \\
    --add-clutter \\
    --use-sparse-graph-v2 \\
    2>&1 | tee training_debug.log
```

### 2. 检查前 10 步的输出
```bash
python testbed.py --mode gnn --steps 10 --use-sparse-graph-v2
```

### 3. 测试标签生成
```bash
python test_label_generation.py
```

---

## 预期的正常输出

### 训练初期（Step 10-20）:
```
[稀疏图 GAT V2] Sensor 1, Step 10, Loss: 0.8-1.2
  GNN预测: 2-4/7 个真实信号, 3-5 个杂波
  真实标签: 6 个真实信号, 1 个杂波
  识别准确率: 40-60%
```

### 训练中期（Step 50-100）:
```
[稀疏图 GAT V2] Sensor 1, Step 50, Loss: 0.4-0.7
  GNN预测: 4-5/7 个真实信号, 2-3 个杂波
  真实标签: 6 个真实信号, 1 个杂波
  识别准确率: 70-85%
```

如果看不到这样的改善，说明训练有问题！

---

## 下一步行动

1. 立即添加调试输出，确认标签是否正确传递
2. 检查前 10 步的详细输出
3. 根据诊断结果调整参数
4. 重新训练并观察

请先运行诊断，然后告诉我结果！
"""
