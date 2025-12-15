# 当前测试状态

## 问题诊断

### 发现的问题
之前的测试结果显示 GNN 误差比 BP 大 82.5%，原因是：
- **旧的单头架构checkpoint被自动加载**
- 系统使用了旧的单头GNN，而不是新的双头架构
- 配置显示：`gnn_use_dual_head: False`（应该是 True）

### 解决方案
1. ✅ 备份旧的单头架构checkpoint到 `checkpoints/old_single_head/`
2. ✅ 清空checkpoints目录，让系统从头训练双头架构
3. 🔄 正在运行新的测试（双头架构 + 失配模式）

## 当前运行的测试

### 命令
```bash
python testbed.py --mode gnn --steps 900 \
  --load-measurements measurementbadf.mat \
  --add-clutter --mismatch-mode
```

### 预期配置
- **架构模式**: 双头 (质量头+关联头)
- **预热步数**: 45步 (5%)
- **质量阈值**: 0.5
- **关联阈值**: 3.0
- **新锚点缓冲区**: 阈值=0.6, 最少帧数=3, 最大间隔=2
- **失配模式优化**: 阈值=2.0, 权重=10.0

### 预期效果
双头架构应该展现出以下优势：
1. **质量头不依赖BP预测** - 即使BP参数失配，质量头仍能正确识别杂波
2. **关联头只训练好点** - 避免杂波污染模型
3. **新锚点缓冲区** - 防止瞬时噪声被误判为新锚点

### 预期结果
- GNN平均误差应该 **小于或接近** BP的误差（~0.019m）
- 如果成功，改进幅度应该在 10-30% 之间
- 新锚点数量应该更稳定，没有"幽灵路标"

## 测试完成后的操作

### 1. 查看结果
```bash
python -c "
import numpy as np

gnn_data = np.load('results/results_gnn.npz', allow_pickle=True)
bp_data = np.load('results/results_bp.npz', allow_pickle=True)

gnn_mean = gnn_data['mean_error']
bp_mean = bp_data['mean_error']

print(f'GNN平均误差: {gnn_mean:.6f} m')
print(f'BP平均误差: {bp_mean:.6f} m')
print(f'改进: {(bp_mean-gnn_mean)/bp_mean*100:+.2f}%')

# 检查配置
params = gnn_data['parameters'].item()
print(f'\n配置确认:')
print(f'  - 双头架构: {params.get(\"gnn_use_dual_head\", False)}')
print(f'  - 预热步数: {params.get(\"gnn_warmup_steps\", \"未知\")}')
"
```

### 2. 可视化对比
```bash
python analyze_results.py
```

### 3. 如果结果仍不理想

可能需要调整的参数：

#### 选项 1：降低质量阈值（更宽松）
```python
parameters['gnn_quality_threshold'] = 0.3  # 从0.5降到0.3
```
- 效果：更多测量参与关联训练
- 适用：如果质量头过于严格，导致训练样本太少

#### 选项 2：提高质量阈值（更严格）
```python
parameters['gnn_quality_threshold'] = 0.7  # 从0.5提高到0.7
```
- 效果：只有高质量测量参与关联训练
- 适用：如果杂波仍然污染训练

#### 选项 3：调整损失权重
```python
parameters['gnn_quality_weight'] = 2.0  # 从1.0提高到2.0
parameters['gnn_assoc_weight'] = 1.0   # 从2.0降到1.0
```
- 效果：强化质量判断，弱化关联学习
- 适用：如果质量头判断不准确

#### 选项 4：增加预热步数
```python
parameters['gnn_warmup_steps'] = int(max_steps * 0.10)  # 从5%增加到10%
```
- 效果：给模型更多时间稳定
- 适用：如果前期误差很大

#### 选项 5：放宽新锚点缓冲区
```python
parameters['gnn_new_anchor_threshold'] = 0.5  # 从0.6降到0.5
parameters['gnn_new_anchor_min_frames'] = 2   # 从3降到2
```
- 效果：更容易添加新锚点
- 适用：如果新锚点检测太少

## 调试技巧

### 监控质量头性能
在 `gnn_trainer_improved.py` 的 `_compute_quality_loss` 中添加：
```python
if hasattr(self, 'step_count') and self.step_count % 100 == 0:
    good_count = good_mask.sum().item()
    bad_count = bad_mask.sum().item()
    print(f"[Step {self.step_count}] 质量判断:")
    print(f"  好点: {good_count}/{M} ({good_count/M*100:.1f}%)")
    print(f"  坏点: {bad_count}/{M} ({bad_count/M*100:.1f}%)")
```

### 监控关联头性能
在 `gnn_trainer_improved.py` 的 `_compute_association_loss` 中添加：
```python
if hasattr(self, 'step_count') and self.step_count % 100 == 0:
    print(f"[Step {self.step_count}] 关联训练:")
    print(f"  质量好的测量: {len(valid_indices)}/{M}")
    print(f"  匹配成功: {match_mask.sum().item()}/{len(valid_indices)}")
```

### 监控新锚点缓冲区
在 `slam.py` 的候选缓冲区更新后添加：
```python
if step % 50 == 0 and len(new_candidates) > 0:
    print(f"\n[Step {step}] 新锚点候选缓冲区:")
    for cand_id, cand_info in new_candidates.items():
        print(f"  {cand_id}: count={cand_info['count']}, dist={cand_info['distance']:.2f}m")
```

## 理论预期

根据双头架构的设计原理，在失配模式下：

| 指标 | BP | 单头GNN | 双头GNN | 原因 |
|-----|----|---------|---------|----|
| 平均误差 | 0.019m | 0.034m ❌ | **0.015m** ✅ | 质量头不依赖预测 |
| 鲁棒性 | 差 | 差 | **强** | 物理约束优先 |
| 新锚点质量 | 中 | 差 | **好** | 多帧验证机制 |

如果双头GNN仍然劣于BP，说明：
1. 质量头的RSS内在一致性判断可能不够准确
2. 路径损耗模型参数（P_tx=15.41, n=2.0）可能不匹配实际环境
3. 需要调整质量阈值或损失权重

## 下一步

等待测试完成后：
1. 检查配置是否正确（`gnn_use_dual_head: True`）
2. 对比误差结果
3. 根据结果决定是否需要调参
4. 如果成功，运行完整的对比测试（BP vs 双头GNN）
