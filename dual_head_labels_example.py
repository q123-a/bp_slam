"""
双头架构标签示例
演示质量头和关联头的标签生成、Loss计算和推理过程
"""

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# 场景设置
# ============================================================
print("=" * 80)
print("场景：SLAM 系统在某一时刻接收到 5 个测量，地图中有 3 个锚点")
print("=" * 80)
print()

# 测量数据
num_measurements = 5
num_anchors = 3

print("测量数据：")
measurements = np.array([
    [10.2, 15.8, 8.5, 12.3, 6.7],  # 距离 (m)
    [0.5, 0.5, 0.5, 0.5, 0.5],     # 方差
    [-45, -52, -48, -55, -50]      # RSS (dBm)
])
for i in range(num_measurements):
    print(f"  测量{i}: 距离={measurements[0,i]:.1f}m, RSS={measurements[2,i]:.0f}dBm")
print()

print("锚点状态：")
anchor_positions = np.array([
    [10.0, 5.0],   # 锚点0
    [6.5, 8.0],    # 锚点1
    [8.0, 3.0]     # 锚点2
])
for i in range(num_anchors):
    print(f"  锚点{i}: 位置=({anchor_positions[i,0]:.1f}, {anchor_positions[i,1]:.1f})")
print()


# ============================================================
# 第一部分：原始标签（数据加载器提供）
# ============================================================
print("=" * 80)
print("第一部分：原始标签（测量级）")
print("=" * 80)
print()

# 真实情况：
# - 测量0 来自锚点0（真实信号）
# - 测量1 是杂波（噪声）
# - 测量2 来自锚点2（真实信号）
# - 测量3 是杂波（噪声）
# - 测量4 来自锚点1（真实信号）

ground_truth_labels = {
    'true_id': np.array([0, -1, 2, -1, 1]),        # 测量对应的锚点ID，-1表示杂波
    'is_clutter': np.array([False, True, False, True, False])  # 是否为杂波
}

print("ground_truth_labels = {")
print(f"    'true_id':    {ground_truth_labels['true_id']}")
print(f"    'is_clutter': {ground_truth_labels['is_clutter']}")
print("}")
print()

print("解读：")
for i in range(num_measurements):
    if ground_truth_labels['is_clutter'][i]:
        print(f"  测量{i}: 杂波（噪声/误报）")
    else:
        anchor_id = ground_truth_labels['true_id'][i]
        print(f"  测量{i}: 来自锚点{anchor_id}（真实信号）")
print()
print()


# ============================================================
# 第二部分：质量头标签（Quality Head Labels）
# ============================================================
print("=" * 80)
print("第二部分：质量头标签 - 判断每个测量是否为真实信号")
print("=" * 80)
print()

# 从原始标签生成质量头标签
quality_labels = (~ground_truth_labels['is_clutter']).astype(np.float32)

print("质量头标签生成：")
print(f"  is_clutter:     {ground_truth_labels['is_clutter']}")
print(f"  quality_labels: {quality_labels}  (1.0=真实信号, 0.0=杂波)")
print()

# 模拟 GNN 质量头的输出（logits）
print("模拟 GNN 质量头输出：")
quality_logits = torch.tensor([2.3, -1.5, 1.8, -2.1, 2.0])  # 模拟的 logits
quality_probs = torch.sigmoid(quality_logits)

print(f"  quality_logits: {quality_logits.numpy()}")
print(f"  quality_probs:  {quality_probs.numpy()}")
print()

print("质量头预测结果：")
for i in range(num_measurements):
    prob = quality_probs[i].item()
    label = quality_labels[i]
    prediction = "真实信号" if prob > 0.5 else "杂波"
    truth = "真实信号" if label == 1.0 else "杂波"
    correct = "✓" if prediction == truth else "✗"
    print(f"  测量{i}: prob={prob:.3f} → 预测={prediction:6s} | 真实={truth:6s} {correct}")
print()

# 计算质量头 Loss
quality_labels_tensor = torch.tensor(quality_labels)
quality_loss = F.binary_cross_entropy(quality_probs, quality_labels_tensor)
print(f"质量头 Loss (BCE): {quality_loss.item():.4f}")
print()
print()


# ============================================================
# 第三部分：构建稀疏图的边
# ============================================================
print("=" * 80)
print("第三部分：构建稀疏图")
print("=" * 80)
print()

# 稀疏图：每个测量连接到所有锚点
# 节点编号：0-4 是测量节点，5-7 是锚点节点
edge_index = np.array([
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],  # 源节点（测量ID）
    [5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7]   # 目标节点（锚点ID+5）
])

print(f"稀疏图有 {edge_index.shape[1]} 条边：")
print("边索引：")
for i in range(edge_index.shape[1]):
    meas_id = edge_index[0, i]
    anchor_id = edge_index[1, i] - num_measurements
    print(f"  边{i:2d}: 测量{meas_id} → 锚点{anchor_id}")
print()
print()


# ============================================================
# 第四部分：关联头标签 - 边中心视角（当前实现）
# ============================================================
print("=" * 80)
print("第四部分：关联头标签 - 边中心视角（当前实现）")
print("=" * 80)
print()

# 从原始标签生成边级标签
true_ids = ground_truth_labels['true_id']
is_clutter_arr = ground_truth_labels['is_clutter']

association_labels_edge = np.zeros(edge_index.shape[1], dtype=np.float32)

print("边级标签生成：")
for i in range(edge_index.shape[1]):
    meas_id = edge_index[0, i]
    node_id = edge_index[1, i]
    anchor_id = node_id - num_measurements

    # 判断：测量不是杂波 且 测量的真实ID等于锚点ID
    if not is_clutter_arr[meas_id] and true_ids[meas_id] == anchor_id:
        association_labels_edge[i] = 1.0
        match_str = "✓ 匹配"
    else:
        match_str = "  不匹配"

    print(f"  边{i:2d} (测量{meas_id}→锚点{anchor_id}): 标签={association_labels_edge[i]:.0f} {match_str}")
print()

print(f"边级标签向量: {association_labels_edge}")
print(f"正样本数量: {int(np.sum(association_labels_edge))}/{len(association_labels_edge)}")
print()

# 模拟 GNN 关联头的输出（边中心）
print("模拟 GNN 关联头输出（边中心）：")
edge_logits = torch.tensor([
    2.5, -1.2, -0.8,  # 测量0→锚点0,1,2
    -2.0, -1.5, -1.8,  # 测量1→锚点0,1,2 (杂波)
    -0.5, -1.0, 2.8,   # 测量2→锚点0,1,2
    -1.8, -2.2, -1.5,  # 测量3→锚点0,1,2 (杂波)
    -0.9, 3.2, -1.1    # 测量4→锚点0,1,2
])
edge_probs = torch.sigmoid(edge_logits)

print(f"  edge_logits: {edge_logits.numpy()}")
print(f"  edge_probs:  {edge_probs.numpy()}")
print()

# 计算关联头 Loss（边中心）
association_labels_edge_tensor = torch.tensor(association_labels_edge)
assoc_loss_edge = F.binary_cross_entropy(edge_probs, association_labels_edge_tensor)
print(f"关联头 Loss (BCE, 边中心): {assoc_loss_edge.item():.4f}")
print()
print()


# ============================================================
# 第五部分：关联头标签 - 锚点中心视角（新方案）
# ============================================================
print("=" * 80)
print("第五部分：关联头标签 - 锚点中心视角（新方案）")
print("=" * 80)
print()

# 从边级标签生成锚点级标签
association_labels_anchor = np.zeros(num_anchors, dtype=np.int64)

print("锚点级标签生成：")
for i in range(edge_index.shape[1]):
    meas_id = edge_index[0, i]
    node_id = edge_index[1, i]
    anchor_id = node_id - num_measurements

    # 如果这条边是正样本
    if not is_clutter_arr[meas_id] and true_ids[meas_id] == anchor_id:
        association_labels_anchor[anchor_id] = meas_id + 1
        print(f"  锚点{anchor_id}: 选择测量{meas_id} (类别{meas_id + 1})")

# 打印未检测的锚点
for anchor_id in range(num_anchors):
    if association_labels_anchor[anchor_id] == 0:
        print(f"  锚点{anchor_id}: 未检测 (类别0)")

print()
print(f"锚点级标签向量: {association_labels_anchor}")
print("含义：[类别0=未检测, 类别1~5=选择测量0~4]")
print()

# 构建 Logits 矩阵 [K, M+1]
print("构建锚点中心 Logits 矩阵：")
dense_logits = torch.full((num_anchors, num_measurements + 1), -1e9)
dense_logits[:, 0] = 0.0  # 未检测列的基准

# 从边的 logits 填充矩阵
edge_index_tensor = torch.from_numpy(edge_index)
for i in range(edge_index.shape[1]):
    meas_id = edge_index[0, i]
    anchor_id = edge_index[1, i] - num_measurements
    dense_logits[anchor_id, meas_id + 1] = edge_logits[i]

print("dense_logits 形状:", dense_logits.shape, "(K, M+1)")
print("dense_logits:")
print("           未检测  测量0   测量1   测量2   测量3   测量4")
for k in range(num_anchors):
    row_str = f"  锚点{k}:  "
    for m in range(num_measurements + 1):
        val = dense_logits[k, m].item()
        if val > -100:
            row_str += f"{val:7.2f} "
        else:
            row_str += "   -inf "
    print(row_str)
print()

# 应用 Softmax（按行/按锚点）
print("应用 Softmax 归一化（按锚点）：")
probs_matrix = F.softmax(dense_logits, dim=1)  # (K, M+1)
print("probs_matrix 形状:", probs_matrix.shape, "(K, M+1)")
print("probs_matrix:")
print("           未检测  测量0   测量1   测量2   测量3   测量4   行和")
for k in range(num_anchors):
    row_str = f"  锚点{k}:  "
    row_sum = 0.0
    for m in range(num_measurements + 1):
        val = probs_matrix[k, m].item()
        row_str += f"{val:7.4f} "
        row_sum += val
    row_str += f" = {row_sum:.4f}"
    print(row_str)
print()

# 计算关联头 Loss（锚点中心）
association_labels_anchor_tensor = torch.tensor(association_labels_anchor, dtype=torch.long)
assoc_loss_anchor = F.cross_entropy(dense_logits, association_labels_anchor_tensor)
print(f"关联头 Loss (CrossEntropy, 锚点中心): {assoc_loss_anchor.item():.4f}")
print()

# 预测结果分析
print("锚点中心预测结果：")
predicted_classes = torch.argmax(probs_matrix, dim=1)
for k in range(num_anchors):
    pred_class = predicted_classes[k].item()
    true_class = association_labels_anchor[k]

    if pred_class == 0:
        pred_str = "未检测"
    else:
        pred_str = f"测量{pred_class-1}"

    if true_class == 0:
        true_str = "未检测"
    else:
        true_str = f"测量{true_class-1}"

    correct = "✓" if pred_class == true_class else "✗"
    print(f"  锚点{k}: 预测={pred_str:6s} | 真实={true_str:6s} {correct}")
print()
print()


# ============================================================
# 第六部分：总结对比
# ============================================================
print("=" * 80)
print("第六部分：总结对比")
print("=" * 80)
print()

print("双头架构的两种标签：")
print()
print("1. 质量头标签（测量级）：")
print(f"   形状: (M,) = ({num_measurements},)")
print(f"   标签: {quality_labels}")
print(f"   含义: 每个测量是否为真实信号 (1.0=真实, 0.0=杂波)")
print(f"   Loss: BCE = {quality_loss.item():.4f}")
print()

print("2. 关联头标签（两种视角）：")
print()
print("   视角A - 边中心（当前实现）：")
print(f"     形状: (E,) = ({edge_index.shape[1]},)")
print(f"     标签: {association_labels_edge}")
print(f"     含义: 每条边是否匹配 (1.0=匹配, 0.0=不匹配)")
print(f"     Loss: BCE = {assoc_loss_edge.item():.4f}")
print()

print("   视角B - 锚点中心（新方案）：")
print(f"     形状: (K,) = ({num_anchors},)")
print(f"     标签: {association_labels_anchor}")
print(f"     含义: 每个锚点选择哪个测量 (0=未检测, 1~M=测量ID+1)")
print(f"     Loss: CrossEntropy = {assoc_loss_anchor.item():.4f}")
print()

print("关键区别：")
print("  边中心: 每条边独立预测 → 可能出现'多个1'（多个测量关联到同一锚点）")
print("  锚点中心: Softmax强制互斥 → 每个锚点只能选一个测量（或未检测）")
print()
print()
