"""
双头架构标签转换示例
演示如何从测量级标签生成质量头和关联头的标签
"""

import numpy as np
import torch


def convert_labels_for_dual_head(ground_truth_labels, edge_index, num_measurements, num_anchors):
    """
    从测量级标签转换为双头架构的标签

    参数:
        ground_truth_labels: {
            'true_id': (M,) 每个测量对应的锚点ID，-1表示杂波
            'is_clutter': (M,) 每个测量是否为杂波
        }
        edge_index: (2, E) 稀疏图的边索引
        num_measurements: M
        num_anchors: K

    返回:
        quality_labels: (M,) 质量头标签（0=杂波, 1=真实信号）
        association_labels_edge: (E,) 关联头标签-边视角（0/1）
        association_labels_anchor: (K,) 关联头标签-锚点视角（0~M）
    """

    true_ids = ground_truth_labels['true_id']  # (M,)
    is_clutter = ground_truth_labels['is_clutter']  # (M,)

    # ============================================================
    # 1. 质量头标签：直接使用 is_clutter
    # ============================================================
    quality_labels = ~is_clutter  # True信号=True, 杂波=False
    # 或者转换为 float: 1.0=真实信号, 0.0=杂波
    quality_labels_float = quality_labels.astype(np.float32)

    # ============================================================
    # 2. 关联头标签（边视角）：遍历每条边判断是否匹配
    # ============================================================
    src_nodes = edge_index[0]  # 测量ID
    dst_nodes = edge_index[1]  # 节点ID

    association_labels_edge = np.zeros(edge_index.shape[1], dtype=np.float32)

    for i in range(edge_index.shape[1]):
        meas_id = src_nodes[i]
        node_id = dst_nodes[i]

        # 只处理锚点边
        if node_id >= num_measurements:
            anchor_id = node_id - num_measurements

            # 判断：测量不是杂波 且 测量的真实ID等于锚点ID
            if not is_clutter[meas_id] and true_ids[meas_id] == anchor_id:
                association_labels_edge[i] = 1.0

    # ============================================================
    # 3. 关联头标签（锚点视角）：每个锚点选择哪个测量
    # ============================================================
    association_labels_anchor = np.zeros(num_anchors, dtype=np.int64)

    for i in range(edge_index.shape[1]):
        meas_id = src_nodes[i]
        node_id = dst_nodes[i]

        if node_id >= num_measurements:
            anchor_id = node_id - num_measurements

            # 如果这条边是正样本
            if not is_clutter[meas_id] and true_ids[meas_id] == anchor_id:
                # 锚点选择该测量（类别 = 测量ID + 1）
                association_labels_anchor[anchor_id] = meas_id + 1

    return quality_labels_float, association_labels_edge, association_labels_anchor


# ============================================================
# 示例：完整的标签转换流程
# ============================================================

if __name__ == "__main__":
    # 模拟数据
    num_measurements = 5
    num_anchors = 3

    # 测量级标签（数据加载器提供）
    ground_truth_labels = {
        'true_id': np.array([0, -1, 2, -1, 1]),  # 测量0→锚点0, 测量2→锚点2, 测量4→锚点1
        'is_clutter': np.array([False, True, False, True, False])
    }

    # 稀疏图的边（假设每个测量连接到所有锚点）
    edge_index = np.array([
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],  # 测量ID
        [5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7]   # 节点ID (5,6,7对应锚点0,1,2)
    ])

    # 转换标签
    quality_labels, assoc_labels_edge, assoc_labels_anchor = convert_labels_for_dual_head(
        ground_truth_labels, edge_index, num_measurements, num_anchors
    )

    print("=" * 60)
    print("输入：测量级标签")
    print("=" * 60)
    print(f"true_id:    {ground_truth_labels['true_id']}")
    print(f"is_clutter: {ground_truth_labels['is_clutter']}")
    print()

    print("=" * 60)
    print("输出1：质量头标签 (M,)")
    print("=" * 60)
    print(f"quality_labels: {quality_labels}")
    print("含义：[1.0=真实信号, 0.0=杂波]")
    print()

    print("=" * 60)
    print("输出2：关联头标签-边视角 (E,)")
    print("=" * 60)
    print(f"association_labels_edge: {assoc_labels_edge}")
    print("边索引：")
    for i in range(edge_index.shape[1]):
        meas_id = edge_index[0, i]
        anchor_id = edge_index[1, i] - num_measurements
        label = assoc_labels_edge[i]
        print(f"  边{i}: 测量{meas_id}→锚点{anchor_id}  标签={label}")
    print()

    print("=" * 60)
    print("输出3：关联头标签-锚点视角 (K,)")
    print("=" * 60)
    print(f"association_labels_anchor: {assoc_labels_anchor}")
    print("含义：")
    for k in range(num_anchors):
        choice = assoc_labels_anchor[k]
        if choice == 0:
            print(f"  锚点{k}: 未检测 (类别0)")
        else:
            print(f"  锚点{k}: 选择测量{choice-1} (类别{choice})")
