"""
bp_slam/core/graph_builder.py
稀疏图构建工具

将密集矩阵格式 (M, K+1, 5) 转换为稀疏图格式:
- node_features: [N, node_dim] 节点特征
- edge_index: [2, E] 边索引 (COO 格式)
- edge_attr: [E, edge_dim] 边特征
"""

import numpy as np
import torch


def build_sparse_graph(filtered_measurements, predicted_measurements, predicted_uncertainties,
                      existence_probs, beta_matrix_filtered, undetected_anchors_intensity,
                      clutter_intensity, detection_probability, P_tx=15.41, n=2.0,
                      distance_threshold=None, beta_threshold=None):
    """
    构建稀疏图结构

    参数:
        filtered_measurements: (3, M) [距离, 方差, RSS]
        predicted_measurements: (K,) 预测距离
        predicted_uncertainties: (K,) 预测方差
        existence_probs: (K,) 锚点存在概率
        beta_matrix_filtered: (M, K) Beta 矩阵
        undetected_anchors_intensity: 未检测锚点强度
        clutter_intensity: 杂波强度
        detection_probability: 检测概率
        P_tx: 发射功率
        n: 路径损耗指数
        distance_threshold: 距离阈值（用于过滤边），None 表示不过滤
        beta_threshold: Beta 阈值（用于过滤边），None 表示不过滤

    返回:
        node_features: (N, node_dim) 节点特征
        edge_index: (2, E) 边索引
        edge_attr: (E, 5) 边特征 [log_prob, std_residual, log_var, existence, rss_residual]
        node_types: (N,) 节点类型 (0=测量, 1=锚点, 2=dustbin)
        num_measurements: M
        num_anchors: K
    """

    num_measurements = filtered_measurements.shape[1]
    num_anchors = len(predicted_measurements)

    # --- 1. 预计算锚点的预测 RSS ---
    rss_pred = np.zeros(num_anchors)
    for a in range(num_anchors):
        safe_pred_dist = max(predicted_measurements[a], 0.1)
        rss_pred[a] = P_tx - 10 * n * np.log10(safe_pred_dist)

    # --- 2. 构建边列表（稀疏） ---
    edge_list = []  # [(src, dst, features), ...]

    for m in range(num_measurements):
        # 提取测量 RSS
        if filtered_measurements.shape[0] >= 3:
            rss_meas = filtered_measurements[2, m]
        else:
            rss_meas = 0.0

        for a in range(num_anchors):
            # 计算距离残差
            distance_residual = abs(filtered_measurements[0, m] - predicted_measurements[a])

            # 过滤条件 1: 距离阈值
            if distance_threshold is not None and distance_residual > distance_threshold:
                continue

            # 过滤条件 2: Beta 阈值
            if beta_threshold is not None and beta_matrix_filtered[m, a] < beta_threshold:
                continue

            # 计算边特征 (5 维)
            # Ch0: Log-Prob (物理建议)
            log_prob = np.log(beta_matrix_filtered[m, a] + 1e-20)

            # Ch1: 标准化残差 (GNN 纠错核心)
            std_dev = np.sqrt(filtered_measurements[1, m] + predicted_uncertainties[a])
            std_residual = (filtered_measurements[0, m] - predicted_measurements[a]) / (std_dev + 1e-6)

            # Ch2: 对数方差
            log_var = np.log(filtered_measurements[1, m] + predicted_uncertainties[a] + 1e-6)

            # Ch3: 存在概率
            existence = existence_probs[a]

            # Ch4: RSS 残差（归一化）
            rss_residual = abs(rss_meas - rss_pred[a]) / 5.0

            # 添加边: 测量节点 m -> 锚点节点 (M + a)
            edge_list.append((
                m,  # 源节点 (测量)
                num_measurements + a,  # 目标节点 (锚点)
                [log_prob, std_residual, log_var, existence, rss_residual]
            ))

    # --- 3. 添加 Dustbin 边 ---
    # Dustbin 节点索引: N = M + K
    dustbin_idx = num_measurements + num_anchors

    # 计算 Xi 参考值
    mu_new = undetected_anchors_intensity
    xi_val = np.log(1.0 + mu_new / clutter_intensity)

    for m in range(num_measurements):
        # Dustbin 边特征
        edge_list.append((
            m,  # 源节点 (测量)
            dustbin_idx,  # 目标节点 (dustbin)
            [xi_val, 0.0, 2.0, 1.0, 0.5]  # [Ch0, Ch1, Ch2, Ch3, Ch4]
        ))

    # --- 4. 转换为 PyTorch Geometric 格式 ---
    if len(edge_list) == 0:
        # 没有有效边，返回空图
        num_nodes = num_measurements + num_anchors + 1
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 5), dtype=torch.float32)
    else:
        # 提取边索引和边特征
        src_nodes = [e[0] for e in edge_list]
        dst_nodes = [e[1] for e in edge_list]
        edge_features = [e[2] for e in edge_list]

        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)  # (2, E)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)  # (E, 5)

    # --- 5. 构建节点特征 ---
    # 节点类型: 0=测量, 1=锚点, 2=dustbin
    num_nodes = num_measurements + num_anchors + 1

    # 简单的节点特征: one-hot 编码节点类型
    node_features = torch.zeros((num_nodes, 3), dtype=torch.float32)
    node_features[:num_measurements, 0] = 1.0  # 测量节点
    node_features[num_measurements:num_measurements+num_anchors, 1] = 1.0  # 锚点节点
    node_features[dustbin_idx, 2] = 1.0  # Dustbin 节点

    # 节点类型标签
    node_types = torch.zeros(num_nodes, dtype=torch.long)
    node_types[:num_measurements] = 0  # 测量
    node_types[num_measurements:num_measurements+num_anchors] = 1  # 锚点
    node_types[dustbin_idx] = 2  # Dustbin

    return (node_features, edge_index, edge_attr, node_types,
            num_measurements, num_anchors)


def sparse_to_dense_output(edge_index, edge_probs, num_measurements, num_anchors):
    """
    将稀疏图的输出转换回密集矩阵格式

    参数:
        edge_index: (2, E) 边索引
        edge_probs: (E,) 边的关联概率
        num_measurements: M
        num_anchors: K

    返回:
        assoc_probs: (M, K) 关联概率矩阵
        dustbin_probs: (M,) Dustbin 概率
    """

    # 初始化密集矩阵（默认为 0）
    assoc_probs = np.zeros((num_measurements, num_anchors))
    dustbin_probs = np.zeros(num_measurements)

    dustbin_idx = num_measurements + num_anchors

    # 填充稀疏边的概率
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        prob = edge_probs[i].item()

        if dst == dustbin_idx:
            # Dustbin 边
            dustbin_probs[src] = prob
        elif dst >= num_measurements and dst < dustbin_idx:
            # 锚点边
            anchor_idx = dst - num_measurements
            assoc_probs[src, anchor_idx] = prob

    return assoc_probs, dustbin_probs


def get_graph_statistics(edge_index, num_measurements, num_anchors):
    """
    获取稀疏图的统计信息

    参数:
        edge_index: (2, E) 边索引
        num_measurements: M
        num_anchors: K

    返回:
        stats: 字典，包含统计信息
    """

    num_edges = edge_index.shape[1]
    max_possible_edges = num_measurements * (num_anchors + 1)  # 包括 dustbin
    sparsity = 1.0 - (num_edges / max_possible_edges) if max_possible_edges > 0 else 0.0

    # 统计每个测量节点的度数
    src_nodes = edge_index[0].numpy()
    unique, counts = np.unique(src_nodes, return_counts=True)
    avg_degree = np.mean(counts) if len(counts) > 0 else 0.0
    max_degree = np.max(counts) if len(counts) > 0 else 0
    min_degree = np.min(counts) if len(counts) > 0 else 0

    stats = {
        'num_edges': num_edges,
        'max_possible_edges': max_possible_edges,
        'sparsity': sparsity,
        'avg_degree': avg_degree,
        'max_degree': max_degree,
        'min_degree': min_degree,
        'compression_ratio': max_possible_edges / num_edges if num_edges > 0 else float('inf')
    }

    return stats
