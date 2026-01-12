"""
bp_slam/core/graph_builder_v2.py
稀疏图构建工具 V2 - ReID 增强版

改进：
1. 删除 Dustbin 节点和边
2. 边特征 4 维：[Δx, Δy, Σ, ΔF]  (新增 ReID 指纹差异)
3. 节点特征 4 维：
   - 测量节点：[type_emb, RSS_norm, meas_uncertainty, F_instant]
   - 锚点节点：[type_emb, existence_prob, pred_uncertainty, F_history]
"""

import numpy as np
import torch
from .reid_utils import (compute_fingerprint, normalize_fingerprint, 
                         compute_delta_f_norm, FINGERPRINT_CENTER, FINGERPRINT_SCALE)


def build_sparse_graph_v2(filtered_measurements, predicted_measurements, predicted_uncertainties,
                          existence_probs, predicted_particles_agent, predicted_particles_anchors,
                          weights_anchor, P_tx=15.41, n=2.0,
                          distance_threshold=None, beta_threshold=None,
                          reid_fingerprints=None):
    """
    构建简化的稀疏图结构（V2版本 - ReID 增强）

    参数:
        filtered_measurements: (3, M) [距离, 方差, RSS]
        predicted_measurements: (K,) 预测距离均值
        predicted_uncertainties: (K,) 预测距离方差
        existence_probs: (K,) 锚点存在概率
        predicted_particles_agent: (4, num_particles) 移动体粒子
        predicted_particles_anchors: (2, num_particles, K) 锚点粒子
        weights_anchor: (num_particles, K) 锚点权重
        P_tx: 发射功率
        n: 路径损耗指数
        distance_threshold: 距离阈值（用于过滤边），None 表示不过滤
        beta_threshold: Beta 阈值（用于过滤边），None 表示不过滤
        reid_fingerprints: (K,) 锚点历史指纹列表，元素可为 None

    返回:
        node_features: (N, 4) 节点特征，N = M + K（无Dustbin）
        edge_index: (2, E) 边索引
        edge_attr: (E, 4) 边特征 [Δx, Δy, Σ, ΔF]
        node_types: (N,) 节点类型 (0=测量, 1=锚点)
        num_measurements: M
        num_anchors: K
    """

    num_measurements = filtered_measurements.shape[1]
    num_anchors = len(predicted_measurements)
    num_particles = predicted_particles_agent.shape[1]

    # --- 1. 计算移动体的加权平均位置 ---
    agent_x = np.mean(predicted_particles_agent[0, :])
    agent_y = np.mean(predicted_particles_agent[1, :])

    # --- 2. 计算每个锚点的加权平均位置 ---
    anchor_positions_mean = np.zeros((2, num_anchors))
    for k in range(num_anchors):
        weight_sum = np.sum(weights_anchor[:, k])
        if weight_sum > 0:
            anchor_positions_mean[0, k] = np.dot(
                predicted_particles_anchors[0, :, k],
                weights_anchor[:, k]
            ) / weight_sum
            anchor_positions_mean[1, k] = np.dot(
                predicted_particles_anchors[1, :, k],
                weights_anchor[:, k]
            ) / weight_sum

    # --- 3. 预计算锚点的预测 RSS ---
    rss_pred = np.zeros(num_anchors)
    for k in range(num_anchors):
        safe_pred_dist = max(predicted_measurements[k], 0.1)
        rss_pred[k] = P_tx - 10 * n * np.log10(safe_pred_dist)

    # --- 4. 归一化 RSS（用于节点特征）---
    # 统计所有测量的 RSS 均值和标准差
    rss_measurements = []
    for m in range(num_measurements):
        if filtered_measurements.shape[0] >= 3:
            rss_measurements.append(filtered_measurements[2, m])
        else:
            rss_measurements.append(0.0)

    rss_measurements = np.array(rss_measurements)

    # [修改] 使用固定的归一化参数（RSS 已经是 dBm 单位）
    # 根据 testbed.py 的杂波生成逻辑（修改后）：
    # - 真实信号 RSS: -13 ~ 1 dBm（平均 -6 dBm）
    # - 杂波 RSS: -60 ~ -30 dBm（平均 -45 dBm）
    # - 整体范围: -60 ~ 1 dBm，中心约 -30 dBm
    rss_mean = -30.0  # 设置为整体数据的中心
    rss_std = 20.0    # 覆盖 ±3σ 范围（-30 ± 60 = [-90, 30]）

    # 归一化 RSS
    rss_norm = (rss_measurements - rss_mean) / rss_std

    # --- 5. 预计算测量的瞬时指纹 ---
    meas_fingerprints = np.zeros(num_measurements)
    for m in range(num_measurements):
        meas_dist = filtered_measurements[0, m]
        if filtered_measurements.shape[0] >= 3:
            meas_rss = filtered_measurements[2, m]
        else:
            meas_rss = 0.0
        meas_fingerprints[m] = compute_fingerprint(meas_rss, meas_dist)

    # --- 6. 构建边列表（稀疏，无 Dustbin）---
    edge_list = []  # [(src, dst, features), ...]

    for m in range(num_measurements):
        meas_dist = filtered_measurements[0, m]
        meas_var = filtered_measurements[1, m]
        F_instant = meas_fingerprints[m]  # 瞬时指纹

        for k in range(num_anchors):
            # 计算预测的 2D 向量（从移动体到锚点）
            delta_x_pred = anchor_positions_mean[0, k] - agent_x
            delta_y_pred = anchor_positions_mean[1, k] - agent_y
            pred_dist = np.sqrt(delta_x_pred**2 + delta_y_pred**2)
            pred_dist = max(pred_dist, 0.1)  # 避免除零

            # 计算距离残差
            delta_d = meas_dist - predicted_measurements[k]

            # 过滤条件 1: 距离阈值
            if distance_threshold is not None and abs(delta_d) > distance_threshold:
                continue

            # 过滤条件 2: Beta 阈值（简化版：基于标准化残差）
            sigma_combined = np.sqrt(meas_var + predicted_uncertainties[k])
            std_residual = abs(delta_d) / (sigma_combined + 1e-6)
            if beta_threshold is not None and std_residual > 3.0:  # 3-sigma 规则
                continue

            # 将 1D 残差投影到 2D（沿预测方向）
            delta_x = delta_d * (delta_x_pred / pred_dist)
            delta_y = delta_d * (delta_y_pred / pred_dist)

            # [修改] 手动缩放 Δx 和 Δy - 使用更小的分母放大差异
            scale_factor = 3.0
            delta_x_norm = delta_x / scale_factor
            delta_y_norm = delta_y / scale_factor

            # 防止极远的点爆表，clamp 到 [-2, 2]
            delta_x_norm = np.clip(delta_x_norm, -2.0, 2.0)
            delta_y_norm = np.clip(delta_y_norm, -2.0, 2.0)

            # 综合不确定度（也进行缩放，假设最大不确定度 5.0）
            max_sigma = 5.0
            sigma_norm = sigma_combined / max_sigma

            # [ReID] 计算指纹差异
            F_history = reid_fingerprints[k] if reid_fingerprints is not None else None
            delta_F_norm = compute_delta_f_norm(F_instant, F_history)

            # 添加边: 测量节点 m -> 锚点节点 (M + k)
            edge_list.append((
                m,  # 源节点 (测量)
                num_measurements + k,  # 目标节点 (锚点)
                [delta_x_norm, delta_y_norm, sigma_norm, delta_F_norm]  # 边特征 [Δx, Δy, Σ, ΔF]
            ))

    # --- 7. 转换为 PyTorch Geometric 格式 ---
    if len(edge_list) == 0:
        # 没有有效边，返回空图
        num_nodes = num_measurements + num_anchors
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.float32)  # 4维边特征
    else:
        # 提取边索引和边特征
        src_nodes = [e[0] for e in edge_list]
        dst_nodes = [e[1] for e in edge_list]
        edge_features = [e[2] for e in edge_list]

        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)  # (2, E)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)  # (E, 4)

    # --- 8. 构建节点特征 ---
    # 节点类型: 0=测量, 1=锚点（无 Dustbin）
    num_nodes = num_measurements + num_anchors

    # 节点特征维度：4维
    # - 测量节点: [type_emb=1.0, RSS_norm, meas_uncertainty, F_instant]
    # - 锚点节点: [type_emb=0.0, existence_prob, pred_uncertainty, F_history]
    node_features = torch.zeros((num_nodes, 4), dtype=torch.float32)

    # 测量节点特征：[type_emb=1.0, RSS_norm, meas_uncertainty, F_instant]
    for m in range(num_measurements):
        node_features[m, 0] = 1.0  # type_emb (测量节点)
        node_features[m, 1] = rss_norm[m]  # RSS归一化
        node_features[m, 2] = filtered_measurements[1, m]  # 测量不确定度
        node_features[m, 3] = normalize_fingerprint(meas_fingerprints[m])  # [ReID] 瞬时指纹

    # 锚点节点特征：[type_emb=0.0, existence_prob, pred_uncertainty, F_history]
    for k in range(num_anchors):
        node_features[num_measurements + k, 0] = 0.0  # type_emb (锚点节点)
        node_features[num_measurements + k, 1] = existence_probs[k]  # 存在概率
        node_features[num_measurements + k, 2] = predicted_uncertainties[k]  # 预测不确定度
        # [ReID] 历史指纹
        F_history = reid_fingerprints[k] if reid_fingerprints is not None else None
        node_features[num_measurements + k, 3] = normalize_fingerprint(F_history)

    # 节点类型标签
    node_types = torch.zeros(num_nodes, dtype=torch.long)
    node_types[:num_measurements] = 0  # 测量
    node_types[num_measurements:] = 1  # 锚点

    # [DEBUG] 打印节点特征统计信息
    print(f"\n[节点特征归一化检查]")
    print(f"  节点总数: {num_nodes} (测量: {num_measurements}, 锚点: {num_anchors})")
    if num_measurements > 0:
        meas_rss = node_features[:num_measurements, 1].numpy()
        meas_unc = node_features[:num_measurements, 2].numpy()
        meas_fp = node_features[:num_measurements, 3].numpy()
        print(f"  测量节点 RSS 归一化范围: [{np.min(meas_rss):.4f}, {np.max(meas_rss):.4f}]")
        print(f"  测量节点不确定度范围: [{np.min(meas_unc):.4f}, {np.max(meas_unc):.4f}]")
        print(f"  测量节点 F_instant 范围: [{np.min(meas_fp):.4f}, {np.max(meas_fp):.4f}]")

        if np.max(np.abs(meas_rss)) > 5.0:
            print(f"  ⚠️  警告: 测量 RSS 归一化值超出 [-5, 5]！")
        if np.max(meas_unc) > 10.0:
            print(f"  ⚠️  警告: 测量不确定度超出 10.0！")

    if num_anchors > 0:
        anchor_exist = node_features[num_measurements:, 1].numpy()
        anchor_unc = node_features[num_measurements:, 2].numpy()
        anchor_fp = node_features[num_measurements:, 3].numpy()
        print(f"  锚点节点存在概率范围: [{np.min(anchor_exist):.4f}, {np.max(anchor_exist):.4f}]")
        print(f"  锚点节点不确定度范围: [{np.min(anchor_unc):.4f}, {np.max(anchor_unc):.4f}]")
        print(f"  锚点节点 F_history 范围: [{np.min(anchor_fp):.4f}, {np.max(anchor_fp):.4f}]")

        if np.max(anchor_unc) > 10.0:
            print(f"  ⚠️  警告: 锚点不确定度超出 10.0！")

    return (node_features, edge_index, edge_attr, node_types,
            num_measurements, num_anchors)


def sparse_to_dense_output_v2(edge_index, edge_probs, num_measurements, num_anchors):
    """
    将稀疏图的输出转换回密集矩阵格式（V2版本，无Dustbin）

    参数:
        edge_index: (2, E) 边索引
        edge_probs: (E,) 边的关联概率
        num_measurements: M
        num_anchors: K

    返回:
        assoc_probs: (M, K) 关联概率矩阵
        dustbin_probs: (M,) Dustbin 概率（由质量头单独计算）
    """

    # 初始化密集矩阵（默认为 0）
    assoc_probs = np.zeros((num_measurements, num_anchors))

    # 填充稀疏边的概率
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        prob = edge_probs[i].item()

        # 所有边都是锚点边（无Dustbin）
        if dst >= num_measurements and dst < num_measurements + num_anchors:
            anchor_idx = dst - num_measurements
            assoc_probs[src, anchor_idx] = prob

    # Dustbin 概率由质量头单独计算，这里返回占位符
    dustbin_probs = np.zeros(num_measurements)

    return assoc_probs, dustbin_probs


def get_graph_statistics_v2(edge_index, num_measurements, num_anchors):
    """
    获取稀疏图的统计信息（V2版本）

    参数:
        edge_index: (2, E) 边索引
        num_measurements: M
        num_anchors: K

    返回:
        stats: 字典，包含统计信息
    """

    num_edges = edge_index.shape[1]
    max_possible_edges = num_measurements * num_anchors  # 无Dustbin
    sparsity = 1.0 - (num_edges / max_possible_edges) if max_possible_edges > 0 else 0.0

    # 统计每个测量节点的度数
    if num_edges > 0:
        src_nodes = edge_index[0].numpy()
        unique, counts = np.unique(src_nodes, return_counts=True)
        avg_degree = np.mean(counts) if len(counts) > 0 else 0.0
        max_degree = np.max(counts) if len(counts) > 0 else 0
        min_degree = np.min(counts) if len(counts) > 0 else 0
    else:
        avg_degree = 0.0
        max_degree = 0
        min_degree = 0

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
