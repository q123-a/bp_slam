"""
bp_slam/core/graph_attention.py
时空图注意力模块 + 自适应RANSAC后处理

Author: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatioTemporalGraphAttention(nn.Module):
    """
    时空图注意力模块

    功能：
    1. 空间注意力：建模同一时刻不同测量之间的关系
    2. 时间注意力：建模相邻时刻测量之间的关系
    3. 杂波检测：通过注意力权重识别孤立的杂波测量
    """

    def __init__(self, feature_dim=64, num_heads=4, dropout=0.1, temporal_window=3):
        """
        参数:
            feature_dim: 特征维度
            num_heads: 多头注意力的头数
            dropout: Dropout 比率
            temporal_window: 时间窗口大小（保留多少历史帧）
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.temporal_window = temporal_window

        assert feature_dim % num_heads == 0, "feature_dim 必须能被 num_heads 整除"

        # 空间注意力：测量之间的关系
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 时间注意力：跨帧的关系
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 杂波检测头
        self.clutter_detector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )

        # 历史帧缓存
        self.history_buffer = []

    def forward(self, features, positions=None):
        """
        前向传播

        参数:
            features: (B, M, feature_dim) 测量特征
            positions: (B, M, 2) 测量位置（可选，用于位置编码）

        返回:
            enhanced_features: (B, M, feature_dim) 增强后的特征
            attention_scores: (B, M, M) 空间注意力权重
            clutter_scores: (B, M) 杂波概率 [0, 1]
        """
        B, M, D = features.shape

        # 1. 空间注意力：同一时刻测量之间的关系
        spatial_out, spatial_attn = self.spatial_attention(
            features, features, features,
            need_weights=True,
            average_attn_weights=True
        )

        # 2. 时间注意力：与历史帧的关系
        if len(self.history_buffer) > 0:
            # 拼接历史帧
            history_features = torch.cat(self.history_buffer, dim=1)  # (B, M*T, D)

            temporal_out, temporal_attn = self.temporal_attention(
                features, history_features, history_features,
                need_weights=True,
                average_attn_weights=False
            )
        else:
            # 没有历史帧时，使用零向量
            temporal_out = torch.zeros_like(features)

        # 3. 特征融合
        combined = torch.cat([spatial_out, temporal_out], dim=-1)  # (B, M, 2*D)
        enhanced_features = self.fusion(combined)  # (B, M, D)

        # 残差连接
        enhanced_features = enhanced_features + features

        # 4. 杂波检测
        clutter_scores = self.clutter_detector(enhanced_features).squeeze(-1)  # (B, M)

        # 5. 更新历史缓存
        self._update_history(features.detach())

        return enhanced_features, spatial_attn, clutter_scores

    def _update_history(self, current_features):
        """更新历史帧缓存"""
        self.history_buffer.append(current_features)

        # 保持窗口大小
        if len(self.history_buffer) > self.temporal_window:
            self.history_buffer.pop(0)

    def reset_history(self):
        """重置历史缓存（新序列开始时调用）"""
        self.history_buffer = []


class AdaptiveRANSAC:
    """
    自适应RANSAC后处理模块

    功能：
    1. 根据杂波率自适应调整迭代次数
    2. 使用图注意力权重作为先验
    3. 鲁棒地过滤杂波测量
    """

    def __init__(self,
                 distance_threshold=2.0,
                 min_inliers=3,
                 max_iterations=100,
                 confidence=0.99,
                 enable_adaptive=True):
        """
        参数:
            distance_threshold: 内点距离阈值（米）
            min_inliers: 最小内点数量
            max_iterations: 最大迭代次数
            confidence: 置信度（用于计算自适应迭代次数）
            enable_adaptive: 是否启用自适应模式
        """
        self.distance_threshold = distance_threshold
        self.min_inliers = min_inliers
        self.max_iterations = max_iterations
        self.confidence = confidence
        self.enable_adaptive = enable_adaptive

    def estimate_clutter_ratio(self, clutter_scores):
        """
        估计杂波率

        参数:
            clutter_scores: (M,) 杂波概率

        返回:
            clutter_ratio: float, 估计的杂波率 [0, 1]
        """
        # 使用阈值 0.5 判断杂波
        estimated_clutter = (clutter_scores > 0.5).float().mean().item()
        return estimated_clutter

    def compute_adaptive_iterations(self, clutter_ratio, num_samples=3):
        """
        计算自适应迭代次数

        公式: N = log(1 - confidence) / log(1 - (1 - clutter_ratio)^num_samples)

        参数:
            clutter_ratio: 杂波率
            num_samples: 每次采样的点数

        返回:
            num_iterations: int, 迭代次数
        """
        if clutter_ratio < 0.05:
            # 杂波率 < 5%：几乎无杂波，只需少量迭代
            return 2
        elif clutter_ratio > 0.95:
            # 杂波率 > 95%：几乎全是杂波，无法拟合
            return 0

        # 标准RANSAC迭代次数公式
        inlier_ratio = 1.0 - clutter_ratio
        prob_all_inliers = inlier_ratio ** num_samples

        if prob_all_inliers < 1e-10:
            return self.max_iterations

        num_iterations = np.log(1 - self.confidence) / np.log(1 - prob_all_inliers)
        num_iterations = int(np.ceil(num_iterations))

        # 限制在合理范围内
        num_iterations = max(2, min(num_iterations, self.max_iterations))

        return num_iterations

    def filter_measurements(self,
                           measurements,
                           predicted_measurements,
                           clutter_scores=None,
                           attention_weights=None):
        """
        使用RANSAC过滤杂波测量

        参数:
            measurements: (M,) 测量距离
            predicted_measurements: (K,) 预测距离
            clutter_scores: (M,) 杂波概率（可选）
            attention_weights: (M, M) 注意力权重（可选）

        返回:
            inlier_mask: (M,) bool tensor, True表示内点
            best_model: dict, 最佳模型参数
        """
        M = measurements.shape[0]
        K = predicted_measurements.shape[0]

        # 如果测量太少，直接返回全部
        if M < self.min_inliers:
            return torch.ones(M, dtype=torch.bool, device=measurements.device), None

        # 估计杂波率
        if clutter_scores is not None and self.enable_adaptive:
            clutter_ratio = self.estimate_clutter_ratio(clutter_scores)
            num_iterations = self.compute_adaptive_iterations(clutter_ratio)
        else:
            clutter_ratio = 0.3  # 默认假设30%杂波率
            num_iterations = self.max_iterations

        # 如果杂波率极低，跳过RANSAC
        if num_iterations <= 2:
            # 使用简单的距离阈值过滤
            distances = torch.abs(measurements.unsqueeze(1) - predicted_measurements.unsqueeze(0))
            min_distances, _ = torch.min(distances, dim=1)
            inlier_mask = min_distances < self.distance_threshold
            return inlier_mask, {'method': 'threshold', 'clutter_ratio': clutter_ratio}

        # 执行RANSAC
        best_inliers = None
        best_num_inliers = 0
        best_model = None

        # 计算采样权重（使用注意力权重和杂波分数）
        if clutter_scores is not None:
            # 内点概率 = 1 - 杂波概率
            sample_weights = 1.0 - clutter_scores.cpu().numpy()
            sample_weights = np.clip(sample_weights, 0.01, 1.0)
            sample_weights = sample_weights / sample_weights.sum()
        else:
            sample_weights = np.ones(M) / M

        for iteration in range(num_iterations):
            # 随机采样（使用加权采样）
            try:
                sample_indices = np.random.choice(
                    M,
                    size=min(self.min_inliers, M),
                    replace=False,
                    p=sample_weights
                )
            except:
                # 如果加权采样失败，使用均匀采样
                sample_indices = np.random.choice(M, size=min(self.min_inliers, M), replace=False)

            sample_measurements = measurements[sample_indices]

            # 拟合模型：找到与采样测量最匹配的锚点子集
            distances = torch.abs(sample_measurements.unsqueeze(1) - predicted_measurements.unsqueeze(0))
            min_distances, matched_anchors = torch.min(distances, dim=1)

            # 检查采样是否有效（距离在阈值内）
            if torch.all(min_distances < self.distance_threshold):
                # 使用这个模型评估所有测量
                all_distances = torch.abs(measurements.unsqueeze(1) - predicted_measurements.unsqueeze(0))
                min_all_distances, _ = torch.min(all_distances, dim=1)

                current_inliers = min_all_distances < self.distance_threshold
                num_inliers = current_inliers.sum().item()

                # 更新最佳模型
                if num_inliers > best_num_inliers:
                    best_num_inliers = num_inliers
                    best_inliers = current_inliers
                    best_model = {
                        'matched_anchors': matched_anchors,
                        'num_inliers': num_inliers,
                        'clutter_ratio': clutter_ratio,
                        'iterations': iteration + 1
                    }

        # 如果没有找到有效模型，使用简单阈值
        if best_inliers is None:
            distances = torch.abs(measurements.unsqueeze(1) - predicted_measurements.unsqueeze(0))
            min_distances, _ = torch.min(distances, dim=1)
            best_inliers = min_distances < self.distance_threshold
            best_model = {'method': 'fallback', 'clutter_ratio': clutter_ratio}

        return best_inliers, best_model


class HybridDataAssociation(nn.Module):
    """
    混合数据关联模块：图注意力 + 自适应RANSAC

    这是一个完整的端到端模块，集成了：
    1. 时空图注意力（自监督学习）
    2. 自适应RANSAC后处理（经典鲁棒估计）
    """

    def __init__(self,
                 feature_dim=64,
                 num_heads=4,
                 dropout=0.1,
                 temporal_window=3,
                 ransac_threshold=2.0,
                 enable_ransac=True):
        """
        参数:
            feature_dim: 特征维度
            num_heads: 注意力头数
            dropout: Dropout比率
            temporal_window: 时间窗口大小
            ransac_threshold: RANSAC距离阈值
            enable_ransac: 是否启用RANSAC后处理
        """
        super().__init__()

        self.graph_attention = SpatioTemporalGraphAttention(
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            temporal_window=temporal_window
        )

        self.ransac = AdaptiveRANSAC(
            distance_threshold=ransac_threshold,
            enable_adaptive=True
        )

        self.enable_ransac = enable_ransac

    def forward(self, features, measurements, predicted_measurements):
        """
        前向传播

        参数:
            features: (B, M, feature_dim) 测量特征
            measurements: (B, M) 测量距离
            predicted_measurements: (B, K) 预测距离

        返回:
            enhanced_features: (B, M, feature_dim) 增强特征
            inlier_mask: (B, M) 内点掩码
            attention_scores: (B, M, M) 注意力权重
            clutter_scores: (B, M) 杂波概率
            ransac_info: dict, RANSAC统计信息
        """
        B, M, D = features.shape

        # Stage 1: 图注意力（自监督学习）
        enhanced_features, attention_scores, clutter_scores = self.graph_attention(features)

        # Stage 2: 自适应RANSAC（可选）
        if self.enable_ransac:
            inlier_masks = []
            ransac_infos = []

            for b in range(B):
                inlier_mask, ransac_info = self.ransac.filter_measurements(
                    measurements[b],
                    predicted_measurements[b],
                    clutter_scores=clutter_scores[b],
                    attention_weights=attention_scores[b]
                )
                inlier_masks.append(inlier_mask)
                ransac_infos.append(ransac_info)

            inlier_mask = torch.stack(inlier_masks, dim=0)  # (B, M)
            ransac_info = ransac_infos[0] if len(ransac_infos) > 0 else None
        else:
            # 不使用RANSAC，所有测量都是内点
            inlier_mask = torch.ones(B, M, dtype=torch.bool, device=features.device)
            ransac_info = {'method': 'disabled'}

        return enhanced_features, inlier_mask, attention_scores, clutter_scores, ransac_info

    def reset(self):
        """重置历史状态（新序列开始时调用）"""
        self.graph_attention.reset_history()
