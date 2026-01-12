"""
bp_slam/core/gnn_model_sparse_gat_v2.py
稀疏图版本的 GAT 模型 V2 - ReID 增强版

改进：
1. 边特征 4 维：[Δx, Δy, Σ, ΔF]  (新增 ReID 指纹差异)
2. 节点特征 4 维：
   - 测量节点：[type_emb, RSS_norm, meas_uncertainty, F_instant]
   - 锚点节点：[type_emb, existence_prob, pred_uncertainty, F_history]
3. 删除 Dustbin 节点，杂波判断完全由质量头完成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 检查是否安装了 torch_geometric
try:
    from torch_geometric.nn import GATv2Conv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not installed. Sparse GAT V2 model will not be available.")
    print("Install with: pip install torch-geometric")


class SparseGAT_V2_DualHead(nn.Module):
    """
    稀疏图版本的双头 GAT 模型 V2 - ReID 增强版

    架构:
    - 共享的 GAT 主干
    - 关联头: 预测边的关联概率
    - 质量头: 预测测量节点的质量分数（杂波判断）

    改进:
    - 边特征: 4维 [Δx, Δy, Σ, ΔF]  (新增 ReID)
    - 节点特征: 4维语义特征 (新增指纹)
    - 无Dustbin节点
    """

    def __init__(self, node_dim=4, edge_dim=4, hidden_dim=32, num_layers=2,
                 heads=2, dropout=0.1, use_temporal_gru=False):
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required for Sparse GAT V2 model")

        self.hidden_dim = hidden_dim
        self.heads = heads
        self.use_temporal_gru = use_temporal_gru

        # 1. 节点编码器
        # 输入: 4维语义特征 [type_emb, feature1, feature2, fingerprint]
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 2. 边编码器
        # 输入: 3维边特征 [Δx, Δy, Σ]
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64)
        )

        # 3. GATv2 层
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    edge_dim=64,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False  # 二部图不加自环
                )
            )

        # 4. GRU 时序记忆（可选）
        if use_temporal_gru:
            self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.gru = None

        # 5. 双头解码器

        # 关联头: 预测边的关联概率
        # 输入拼接: [源GNN(32), 源原始(32), 目标GNN(32), 目标原始(32), 边编码(64), 边原始(4), 预计算(3)] = 199维
        # 预计算特征: [dist, dist_score, reid_score] - 直接告诉模型距离和ReID差异
        self.assoc_head = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 64 + edge_dim + 3, 64),  # 32*4 + 64 + 4 + 3 = 199
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 保存edge_dim供forward使用
        self.edge_dim = edge_dim
        
        # 关联头使用默认初始化 (bias=0, 初始预测~50%)
        # 配合 10x 学习率 + 动态加权BCE 学习

        # 质量头: 预测测量节点的质量分数（杂波判断）
        # 输入拼接: [测量GNN(32), 测量原始(32)] = 64维
        # 拼接升维后的x_raw，而不是原始4维，数值量级更匹配
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        print(f"✓ 稀疏 GAT V2 双头模型初始化完成 (简化版)")
        print(f"  - 节点特征: 4维 → 升维至 {hidden_dim}维")
        print(f"  - 边特征: 4维 [Δx, Δy, Σ, ΔF]")
        print(f"  - GAT: {num_layers}层, {heads}头")
        print(f"  - 关联头输入: {hidden_dim*4 + 64 + edge_dim + 3}维")
        print(f"  - 质量头输入: {hidden_dim*2}维")
        print(f"  - 注意力头数: {heads}")

    def forward(self, node_features, edge_index, edge_attr, node_types,
                num_measurements, hidden_state=None):
        """
        前向传播 (残差连接 + 输入拼接版)

        参数:
            node_features: (N, 4) 节点特征 [type, RSS_norm, uncertainty, fingerprint]
            edge_index: (2, E) 边索引
            edge_attr: (E, 4) 边特征 [Δx, Δy, Σ, ΔF]
            node_types: (N,) 节点类型 (0=测量, 1=锚点)
            num_measurements: M
            hidden_state: (N, hidden_dim) 上一帧的隐状态

        返回:
            edge_logits: (E,) 边的关联 logits
            quality_logits: (M,) 测量节点的质量 logits
            new_hidden_state: (N, hidden_dim) 更新后的隐状态
        """

        # 1. 【升维】先将 4维 → 64维，作为残差和拼接的基准
        x_raw = self.node_encoder(node_features)  # (N, 64)
        edge_emb = self.edge_encoder(edge_attr)   # (E, 64)
        
        # x 是在网络中流动的特征
        x = x_raw

        # 2. 【GAT 循环 + 残差连接】
        for gat_layer in self.gat_layers:
            # 计算这一层的更新量
            x_delta = gat_layer(x, edge_index, edge_attr=edge_emb)
            x_delta = F.elu(x_delta)
            # 标准残差连接: x 和 x_delta 都是 64维，直接相加
            x = x + x_delta

        # 3. GRU（可选）
        if self.use_temporal_gru:
            if hidden_state is None:
                hidden_state = torch.zeros_like(x)
            new_hidden_state = self.gru(x, hidden_state)
            x = new_hidden_state
        else:
            new_hidden_state = x

        # 4. 【关联头 - 输入拼接】
        # 拼接: [源GNN, 源原始, 目标GNN, 目标原始, 边编码, 边原始, 预计算特征]
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        # 预计算关键特征，让模型不需要自己学
        delta_x = edge_attr[:, 0]  # (E,)
        delta_y = edge_attr[:, 1]  # (E,)
        delta_f = edge_attr[:, 3]  # (E,) ReID差异
        
        # 距离: sqrt(Δx² + Δy²)，正样本≈0，负样本≈1
        dist = torch.sqrt(delta_x**2 + delta_y**2 + 1e-6)  # (E,)
        
        # 距离得分: 距离越小越好，用负指数变换到[0,1]
        dist_score = torch.exp(-dist)  # (E,) 正样本≈1，负样本≈0.3
        
        # ReID得分: ΔF越小越好
        reid_score = torch.exp(-delta_f)  # (E,) 正样本≈0.8，负样本≈0.5
        
        # 拼接预计算特征
        precomputed = torch.stack([dist, dist_score, reid_score], dim=-1)  # (E, 3)
        
        edge_input = torch.cat([
            x[src_nodes],       x_raw[src_nodes],
            x[dst_nodes],       x_raw[dst_nodes],
            edge_emb,           edge_attr,
            precomputed  # 新增预计算特征
        ], dim=-1)
        edge_logits = self.assoc_head(edge_input).squeeze(-1)  # (E,)

        # 5. 【质量头 - 输入拼接】
        # 拼接: [测量GNN, 测量原始]
        # 维度: 64 + 64 = 128
        # 即使 GAT 把特征平滑了，这里也能看到升维后的原始 RSS/fingerprint 特征
        meas_features_gnn = x[:num_measurements]
        meas_features_raw = x_raw[:num_measurements]
        meas_input = torch.cat([meas_features_gnn, meas_features_raw], dim=-1)
        quality_logits = self.quality_head(meas_input).squeeze(-1)  # (M,)

        return edge_logits, quality_logits, new_hidden_state
