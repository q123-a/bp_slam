"""
bp_slam/core/gnn_model_sparse_gat_v2.py
稀疏图版本的 GAT 模型 V2 - 简化版

改进：
1. 边特征从 5 维简化为 3 维：[Δx, Δy, Σ]
2. 节点特征从 one-hot 改为语义特征：
   - 测量节点：[type_emb, RSS_norm, meas_uncertainty]
   - 锚点节点：[type_emb, existence_prob, pred_uncertainty]
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
    稀疏图版本的双头 GAT 模型 V2

    架构:
    - 共享的 GAT 主干
    - 关联头: 预测边的关联概率
    - 质量头: 预测测量节点的质量分数（杂波判断）

    改进:
    - 边特征: 3维 [Δx, Δy, Σ]
    - 节点特征: 3维语义特征
    - 无Dustbin节点
    """

    def __init__(self, node_dim=3, edge_dim=3, hidden_dim=64, num_layers=2,
                 heads=4, dropout=0.1, use_temporal_gru=False):
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required for Sparse GAT V2 model")

        self.hidden_dim = hidden_dim
        self.heads = heads
        self.use_temporal_gru = use_temporal_gru

        # 1. 节点编码器
        # 输入: 3维语义特征 [type_emb, feature1, feature2]
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
        self.assoc_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # 质量头: 预测测量节点的质量分数（杂波判断）
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        print(f"✓ 稀疏 GAT V2 双头模型初始化完成")
        print(f"  - 边特征: 3维 [Δx, Δy, Σ]")
        print(f"  - 节点特征: 3维语义特征")
        print(f"  - 无Dustbin节点")
        print(f"  - 注意力头数: {heads}")

    def forward(self, node_features, edge_index, edge_attr, node_types,
                num_measurements, hidden_state=None):
        """
        前向传播

        参数:
            node_features: (N, 3) 节点特征
            edge_index: (2, E) 边索引
            edge_attr: (E, 3) 边特征 [Δx, Δy, Σ]
            node_types: (N,) 节点类型 (0=测量, 1=锚点)
            num_measurements: M
            hidden_state: (N, hidden_dim) 上一帧的隐状态

        返回:
            edge_logits: (E,) 边的关联 logits
            quality_logits: (M,) 测量节点的质量 logits
            new_hidden_state: (N, hidden_dim) 更新后的隐状态
        """

        # 1. 编码
        x = self.node_encoder(node_features)  # (N, hidden_dim)
        edge_emb = self.edge_encoder(edge_attr)  # (E, 64)

        # 2. GAT 消息传递
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, edge_attr=edge_emb)
            x = F.elu(x)

        # 3. GRU（可选）
        if self.use_temporal_gru:
            if hidden_state is None:
                hidden_state = torch.zeros_like(x)
            new_hidden_state = self.gru(x, hidden_state)
            x = new_hidden_state
        else:
            new_hidden_state = x

        # 4. 关联头: 预测边的关联概率
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        src_features = x[src_nodes]
        dst_features = x[dst_nodes]
        edge_input = torch.cat([src_features, dst_features, edge_emb], dim=-1)
        edge_logits = self.assoc_head(edge_input).squeeze(-1)  # (E,)

        # 5. 质量头: 预测测量节点的质量分数
        # 只对测量节点（前 M 个节点）
        meas_features = x[:num_measurements]  # (M, hidden_dim)
        quality_logits = self.quality_head(meas_features).squeeze(-1)  # (M,)

        return edge_logits, quality_logits, new_hidden_state
