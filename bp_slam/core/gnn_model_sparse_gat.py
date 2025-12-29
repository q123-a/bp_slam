"""
bp_slam/core/gnn_model_sparse_gat.py
稀疏图版本的 GAT 模型

使用 torch_geometric 的 GATv2Conv 实现真正的稀疏图注意力网络
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
    print("Warning: torch_geometric not installed. Sparse GAT model will not be available.")
    print("Install with: pip install torch-geometric")


class SparseGAT_SLAM(nn.Module):
    """
    稀疏图版本的 GAT 模型

    架构:
    1. Node Encoder: 节点类型 one-hot -> 节点嵌入
    2. Edge Encoder: 5维边特征 -> 边嵌入
    3. GATv2 Layers: 多层注意力消息传递
    4. GRU (可选): 时序记忆
    5. Edge Decoder: 预测边的关联概率
    """

    def __init__(self, node_dim=3, edge_dim=5, hidden_dim=64, num_layers=2,
                 heads=4, dropout=0.1, use_temporal_gru=False):
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required for Sparse GAT model")

        self.hidden_dim = hidden_dim
        self.heads = heads
        self.use_temporal_gru = use_temporal_gru

        # 1. 节点编码器
        # 输入: one-hot 节点类型 (3维: 测量/锚点/dustbin)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 2. 边编码器
        # 输入: 5维边特征 [log_prob, std_residual, log_var, existence, rss_residual]
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
            if i == 0:
                # 第一层: hidden_dim -> hidden_dim
                in_channels = hidden_dim
            else:
                # 后续层: hidden_dim -> hidden_dim
                in_channels = hidden_dim

            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    edge_dim=64,  # 边嵌入维度
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

        # 5. 边解码器
        # 输入: [src_node, dst_node, edge_emb]
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # 输出边的 logit
        )

        print(f"✓ 稀疏 GAT 模型初始化完成")
        print(f"  - 隐藏维度: {hidden_dim}")
        print(f"  - GAT 层数: {num_layers}")
        print(f"  - 注意力头数: {heads}")
        print(f"  - 使用 GRU: {use_temporal_gru}")

    def forward(self, node_features, edge_index, edge_attr, hidden_state=None):
        """
        前向传播

        参数:
            node_features: (N, 3) 节点特征 (one-hot 类型)
            edge_index: (2, E) 边索引
            edge_attr: (E, 5) 边特征
            hidden_state: (N, hidden_dim) 上一帧的节点隐状态

        返回:
            edge_logits: (E,) 边的关联 logits
            new_hidden_state: (N, hidden_dim) 更新后的节点隐状态
        """

        # 1. 编码节点和边
        x = self.node_encoder(node_features)  # (N, hidden_dim)
        edge_emb = self.edge_encoder(edge_attr)  # (E, 64)

        # 2. GAT 消息传递
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, edge_attr=edge_emb)
            x = F.elu(x)

        # 3. GRU 时序更新（可选）
        if self.use_temporal_gru:
            if hidden_state is None:
                hidden_state = torch.zeros_like(x)

            # 逐节点更新
            new_hidden_state = self.gru(x, hidden_state)
            x = new_hidden_state
        else:
            new_hidden_state = x

        # 4. 边解码
        # 提取每条边的源节点和目标节点特征
        src_nodes = edge_index[0]  # (E,)
        dst_nodes = edge_index[1]  # (E,)

        src_features = x[src_nodes]  # (E, hidden_dim)
        dst_features = x[dst_nodes]  # (E, hidden_dim)

        # 拼接: [src, dst, edge_emb]
        edge_input = torch.cat([src_features, dst_features, edge_emb], dim=-1)  # (E, 2*hidden_dim + 64)

        # 预测边的 logit
        edge_logits = self.edge_decoder(edge_input).squeeze(-1)  # (E,)

        return edge_logits, new_hidden_state


class SparseGAT_DualHead(nn.Module):
    """
    稀疏图版本的双头 GAT 模型

    架构:
    - 共享的 GAT 主干
    - 关联头: 预测边的关联概率
    - 质量头: 预测测量节点的质量分数
    """

    def __init__(self, node_dim=3, edge_dim=5, hidden_dim=64, num_layers=2,
                 heads=4, dropout=0.1, use_temporal_gru=False):
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required for Sparse GAT model")

        self.hidden_dim = hidden_dim
        self.heads = heads
        self.use_temporal_gru = use_temporal_gru

        # 共享的编码器和 GAT 层
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64)
        )

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
                    add_self_loops=False
                )
            )

        # GRU（可选）
        if use_temporal_gru:
            self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.gru = None

        # 双头解码器

        # 关联头: 预测边的关联概率
        self.assoc_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # 质量头: 预测测量节点的质量分数
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        print(f"✓ 稀疏 GAT 双头模型初始化完成")
        print(f"  - 架构: 质量头 + 关联头")
        print(f"  - 注意力头数: {heads}")

    def forward(self, node_features, edge_index, edge_attr, node_types,
                num_measurements, hidden_state=None):
        """
        前向传播

        参数:
            node_features: (N, 3) 节点特征
            edge_index: (2, E) 边索引
            edge_attr: (E, 5) 边特征
            node_types: (N,) 节点类型 (0=测量, 1=锚点, 2=dustbin)
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
