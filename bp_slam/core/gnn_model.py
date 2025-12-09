"""
bp_slam/core/gnn_model.py
混合驱动 FGNN 模型定义 - 简化版
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FactorGraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, use_temporal_gru=False, use_layer_gru=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_temporal_gru = use_temporal_gru  # 是否使用跨帧GRU
        self.use_layer_gru = use_layer_gru        # 是否使用层内GRU

        # 1. 特征融合编码器 (Physics-Data Fusion)
        # 将 (LogProb, Residual, Variance, Existence, Amplitude) 映射为 hidden_dim
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 2. GNN 消息传递层 (模拟 BP 迭代)
        self.gnn_layers = nn.ModuleList([
            BiVariableGNNLayer(hidden_dim, use_gru=use_layer_gru) for _ in range(num_layers)
        ])

        # 3. [可选] 跨帧全局记忆 GRU
        # 输入是当前帧的全局特征，输出是更新后的记忆
        if use_temporal_gru:
            self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.gru = None

        # 4. 解码器 - 输出关联分数
        # 如果使用GRU，输入维度是 hidden_dim * 2；否则是 hidden_dim
        head_input_dim = hidden_dim * 2 if use_temporal_gru else hidden_dim
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, hybrid_input, hidden_state=None):
        """
        输入:
            hybrid_input: (Batch, M, K+1, 5)
            hidden_state: (Batch, hidden_dim) 上一帧的记忆，可选（仅当use_temporal_gru=True时使用）
        输出:
            logits: (Batch, M, K+1)
            new_hidden_state: (Batch, hidden_dim) 更新后的记忆（如果use_temporal_gru=False则返回None）
        """
        # A. 编码特征
        # (B, M, K+1, 5) -> (B, M, K+1, H)
        x = self.input_encoder(hybrid_input)

        # B. 迭代消息传递
        for layer in self.gnn_layers:
            x = layer(x)

        batch_size, M, K_plus_1, h_dim = x.shape

        # C. 根据是否使用GRU选择不同的处理方式
        if self.use_temporal_gru:
            # 使用跨帧GRU记忆
            # 提取全局环境特征 (Global Pooling)
            global_feat, _ = torch.max(x.view(batch_size, -1, h_dim), dim=1)  # (Batch, hidden_dim)

            # GRU 记忆更新
            if hidden_state is None:
                hidden_state = torch.zeros_like(global_feat)

            new_hidden_state = self.gru(global_feat, hidden_state)  # (Batch, hidden_dim)

            # 特征融合：把全局记忆扩展，拼回到每一个节点上
            h_expanded = new_hidden_state.view(batch_size, 1, 1, h_dim).expand(batch_size, M, K_plus_1, h_dim)

            # 拼接: (Batch, M, K+1, hidden_dim * 2)
            combined_feat = torch.cat([x, h_expanded], dim=-1)
        else:
            # 不使用跨帧GRU，直接使用当前帧特征
            combined_feat = x
            new_hidden_state = None

        # D. 解码为 Logits
        # (B, M, K+1, hidden_dim or hidden_dim*2) -> (B, M, K+1)
        logits = self.head(combined_feat).squeeze(-1)

        return logits, new_hidden_state

class BiVariableGNNLayer(nn.Module):
    """
    单层 GNN：模拟一次完整的 "测量 <-> 锚点" 双向消息传递
    可选添加 GRU 更新机制
    """
    def __init__(self, hidden_dim, use_gru=False):
        super().__init__()
        self.use_gru = use_gru

        # 行更新 (测量选锚点)
        self.mlp_row = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 列更新 (锚点选测量)
        self.mlp_col = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # [可选] GRU 更新门 (用于融合新旧特征)
        if use_gru:
            self.gru_row = nn.GRUCell(hidden_dim, hidden_dim)
            self.gru_col = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.gru_row = None
            self.gru_col = None

    def forward(self, H):
        B, M, K_plus_1, hidden_dim = H.shape

        # 1. 行竞争 (Row Update)
        # 测量节点聚合所有锚点的信息
        row_max = torch.max(H, dim=2, keepdim=True)[0] # (B, M, 1, H)
        row_update = self.mlp_row(H - row_max) # (B, M, K+1, H)

        # [可选] GRU 更新：逐元素融合
        if self.use_gru:
            H_flat = H.reshape(B * M * K_plus_1, hidden_dim)
            row_update_flat = row_update.reshape(B * M * K_plus_1, hidden_dim)
            H_flat = self.gru_row(row_update_flat, H_flat)
            H = H_flat.reshape(B, M, K_plus_1, hidden_dim)
        else:
            # 不使用GRU，直接残差连接
            H = H + row_update

        # 2. 列竞争 (Col Update)
        # 锚点节点聚合所有测量的信息
        col_max = torch.max(H, dim=1, keepdim=True)[0] # (B, 1, K+1, H)
        col_update = self.mlp_col(H - col_max) # (B, M, K+1, H)

        # [可选] GRU 更新
        if self.use_gru:
            H_flat = H.reshape(B * M * K_plus_1, hidden_dim)
            col_update_flat = col_update.reshape(B * M * K_plus_1, hidden_dim)
            H_flat = self.gru_col(col_update_flat, H_flat)
            H = H_flat.reshape(B, M, K_plus_1, hidden_dim)
        else:
            # 不使用GRU，直接残差连接
            H = H + col_update

        return H