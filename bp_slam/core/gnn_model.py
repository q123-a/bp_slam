"""
bp_slam/core/gnn_model.py
混合驱动 FGNN 模型定义 - 简化版
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FactorGraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2):
        super().__init__()

        # 1. 特征融合编码器 (Physics-Data Fusion)
        # 将 (LogProb, Residual, Variance, Existence) 映射为 hidden_dim
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 2. GNN 消息传递层 (模拟 BP 迭代)
        self.gnn_layers = nn.ModuleList([
            BiVariableGNNLayer(hidden_dim) for _ in range(num_layers)
        ])

        # 3. 解码器 - 输出关联分数
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, hybrid_input):
        """
        输入: hybrid_input (Batch, M, K+1, 4)
        输出: logits (Batch, M, K+1)
        """
        # A. 编码特征
        # (B, M, K+1, 4) -> (B, M, K+1, H)
        x = self.input_encoder(hybrid_input)

        # B. 迭代消息传递
        for layer in self.gnn_layers:
            x = layer(x)

        # C. 解码为 Logits
        # (B, M, K+1, H) -> (B, M, K+1)
        logits = self.head(x).squeeze(-1)

        return logits

class BiVariableGNNLayer(nn.Module):
    """
    单层 GNN：模拟一次完整的 "测量 <-> 锚点" 双向消息传递
    添加 GRU 更新机制
    """
    def __init__(self, hidden_dim):
        super().__init__()
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

        # GRU 更新门 (用于融合新旧特征)
        self.gru_row = nn.GRUCell(hidden_dim, hidden_dim)
        self.gru_col = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, H):
        B, M, K_plus_1, hidden_dim = H.shape

        # 1. 行竞争 (Row Update) + GRU
        # 测量节点聚合所有锚点的信息
        row_max = torch.max(H, dim=2, keepdim=True)[0] # (B, M, 1, H)
        row_update = self.mlp_row(H - row_max) # (B, M, K+1, H)

        # GRU 更新：逐元素融合
        H_flat = H.reshape(B * M * K_plus_1, hidden_dim)
        row_update_flat = row_update.reshape(B * M * K_plus_1, hidden_dim)
        H_flat = self.gru_row(row_update_flat, H_flat)
        H = H_flat.reshape(B, M, K_plus_1, hidden_dim)

        # 2. 列竞争 (Col Update) + GRU
        # 锚点节点聚合所有测量的信息
        col_max = torch.max(H, dim=1, keepdim=True)[0] # (B, 1, K+1, H)
        col_update = self.mlp_col(H - col_max) # (B, M, K+1, H)

        # GRU 更新
        H_flat = H.reshape(B * M * K_plus_1, hidden_dim)
        col_update_flat = col_update.reshape(B * M * K_plus_1, hidden_dim)
        H_flat = self.gru_col(col_update_flat, H_flat)
        H = H_flat.reshape(B, M, K_plus_1, hidden_dim)

        return H