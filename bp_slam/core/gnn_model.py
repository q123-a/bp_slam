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

        self.hidden_dim = hidden_dim

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

        # 3. [新增] 跨帧全局记忆 GRU
        # 输入是当前帧的全局特征，输出是更新后的记忆
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # 4. [修改] 解码器 - 输出关联分数
        # 输入维度变大两倍，因为我们要把 (特征 + 记忆) 拼在一起
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, hybrid_input, hidden_state=None):
        """
        输入:
            hybrid_input: (Batch, M, K+1, 4)
            hidden_state: (Batch, hidden_dim) 上一帧的记忆，可选
        输出:
            logits: (Batch, M, K+1)
            new_hidden_state: (Batch, hidden_dim) 更新后的记忆
        """
        # A. 编码特征
        # (B, M, K+1, 4) -> (B, M, K+1, H)
        x = self.input_encoder(hybrid_input)

        # B. 迭代消息传递
        for layer in self.gnn_layers:
            x = layer(x)

        batch_size, M, K_plus_1, h_dim = x.shape

        # C. [新增] 提取全局环境特征 (Global Pooling)
        # 我们把所有测量和路标的特征聚合起来，看看当前这一帧"整体长什么样"
        # 使用 Max Pooling，对捕捉异常(大残差)比较敏感
        global_feat, _ = torch.max(x.view(batch_size, -1, h_dim), dim=1)  # (Batch, hidden_dim)

        # D. [新增] GRU 记忆更新
        if hidden_state is None:
            hidden_state = torch.zeros_like(global_feat)

        # 这里的 hidden_state 包含了过去几十帧的信息
        new_hidden_state = self.gru(global_feat, hidden_state)  # (Batch, hidden_dim)

        # E. [新增] 特征融合
        # 把全局记忆扩展，拼回到每一个节点上
        h_expanded = new_hidden_state.view(batch_size, 1, 1, h_dim).expand(batch_size, M, K_plus_1, h_dim)

        # 拼接: (Batch, M, K+1, hidden_dim * 2)
        combined_feat = torch.cat([x, h_expanded], dim=-1)

        # F. 解码为 Logits
        # (B, M, K+1, hidden_dim*2) -> (B, M, K+1)
        logits = self.head(combined_feat).squeeze(-1)

        return logits, new_hidden_state

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