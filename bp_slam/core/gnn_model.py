"""
bp_slam/core/gnn_model.py
混合驱动 FGNN 模型定义 - 简化版 + 图注意力增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_attention import SpatioTemporalGraphAttention

class FactorGraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, use_temporal_gru=False, use_layer_gru=False, use_graph_attention=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_temporal_gru = use_temporal_gru  # 是否使用跨帧GRU
        self.use_layer_gru = use_layer_gru        # 是否使用层内GRU
        self.use_graph_attention = use_graph_attention  # 是否使用图注意力

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

        # [新增] 图注意力模块（可选）
        if use_graph_attention:
            self.graph_attention = SpatioTemporalGraphAttention(
                feature_dim=hidden_dim,
                num_heads=4,
                dropout=0.1,
                temporal_window=3
            )
        else:
            self.graph_attention = None

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

        # [新增] 显式权重初始化 - 解决初始化敏感性问题
        self.apply(self._init_weights)

        # [关键] 输出层特殊初始化：微小随机打破对称性
        # 使用小正态分布而非纯 0，给梯度下降一个"启动推力"
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.head[-1].bias, 0)

    def forward(self, hybrid_input, hidden_state=None):
        """
        输入:
            hybrid_input: (Batch, M, K+1, 5)
            hidden_state: (Batch, hidden_dim) 上一帧的记忆，可选（仅当use_temporal_gru=True时使用）
        输出:
            logits: (Batch, M, K+1)
            new_hidden_state: (Batch, hidden_dim) 更新后的记忆（如果use_temporal_gru=False则返回None）
            attention_info: dict, 注意力相关信息（如果use_graph_attention=True）
        """
        # A. 编码特征
        # (B, M, K+1, 5) -> (B, M, K+1, H)
        x = self.input_encoder(hybrid_input)

        # B. 迭代消息传递
        for layer in self.gnn_layers:
            x = layer(x)

        batch_size, M, K_plus_1, h_dim = x.shape

        # [新增] B.5. 图注意力增强（可选）
        attention_info = {}
        if self.graph_attention is not None:
            # 提取测量节点特征 (对K+1维度取平均)
            measurement_features = x.mean(dim=2)  # (B, M, H)

            # 应用图注意力
            enhanced_features, spatial_attn, clutter_scores = self.graph_attention(measurement_features)

            # 将增强特征广播回原始形状
            enhanced_features = enhanced_features.unsqueeze(2).expand(batch_size, M, K_plus_1, h_dim)

            # 残差连接
            x = x + enhanced_features

            # 保存注意力信息
            attention_info = {
                'spatial_attention': spatial_attn,
                'clutter_scores': clutter_scores
            }

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
            combined_feat = torch.cat([x, h_expanded], dim=-1)
        else:
            # 不使用GRU，直接使用当前帧特征
            combined_feat = x
            new_hidden_state = None

        # D. 解码为 Logits
        logits = self.head(combined_feat).squeeze(-1)

        return logits, new_hidden_state, attention_info

    def _init_weights(self, m):
        """
        显式权重初始化方法 - 解决随机种子敏感性问题

        使用 Kaiming 初始化 (He Initialization) 针对 ReLU 网络
        使用正交初始化 (Orthogonal) 针对 GRU 单元
        """
        if isinstance(m, nn.Linear):
            # 线性层使用 Kaiming 初始化
            # 解决 ReLU 导致的方差偏移问题
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            # LayerNorm 初始化为标准值
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.GRUCell):
            # GRU 必须使用正交初始化防止梯度消失/爆炸
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

class BiVariableGNNLayer(nn.Module):
    """
    单层 GNN：模拟一次完整的 "测量 <-> 锚点" 双向消息传递
    可选的 GRU 更新机制
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

        # [新增] 显式权重初始化
        self.apply(self._init_weights)

        # [关键] 输出层特殊初始化：微小随机打破对称性
        # BiVariableGNNLayer 没有输出层，所以这里不需要特殊处理

    def _init_weights(self, m):
        """显式权重初始化方法"""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GRUCell):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def forward(self, H):
        B, M, K_plus_1, hidden_dim = H.shape

        # 1. 行竞争 (Row Update)
        # 测量节点聚合所有锚点的信息
        row_max = torch.max(H, dim=2, keepdim=True)[0] # (B, M, 1, H)
        row_update = self.mlp_row(H - row_max) # (B, M, K+1, H)

        if self.use_gru:
            # GRU 更新：逐元素融合
            H_flat = H.reshape(B * M * K_plus_1, hidden_dim)
            row_update_flat = row_update.reshape(B * M * K_plus_1, hidden_dim)
            H_flat = self.gru_row(row_update_flat, H_flat)
            H = H_flat.reshape(B, M, K_plus_1, hidden_dim)
        else:
            # 残差连接
            H = H + row_update

        # 2. 列竞争 (Col Update)
        # 锚点节点聚合所有测量的信息
        col_max = torch.max(H, dim=1, keepdim=True)[0] # (B, 1, K+1, H)
        col_update = self.mlp_col(H - col_max) # (B, M, K+1, H)

        if self.use_gru:
            # GRU 更新
            H_flat = H.reshape(B * M * K_plus_1, hidden_dim)
            col_update_flat = col_update.reshape(B * M * K_plus_1, hidden_dim)
            H_flat = self.gru_col(col_update_flat, H_flat)
            H = H_flat.reshape(B, M, K_plus_1, hidden_dim)
        else:
            # 残差连接
            H = H + col_update

        return H