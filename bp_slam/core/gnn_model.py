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


# ==============================================================================
# 双头联合架构 (Joint Dual-Head Architecture)
# ==============================================================================

class JointDualHeadGNN(nn.Module):
    """
    联合双头GNN：质量头 + 关联头

    架构设计：
    1. 共享特征提取主干（Backbone）
    2. 质量头（Quality Head）：判断测量是否为真实信号（物理老师监督）
    3. 关联头（Association Head）：判断测量属于哪个锚点（几何老师监督）

    优势：
    - 质量头不依赖预测位置，只看物理特征（RSS一致性）
    - 即使关联头被错误预测误导，质量头仍能正确识别杂波
    - 解耦了"是什么"和"是谁"两个问题
    """

    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2,
                 use_temporal_gru=False, use_layer_gru=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_temporal_gru = use_temporal_gru
        self.use_layer_gru = use_layer_gru

        # ===== 1. 共享特征提取主干 (Shared Backbone) =====
        # 输入: (LogProb, Residual, Variance, Existence, Amplitude)
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GNN 消息传递层
        self.gnn_layers = nn.ModuleList([
            BiVariableGNNLayer(hidden_dim, use_gru=use_layer_gru)
            for _ in range(num_layers)
        ])

        # [可选] 跨帧节点级记忆 GRU
        # 改进：每个测量-锚点对都有独立的记忆，而非全局共享
        # 这样能更精细地平滑每个关联的时序波动，消除OSPA尖峰
        if use_temporal_gru:
            self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.gru = None

        # ===== 2. 质量头 (Quality Head) =====
        # 输入: 每个测量的特征 (M, hidden_dim)
        # 输出: 每个测量的质量分数 (M, 1)，范围 [0, 1]
        # 含义: 0 = 杂波，1 = 真实信号
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出 0~1 的概率
        )

        # ===== 3. 关联头 (Association Head) =====
        # 输入: 每个测量-锚点对的特征 (M, K, hidden_dim)
        # 输出: 每个测量对K个锚点的关联分数 (M, K)
        # 注意: 不再需要垃圾桶列，因为杂波由质量头处理
        # 改进：GRU直接作用在节点特征上，不需要拼接
        self.association_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, hybrid_input, hidden_state=None):
        """
        前向传播

        输入:
            hybrid_input: (Batch, M, K+1, 5) 混合特征
            hidden_state: (Batch*M*K, hidden_dim) 上一帧的节点级记忆（可选）
                         改进：每个测量-锚点对都有独立记忆

        输出:
            assoc_logits: (Batch, M, K) 关联分数（不含垃圾桶列）
            quality_scores: (Batch, M) 质量分数 [0, 1]
            new_hidden_state: (Batch*M*K, hidden_dim) 更新后的节点级记忆
        """
        # A. 编码特征
        # (B, M, K+1, 5) -> (B, M, K+1, H)
        x = self.input_encoder(hybrid_input)

        # B. 迭代消息传递
        for layer in self.gnn_layers:
            x = layer(x)

        batch_size, M, K_plus_1, h_dim = x.shape
        K = K_plus_1 - 1  # 锚点数量（不含垃圾桶列）

        # C. 提取测量级别的特征（用于质量头）
        # 方法：对每个测量，聚合其与所有锚点的特征
        # (B, M, K+1, H) -> (B, M, H)
        meas_feat = torch.max(x, dim=2)[0]  # Max pooling across anchors

        # D. 质量头推理
        # (B, M, H) -> (B, M, 1) -> (B, M)
        quality_scores = self.quality_head(meas_feat).squeeze(-1)

        # E. 关联头推理
        # 只使用前K列（不含垃圾桶列）
        x_anchors = x[:, :, :K, :]  # (B, M, K, H)

        # 如果使用跨帧GRU，进行节点级别的时序平滑
        if self.use_temporal_gru:
            # [改进] 节点级别的GRU记忆（而非全局级别）
            # 每个测量-锚点对都有独立的记忆，能更精细地平滑OSPA尖峰

            # 展平为 (B*M*K, H)
            x_flat = x_anchors.reshape(-1, h_dim)

            # 初始化或使用上一帧的hidden_state
            if hidden_state is None:
                hidden_state = torch.zeros_like(x_flat)

            # GRU更新：每个节点独立记忆
            new_hidden_state = self.gru(x_flat, hidden_state)

            # 恢复形状 (B, M, K, H)
            x_mem = new_hidden_state.view(batch_size, M, K, h_dim)

            # 使用记忆增强的特征
            combined_feat = x_mem
        else:
            combined_feat = x_anchors
            new_hidden_state = None

        # 关联头输出
        # (B, M, K, H) -> (B, M, K)
        assoc_logits = self.association_head(combined_feat).squeeze(-1)

        return assoc_logits, quality_scores, new_hidden_state