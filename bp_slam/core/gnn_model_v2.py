"""
bp_slam/core/gnn_model_v2.py
改进版 FGNN 模型 - 借鉴 Factor-Graph-Neural-Network 的设计

主要改进:
1. Softmax 聚合 (smooth max) 替代 Hard Max
2. 边特征增强 (拼接绝对值和相对值)
3. 多聚合方式集成 (max/softmax/mean)
4. Skip Connections 跨层连接
5. 残差块优化 (Pre-activation + 瓶颈结构)

完全兼容自监督学习，输入输出接口与原版本相同
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiVariableGNNLayerV2(nn.Module):
    """
    改进版 GNN 层：模拟一次完整的 "测量 <-> 锚点" 双向消息传递

    改进点:
    1. 支持多种聚合方式: max, softmax, mean
    2. 边特征增强: 拼接 [h, h - agg(h)]
    3. 可选 GRU 更新机制
    """
    def __init__(self, hidden_dim, use_gru=False, aggregation='softmax',
                 gamma=3.0, edge_mode='concat'):
        """
        参数:
            hidden_dim: 隐藏层维度
            use_gru: 是否使用 GRU 更新门
            aggregation: 聚合方式 ('max', 'softmax', 'mean')
            gamma: softmax 聚合的温度参数 (gamma=3 推荐)
            edge_mode: 边特征模式 ('diff' 只用差分, 'concat' 拼接绝对值和差分)
        """
        super().__init__()
        self.use_gru = use_gru
        self.aggregation = aggregation
        self.gamma = gamma
        self.edge_mode = edge_mode

        # 根据边特征模式确定 MLP 输入维度
        mlp_input_dim = hidden_dim * 2 if edge_mode == 'concat' else hidden_dim

        # 行更新 (测量选锚点)
        self.mlp_row = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 列更新 (锚点选测量)
        self.mlp_col = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # [可选] GRU 更新门
        if use_gru:
            self.gru_row = nn.GRUCell(hidden_dim, hidden_dim)
            self.gru_col = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.gru_row = None
            self.gru_col = None

    def aggregate(self, H, dim):
        """
        统一的聚合函数

        参数:
            H: 输入特征 (B, M, K+1, hidden_dim)
            dim: 聚合维度 (2 for row, 1 for col)

        返回:
            聚合后的特征 (B, M, 1, hidden_dim) or (B, 1, K+1, hidden_dim)
        """
        if self.aggregation == 'softmax':
            # Smooth max (可微分, 梯度流向所有节点)
            # 数学: (1/γ) * log(∑ exp(γ * h_i))
            # γ=3: 平衡, γ→∞: 退化为 max, γ→0: 退化为 mean
            return (1.0 / self.gamma) * torch.logsumexp(self.gamma * H, dim=dim, keepdim=True)

        elif self.aggregation == 'max':
            # Hard max (原实现, 梯度只流向最大值)
            return torch.max(H, dim=dim, keepdim=True)[0]

        elif self.aggregation == 'mean':
            # Mean pooling (全局平均)
            return torch.mean(H, dim=dim, keepdim=True)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def forward(self, H):
        """
        前向传播

        参数:
            H: (B, M, K+1, hidden_dim)

        返回:
            H: (B, M, K+1, hidden_dim)
        """
        B, M, K_plus_1, hidden_dim = H.shape

        # ===== 1. 行竞争 (Row Update) =====
        # 测量节点聚合所有锚点的信息
        row_agg = self.aggregate(H, dim=2)  # (B, M, 1, hidden_dim)

        # 构建边特征
        if self.edge_mode == 'concat':
            # 拼接 [原始特征, 差分特征]
            # 数学: MLP([h_i, h_i - max_j h_j])
            row_diff = H - row_agg
            row_input = torch.cat([H, row_diff], dim=-1)  # (B, M, K+1, 2*hidden_dim)
        else:
            # 只用差分特征
            # 数学: MLP(h_i - max_j h_j)
            row_input = H - row_agg  # (B, M, K+1, hidden_dim)

        row_update = self.mlp_row(row_input)  # (B, M, K+1, hidden_dim)

        # [可选] GRU 更新：逐元素融合
        if self.use_gru:
            H_flat = H.reshape(B * M * K_plus_1, hidden_dim)
            row_update_flat = row_update.reshape(B * M * K_plus_1, hidden_dim)
            H_flat = self.gru_row(row_update_flat, H_flat)
            H = H_flat.reshape(B, M, K_plus_1, hidden_dim)
        else:
            # 残差连接
            H = H + row_update

        # ===== 2. 列竞争 (Col Update) =====
        # 锚点节点聚合所有测量的信息
        col_agg = self.aggregate(H, dim=1)  # (B, 1, K+1, hidden_dim)

        # 构建边特征
        if self.edge_mode == 'concat':
            col_diff = H - col_agg
            col_input = torch.cat([H, col_diff], dim=-1)  # (B, M, K+1, 2*hidden_dim)
        else:
            col_input = H - col_agg  # (B, M, K+1, hidden_dim)

        col_update = self.mlp_col(col_input)  # (B, M, K+1, hidden_dim)

        # [可选] GRU 更新
        if self.use_gru:
            H_flat = H.reshape(B * M * K_plus_1, hidden_dim)
            col_update_flat = col_update.reshape(B * M * K_plus_1, hidden_dim)
            H_flat = self.gru_col(col_update_flat, H_flat)
            H = H_flat.reshape(B, M, K_plus_1, hidden_dim)
        else:
            # 残差连接
            H = H + col_update

        return H


class MultiAggregationGNNLayer(nn.Module):
    """
    多聚合方式集成层：同时使用 max/softmax/mean，然后融合

    优势:
    - Max 捕捉最强信号
    - Softmax 平衡多个候选
    - Mean 提供全局上下文
    - 网络自动学习如何组合
    """
    def __init__(self, hidden_dim, use_gru=False, gamma=3.0):
        super().__init__()
        self.use_gru = use_gru
        self.gamma = gamma

        # 三种聚合方式的独立 MLP
        self.mlp_max = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.mlp_softmax = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.mlp_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 融合模块
        self.merge_row = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.merge_col = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # [可选] GRU
        if use_gru:
            self.gru_row = nn.GRUCell(hidden_dim, hidden_dim)
            self.gru_col = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.gru_row = None
            self.gru_col = None

    def forward(self, H):
        B, M, K_plus_1, hidden_dim = H.shape

        # ===== 行更新：三种聚合方式 =====
        row_max = torch.max(H, dim=2, keepdim=True)[0]
        row_softmax = (1.0 / self.gamma) * torch.logsumexp(self.gamma * H, dim=2, keepdim=True)
        row_mean = torch.mean(H, dim=2, keepdim=True)

        update_max = self.mlp_max(H - row_max)
        update_softmax = self.mlp_softmax(H - row_softmax)
        update_mean = self.mlp_mean(H - row_mean)

        # 融合三种更新
        row_update = self.merge_row(torch.cat([update_max, update_softmax, update_mean], dim=-1))

        if self.use_gru:
            H_flat = H.reshape(B * M * K_plus_1, hidden_dim)
            row_update_flat = row_update.reshape(B * M * K_plus_1, hidden_dim)
            H_flat = self.gru_row(row_update_flat, H_flat)
            H = H_flat.reshape(B, M, K_plus_1, hidden_dim)
        else:
            H = H + row_update

        # ===== 列更新：三种聚合方式 =====
        col_max = torch.max(H, dim=1, keepdim=True)[0]
        col_softmax = (1.0 / self.gamma) * torch.logsumexp(self.gamma * H, dim=1, keepdim=True)
        col_mean = torch.mean(H, dim=1, keepdim=True)

        update_max = self.mlp_max(H - col_max)
        update_softmax = self.mlp_softmax(H - col_softmax)
        update_mean = self.mlp_mean(H - col_mean)

        # 融合三种更新
        col_update = self.merge_col(torch.cat([update_max, update_softmax, update_mean], dim=-1))

        if self.use_gru:
            H_flat = H.reshape(B * M * K_plus_1, hidden_dim)
            col_update_flat = col_update.reshape(B * M * K_plus_1, hidden_dim)
            H_flat = self.gru_col(col_update_flat, H_flat)
            H = H_flat.reshape(B, M, K_plus_1, hidden_dim)
        else:
            H = H + col_update

        return H


class FactorGraphNeuralNetworkV2(nn.Module):
    """
    改进版因子图神经网络

    改进点:
    1. 支持多种聚合方式 (softmax/max/mean)
    2. 边特征增强 (concat 模式)
    3. Skip Connections 跨层连接
    4. 可选多聚合集成

    输入输出接口与原版本完全相同，可直接替换
    """
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2,
                 use_temporal_gru=False, use_layer_gru=False,
                 aggregation='softmax', gamma=3.0, edge_mode='concat',
                 use_multi_aggregation=False, skip_connections=None):
        """
        参数:
            input_dim: 输入特征维度 (默认 5)
            hidden_dim: 隐藏层维度 (默认 64)
            num_layers: GNN 层数 (默认 2)
            use_temporal_gru: 是否使用跨帧 GRU
            use_layer_gru: 是否使用层内 GRU
            aggregation: 聚合方式 ('softmax', 'max', 'mean')
            gamma: softmax 温度参数 (默认 3.0)
            edge_mode: 边特征模式 ('diff', 'concat')
            use_multi_aggregation: 是否使用多聚合集成
            skip_connections: 跨层连接字典 {source_layer: target_layer}
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_temporal_gru = use_temporal_gru
        self.use_layer_gru = use_layer_gru
        self.skip_connections = skip_connections or {}

        # 1. 特征融合编码器
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 2. GNN 消息传递层
        if use_multi_aggregation:
            # 使用多聚合集成
            self.gnn_layers = nn.ModuleList([
                MultiAggregationGNNLayer(hidden_dim, use_gru=use_layer_gru, gamma=gamma)
                for _ in range(num_layers)
            ])
        else:
            # 使用单一聚合方式
            self.gnn_layers = nn.ModuleList([
                BiVariableGNNLayerV2(
                    hidden_dim,
                    use_gru=use_layer_gru,
                    aggregation=aggregation,
                    gamma=gamma,
                    edge_mode=edge_mode
                ) for _ in range(num_layers)
            ])

        # 3. [可选] 跨帧全局记忆 GRU
        if use_temporal_gru:
            self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.gru = None

        # 4. 解码器 - 输出关联分数
        head_input_dim = hidden_dim * 2 if use_temporal_gru else hidden_dim
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, hybrid_input, hidden_state=None):
        """
        前向传播

        输入:
            hybrid_input: (Batch, M, K+1, 5) 混合特征
            hidden_state: (Batch, hidden_dim) 上一帧的记忆 (可选)

        输出:
            logits: (Batch, M, K+1) 关联分数
            new_hidden_state: (Batch, hidden_dim) 更新后的记忆
        """
        # A. 编码特征
        x = self.input_encoder(hybrid_input)  # (B, M, K+1, hidden_dim)

        # B. 迭代消息传递 + Skip Connections
        layer_outputs = []
        for idx, layer in enumerate(self.gnn_layers):
            x = layer(x)

            # 应用 skip connection
            if idx in self.skip_connections:
                source_idx = self.skip_connections[idx]
                if source_idx < len(layer_outputs):
                    x = x + layer_outputs[source_idx]

            layer_outputs.append(x)

        batch_size, M, K_plus_1, h_dim = x.shape

        # C. 根据是否使用 GRU 选择不同的处理方式
        if self.use_temporal_gru:
            # 使用跨帧 GRU 记忆
            global_feat, _ = torch.max(x.view(batch_size, -1, h_dim), dim=1)

            if hidden_state is None:
                hidden_state = torch.zeros_like(global_feat)

            new_hidden_state = self.gru(global_feat, hidden_state)

            # 特征融合
            h_expanded = new_hidden_state.view(batch_size, 1, 1, h_dim).expand(batch_size, M, K_plus_1, h_dim)
            combined_feat = torch.cat([x, h_expanded], dim=-1)
        else:
            combined_feat = x
            new_hidden_state = None

        # D. 解码为 Logits
        logits = self.head(combined_feat).squeeze(-1)  # (B, M, K+1)

        return logits, new_hidden_state


class JointDualHeadGNNV2(nn.Module):
    """
    改进版双头联合 GNN：质量头 + 关联头

    改进点:
    1. 使用改进的 GNN 层 (softmax 聚合等)
    2. Skip Connections
    3. 更好的特征融合

    输入输出接口与原版本完全相同
    """
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2,
                 use_temporal_gru=False, use_layer_gru=False,
                 aggregation='softmax', gamma=3.0, edge_mode='concat',
                 skip_connections=None):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_temporal_gru = use_temporal_gru
        self.use_layer_gru = use_layer_gru
        self.skip_connections = skip_connections or {}

        # ===== 1. 共享特征提取主干 =====
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GNN 消息传递层 (使用改进版)
        self.gnn_layers = nn.ModuleList([
            BiVariableGNNLayerV2(
                hidden_dim,
                use_gru=use_layer_gru,
                aggregation=aggregation,
                gamma=gamma,
                edge_mode=edge_mode
            ) for _ in range(num_layers)
        ])

        # [可选] 跨帧节点级记忆 GRU
        if use_temporal_gru:
            self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.gru = None

        # ===== 2. 质量头 =====
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # ===== 3. 关联头 =====
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
            hidden_state: (Batch*M*K, hidden_dim) 节点级记忆 (可选)

        输出:
            assoc_logits: (Batch, M, K) 关联分数
            quality_scores: (Batch, M) 质量分数 [0, 1]
            new_hidden_state: (Batch*M*K, hidden_dim) 更新后的记忆
        """
        # A. 编码特征
        x = self.input_encoder(hybrid_input)  # (B, M, K+1, hidden_dim)

        # B. 迭代消息传递 + Skip Connections
        layer_outputs = []
        for idx, layer in enumerate(self.gnn_layers):
            x = layer(x)

            # 应用 skip connection
            if idx in self.skip_connections:
                source_idx = self.skip_connections[idx]
                if source_idx < len(layer_outputs):
                    x = x + layer_outputs[source_idx]

            layer_outputs.append(x)

        batch_size, M, K_plus_1, h_dim = x.shape
        K = K_plus_1 - 1

        # C. 提取测量级别的特征（用于质量头）
        meas_feat = torch.max(x, dim=2)[0]  # (B, M, hidden_dim)

        # D. 质量头推理
        quality_scores = self.quality_head(meas_feat).squeeze(-1)  # (B, M)

        # E. 关联头推理
        x_anchors = x[:, :, :K, :]  # (B, M, K, hidden_dim)

        # 如果使用跨帧 GRU
        if self.use_temporal_gru:
            x_flat = x_anchors.reshape(-1, h_dim)

            if hidden_state is None:
                hidden_state = torch.zeros_like(x_flat)

            new_hidden_state = self.gru(x_flat, hidden_state)
            x_mem = new_hidden_state.view(batch_size, M, K, h_dim)
            combined_feat = x_mem
        else:
            combined_feat = x_anchors
            new_hidden_state = None

        # 关联头输出
        assoc_logits = self.association_head(combined_feat).squeeze(-1)  # (B, M, K)

        return assoc_logits, quality_scores, new_hidden_state
