"""
bp_slam/core/gnn_model_gat.py
GAT-based Factor Graph Neural Network for BP-SLAM

主要改进:
1. 使用 GATv2Conv 替代 MLP 聚合
2. 精简边特征到 4 维 [Δx, Δy, Σ_trace, RSS_norm]
3. 加入 GRU 时序记忆解决 OSPA 震荡
4. Edge-Centric 设计，适合 SLAM 的"重边轻点"结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 检查是否安装了 torch_geometric
try:
    from torch_geometric.nn import GATv2Conv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not installed. GAT model will not be available.")
    print("Install with: pip install torch-geometric")


class FGNN_GAT_Module(nn.Module):
    """
    GAT-based Factor Graph Neural Network

    架构:
    1. Edge Encoder: 4维物理特征 -> 64维语义嵌入
    2. GATv2: 动态注意力消息传递
    3. GRU: 时序记忆单元
    4. Dual Heads: 关联头 + 质量头
    """

    def __init__(self, config):
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required for GAT model")

        # --- 配置参数 ---
        self.node_dim = config.get('gnn_hidden_dim', 64)
        self.edge_in_dim = 4  # [dx, dy, sigma_trace, rss_norm]
        self.heads = config.get('gnn_gat_heads', 4)  # 多头注意力
        self.dropout = config.get('gnn_dropout', 0.1)
        self.use_gru = config.get('gnn_use_temporal_gru', True)

        # --- 1. 特征编码器 (Feature Encoder) ---
        # 将 4维 原始物理特征映射到高维语义空间
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.edge_in_dim, 64),
            nn.LayerNorm(64),  # 归一化有助于非高斯数据收敛
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 64)
        )

        # --- 2. GATv2 核心层 (The Brain) ---
        # 使用 GATv2，因为它能处理"动态图"的注意力
        # edge_dim=64: 告诉 GAT 我们有强力的边特征输入
        self.gat = GATv2Conv(
            in_channels=self.node_dim,
            out_channels=self.node_dim // self.heads,  # 保持输出总维度不变
            heads=self.heads,
            edge_dim=64,           # 输入编码后的边特征
            concat=True,           # 拼接多头结果
            dropout=self.dropout,
            add_self_loops=False   # 二部图通常不加自环
        )

        # --- 3. GRU 时序记忆单元 (The Memory) ---
        # 这就是解决 OSPA 凸起的关键！
        # 它让节点记住："我上一帧是杂波，这一帧大概率还是"
        if self.use_gru:
            self.gru = nn.GRUCell(self.node_dim, self.node_dim)
        else:
            self.gru = None

        # --- 4. 解码头 (Decoders) ---

        # A. 关联头 (Association Head) -> 替代 BP 的 Beta
        # 输入: [Node_i, Node_j, Edge_ij]
        # 我们要把交互后的信息重新拼回来判断这条边
        self.assoc_mlp = nn.Sequential(
            nn.Linear(self.node_dim * 2 + 64, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1)  # 输出 Logits
        )

        # B. 质量头 (Quality Head) -> 替代 BP 的 Xi
        # 输入: [Node_Obs (经过 GAT+GRU 更新后的状态)]
        self.quality_mlp = nn.Sequential(
            nn.Linear(self.node_dim, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 1)  # 输出 Logits
        )

        print(f"✓ GAT 模型初始化完成")
        print(f"  - 节点维度: {self.node_dim}")
        print(f"  - 边特征维度: {self.edge_in_dim}")
        print(f"  - 注意力头数: {self.heads}")
        print(f"  - 使用 GRU: {self.use_gru}")

    def forward(self, x, edge_index, raw_edge_attr, hidden_state=None):
        """
        前向传播

        参数:
            x: [N, node_dim] 当前帧节点特征
            edge_index: [2, E] 邻接关系 (COO 格式)
            raw_edge_attr: [E, 4] 原始物理特征 [dx, dy, sigma_trace, rss_norm]
            hidden_state: [N, node_dim] 上一时刻的 GRU 记忆

        返回:
            assoc_logits: [E, 1] 关联 logits
            quality_logits: [N, 1] 质量 logits
            new_hidden_state: [N, node_dim] 更新后的隐状态
        """

        # 1. 编码边特征 (Raw Data -> Embedding)
        edge_emb = self.edge_encoder(raw_edge_attr)  # [E, 64]

        # 2. GAT 消息传递 (Spatial Reasoning)
        # GAT 自动根据 edge_emb 计算注意力权重 alpha
        # x_gat 是包含了邻域几何信息的聚合特征
        x_gat = self.gat(x, edge_index, edge_attr=edge_emb)
        x_gat = F.elu(x_gat)  # ELU 在 GAT 中通常比 ReLU 好

        # 3. GRU 时序更新 (Temporal Reasoning)
        if self.use_gru:
            # 如果是第一帧，初始化 hidden_state 为 0
            if hidden_state is None:
                hidden_state = torch.zeros_like(x_gat)

            # 融合：(当前观测信息 x_gat) + (历史记忆 hidden_state)
            new_hidden_state = self.gru(x_gat, hidden_state)
        else:
            # 不使用 GRU，直接使用当前特征
            new_hidden_state = x_gat

        # 4. 计算输出 (Decoding)

        # --- 关联矩阵计算 ---
        row, col = edge_index
        # 拼接: 源节点(观测) + 目标节点(路标) + 边特征
        # 注意：这里用 new_hidden_state，因为它包含了时空信息
        edge_feat_concat = torch.cat([
            new_hidden_state[row],
            new_hidden_state[col],
            edge_emb
        ], dim=-1)

        assoc_logits = self.assoc_mlp(edge_feat_concat)  # [E, 1]

        # --- 质量分数计算 ---
        # 只基于更新后的节点状态判断
        quality_logits = self.quality_mlp(new_hidden_state)  # [N, 1]

        return assoc_logits, quality_logits, new_hidden_state


class JointDualHeadGNN_GAT(nn.Module):
    """
    GAT-based 双头 GNN 模型

    兼容现有的 JointDualHeadGNN 接口，但使用 GAT 架构
    """

    def __init__(self, input_dim=5, hidden_dim=64, config=None):
        """
        初始化 GAT 双头模型

        参数:
            input_dim: 输入特征维度（保留兼容性，实际使用 4 维边特征）
            hidden_dim: 隐藏层维度
            config: 配置字典
        """
        super().__init__()

        if config is None:
            config = {}
        config['gnn_hidden_dim'] = hidden_dim

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # 初始节点特征编码器
        # 输入: (B, M, K+1, 5) -> 需要转换为节点特征
        self.node_init_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GAT 核心模块
        self.gat_module = FGNN_GAT_Module(config)

    def forward(self, hybrid_input, hidden_state=None):
        """
        前向传播（兼容现有接口）

        输入:
            hybrid_input: (Batch, M, K+1, 5) 混合特征
            hidden_state: 上一帧的记忆（格式待定）

        输出:
            assoc_logits: (Batch, M, K+1) 关联 logits
            quality_logits: (Batch, M) 质量 logits
            new_hidden_state: 更新后的记忆
        """

        # [关键问题] 需要将 (B, M, K+1, 5) 转换为图结构
        # 这需要根据你的具体数据格式来实现

        # 临时实现：假设 batch_size=1
        B, M, K_plus_1, _ = hybrid_input.shape
        K = K_plus_1 - 1

        if B != 1:
            raise NotImplementedError("GAT model currently only supports batch_size=1")

        # 提取边特征 [dx, dy, sigma_trace, rss_norm]
        # 这需要从 hybrid_input 中提取
        # 假设 hybrid_input[:, :, :, :4] 包含这些信息

        # [TODO] 这里需要根据你的实际数据格式来实现
        # 暂时返回占位符
        raise NotImplementedError(
            "需要实现 hybrid_input -> (node_features, edge_index, edge_attr) 的转换"
        )


# ============================================================================
# 辅助函数：数据格式转换
# ============================================================================

def convert_hybrid_to_graph(hybrid_input):
    """
    将 (B, M, K+1, 5) 格式转换为图结构

    参数:
        hybrid_input: (B, M, K+1, 5) 混合特征

    返回:
        node_features: [N, node_dim] 节点特征
        edge_index: [2, E] 边索引
        edge_attr: [E, 4] 边特征 [dx, dy, sigma_trace, rss_norm]
    """

    B, M, K_plus_1, feat_dim = hybrid_input.shape
    K = K_plus_1 - 1

    # [TODO] 实现转换逻辑
    # 1. 创建二部图：M 个测量节点 + K 个锚点节点
    # 2. 提取边特征
    # 3. 构建 edge_index

    raise NotImplementedError("需要实现数据格式转换")
