"""
bp_slam/core/gnn_model_v3_explicit_factor.py
Explicit Factor Node Version - True Factor Graph Neural Network

Core Design:
1. Separate representations for three types of nodes:
   - Measurement nodes: (B, M, hidden_dim)
   - Anchor nodes: (B, K, hidden_dim)
   - Factor nodes: (B, M, K, hidden_dim)

2. Two-stage message passing:
   - Stage 1: Variable nodes -> Factor nodes
   - Stage 2: Factor nodes -> Variable nodes

3. Factor nodes preserve pairwise information:
   - Each factor f_ij connects measurement m_i and anchor a_j
   - Factor nodes learn complex association patterns

Mathematical Principle:
- Variable nodes: represent states (measurements, anchor positions)
- Factor nodes: represent constraints (geometric/physical consistency)
- Message passing: simulates Belief Propagation algorithm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExplicitFactorGNNLayer(nn.Module):
    """
    Explicit Factor Node GNN Layer with Separated Node Representations
    
    Node Structure:
    - Measurement nodes: h_m (B, M, hidden_dim)
    - Anchor nodes: h_a (B, K, hidden_dim)
    - Factor nodes: h_f (B, M, K, hidden_dim)
    
    Message Passing:
    1. Variable -> Factor:
       - m_i -> f_ij: measurement i sends message to factor f_ij
       - a_j -> f_ij: anchor j sends message to factor f_ij
    2. Factor fusion: f_ij fuses messages from m_i and a_j
    3. Factor -> Variable:
       - f_ij -> m_i: factor f_ij sends message back to measurement i
       - f_ij -> a_j: factor f_ij sends message back to anchor j
    """
    
    def __init__(self, hidden_dim, use_gru=False, aggregation='softmax',
                 gamma=3.0, edge_mode='concat'):
        """
        Args:
            hidden_dim: Hidden layer dimension
            use_gru: Whether to use GRU update
            aggregation: Aggregation method ('softmax', 'max', 'mean')
            gamma: Softmax temperature parameter
            edge_mode: Edge feature mode ('diff', 'concat')
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_gru = use_gru
        self.aggregation = aggregation
        self.gamma = gamma
        self.edge_mode = edge_mode
        
        # [Hybrid Aggregation] Input dimension based on edge mode
        # 原来: h_factor + (h_factor - agg) = 2*H
        # 现在: h_factor + (h_factor - agg_mean) + (h_factor - agg_max) = 3*H (如果 concat)
        # 简化版: h_factor + agg_mean + agg_max = 3*H
        if edge_mode == 'concat':
            mlp_input_dim = hidden_dim * 3  # 拼接 Mean 和 Max
        else:
            mlp_input_dim = hidden_dim * 2  # 简化版

        # ===== Stage 1: Variable -> Factor =====
        # Measurement -> Factor message function
        self.meas_to_factor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Anchor -> Factor message function
        self.anchor_to_factor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Factor fusion (fuse messages from measurement and anchor)
        self.factor_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # ===== Stage 2: Factor -> Variable =====
        # Factor -> Measurement message function
        self.factor_to_meas = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Factor -> Anchor message function
        self.factor_to_anchor = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # [Optional] GRU update gates
        if use_gru:
            self.gru_meas = nn.GRUCell(hidden_dim, hidden_dim)
            self.gru_anchor = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.gru_meas = None
            self.gru_anchor = None
    
    def aggregate(self, H, dim):
        """
        Unified aggregation function
        
        Args:
            H: Input features
            dim: Aggregation dimension
            
        Returns:
            Aggregated features
        """
        if self.aggregation == 'softmax':
            return (1.0 / self.gamma) * torch.logsumexp(self.gamma * H, dim=dim, keepdim=True)
        elif self.aggregation == 'max':
            return torch.max(H, dim=dim, keepdim=True)[0]
        elif self.aggregation == 'mean':
            return torch.mean(H, dim=dim, keepdim=True)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
    
    def forward(self, h_meas, h_anchor, h_pairs):
        """
        Forward propagation
        
        Input:
            h_meas: (B, M, hidden_dim) measurement node features
            h_anchor: (B, K, hidden_dim) anchor node features
            h_pairs: (B, M, K, hidden_dim) pairwise features (initial factor features)
            
        Output:
            h_meas_new: (B, M, hidden_dim) updated measurement features
            h_anchor_new: (B, K, hidden_dim) updated anchor features
            h_pairs_new: (B, M, K, hidden_dim) updated pairwise features
        """
        B, M, hidden_dim = h_meas.shape
        K = h_anchor.shape[1]
        
        # ========== Stage 1: Variable -> Factor ==========
        
        # 1.1 Measurement -> Factor messages
        # Each measurement m_i sends message to all factors f_ij (j=1..K)
        msg_meas = self.meas_to_factor(h_meas)  # (B, M, hidden_dim)
        msg_meas_expanded = msg_meas.unsqueeze(2).expand(B, M, K, hidden_dim)  # (B, M, K, hidden_dim)
        
        # 1.2 Anchor -> Factor messages
        # Each anchor a_j sends message to all factors f_ij (i=1..M)
        msg_anchor = self.anchor_to_factor(h_anchor)  # (B, K, hidden_dim)
        msg_anchor_expanded = msg_anchor.unsqueeze(1).expand(B, M, K, hidden_dim)  # (B, M, K, hidden_dim)
        
        # 1.3 Factor fusion
        # Each factor f_ij fuses messages from m_i and a_j
        factor_input = torch.cat([msg_meas_expanded, msg_anchor_expanded], dim=-1)  # (B, M, K, 2*hidden_dim)
        h_factor = self.factor_fusion(factor_input)  # (B, M, K, hidden_dim)
        
        # Add residual connection with initial pairwise features
        h_factor = h_factor + h_pairs  # (B, M, K, hidden_dim)
        
        # ========== Stage 2: Factor -> Variable (Hybrid Aggregation) ==========

        # 2.1 Factor -> Measurement (多头聚合版本)
        # 同时计算 Mean 和 Max，让网络自动学习何时该信任哪个

        # 计算 Mean 聚合
        agg_mean_meas = torch.mean(h_factor, dim=2, keepdim=True)  # (B, M, 1, hidden_dim)

        # 计算 Max 聚合
        agg_max_meas = torch.max(h_factor, dim=2, keepdim=True)[0]  # (B, M, 1, hidden_dim)

        # 广播聚合结果以匹配 h_factor 的形状
        agg_mean_meas_expanded = agg_mean_meas.expand(-1, -1, K, -1)  # (B, M, K, hidden_dim)
        agg_max_meas_expanded = agg_max_meas.expand(-1, -1, K, -1)   # (B, M, K, hidden_dim)

        # Build edge features (measurement side) - 拼接三种信息
        if self.edge_mode == 'concat':
            # 拼接: 自身 + 环境平均 + 环境最强
            meas_input = torch.cat([
                h_factor,                # 自身特征
                agg_mean_meas_expanded,  # 环境平均（召回率）
                agg_max_meas_expanded    # 环境最强（精确度）
            ], dim=-1)  # (B, M, K, 3*hidden_dim)
        else:
            # 简化版：只拼接 Mean 和 Max
            meas_input = torch.cat([
                agg_mean_meas_expanded,
                agg_max_meas_expanded
            ], dim=-1)  # (B, M, K, 2*hidden_dim)

        msg_factor_to_meas = self.factor_to_meas(meas_input)  # (B, M, K, hidden_dim)

        # 最后的聚合更新（这里保持原样，使用单一聚合方式）
        meas_update = self.aggregate(msg_factor_to_meas, dim=2).squeeze(2)  # (B, M, hidden_dim)
        
        # 2.2 Factor -> Anchor (多头聚合版本)
        # 同时计算 Mean 和 Max，让网络自动学习何时该信任哪个

        # 计算 Mean 聚合
        agg_mean_anchor = torch.mean(h_factor, dim=1, keepdim=True)  # (B, 1, K, hidden_dim)

        # 计算 Max 聚合
        agg_max_anchor = torch.max(h_factor, dim=1, keepdim=True)[0]  # (B, 1, K, hidden_dim)

        # 广播聚合结果以匹配 h_factor 的形状
        agg_mean_anchor_expanded = agg_mean_anchor.expand(-1, M, -1, -1)  # (B, M, K, hidden_dim)
        agg_max_anchor_expanded = agg_max_anchor.expand(-1, M, -1, -1)   # (B, M, K, hidden_dim)

        # Build edge features (anchor side) - 拼接三种信息
        if self.edge_mode == 'concat':
            # 拼接: 自身 + 环境平均 + 环境最强
            anchor_input = torch.cat([
                h_factor,                 # 自身特征
                agg_mean_anchor_expanded, # 环境平均（召回率）
                agg_max_anchor_expanded   # 环境最强（精确度）
            ], dim=-1)  # (B, M, K, 3*hidden_dim)
        else:
            # 简化版：只拼接 Mean 和 Max
            anchor_input = torch.cat([
                agg_mean_anchor_expanded,
                agg_max_anchor_expanded
            ], dim=-1)  # (B, M, K, 2*hidden_dim)

        msg_factor_to_anchor = self.factor_to_anchor(anchor_input)  # (B, M, K, hidden_dim)

        # 最后的聚合更新（这里保持原样，使用单一聚合方式）
        anchor_update = self.aggregate(msg_factor_to_anchor, dim=1).squeeze(1)  # (B, K, hidden_dim)
        
        # 2.3 Apply updates (residual connection or GRU)
        if self.use_gru:
            # GRU update for measurements
            h_meas_flat = h_meas.reshape(B * M, hidden_dim)
            meas_update_flat = meas_update.reshape(B * M, hidden_dim)
            h_meas_new = self.gru_meas(meas_update_flat, h_meas_flat).reshape(B, M, hidden_dim)
            
            # GRU update for anchors
            h_anchor_flat = h_anchor.reshape(B * K, hidden_dim)
            anchor_update_flat = anchor_update.reshape(B * K, hidden_dim)
            h_anchor_new = self.gru_anchor(anchor_update_flat, h_anchor_flat).reshape(B, K, hidden_dim)
        else:
            # Residual connection
            h_meas_new = h_meas + meas_update
            h_anchor_new = h_anchor + anchor_update
        
        # Update pairwise features (factor nodes)
        h_pairs_new = h_factor
        
        return h_meas_new, h_anchor_new, h_pairs_new


class FactorGraphNeuralNetworkV3(nn.Module):
    """
    Explicit Factor Node Version of Factor Graph Neural Network
    
    Architecture:
    1. Input encoder: encode raw features to node features
    2. Explicit factor GNN layers: multi-layer message passing
    3. Output decoder: decode node features to association scores
    
    Fully compatible with V2 interface, can be directly replaced
    """
    
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2,
                 use_temporal_gru=False, use_layer_gru=False,
                 aggregation='softmax', gamma=3.0, edge_mode='concat',
                 skip_connections=None):
        """
        Args:
            input_dim: Input feature dimension (default 5)
            hidden_dim: Hidden layer dimension (default 64)
            num_layers: Number of GNN layers (default 2)
            use_temporal_gru: Whether to use cross-frame GRU
            use_layer_gru: Whether to use intra-layer GRU
            aggregation: Aggregation method ('softmax', 'max', 'mean')
            gamma: Softmax temperature parameter (default 3.0)
            edge_mode: Edge feature mode ('diff', 'concat')
            skip_connections: Skip connection dict {target: source}
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_temporal_gru = use_temporal_gru
        self.use_layer_gru = use_layer_gru
        self.skip_connections = skip_connections or {}
        
        # 1. Input encoder
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. Explicit factor GNN layers
        self.gnn_layers = nn.ModuleList([
            ExplicitFactorGNNLayer(
                hidden_dim,
                use_gru=use_layer_gru,
                aggregation=aggregation,
                gamma=gamma,
                edge_mode=edge_mode
            ) for _ in range(num_layers)
        ])
        
        # 3. [Optional] Cross-frame global memory GRU
        if use_temporal_gru:
            self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.gru = None
        
        # 4. Output decoder - output association scores
        head_input_dim = hidden_dim * 2 if use_temporal_gru else hidden_dim
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, hybrid_input, hidden_state=None):
        """
        Forward propagation
        
        Input:
            hybrid_input: (Batch, M, K+1, 5) hybrid features
            hidden_state: (Batch, hidden_dim) memory from previous frame (optional)
            
        Output:
            logits: (Batch, M, K+1) association scores
            new_hidden_state: (Batch, hidden_dim) updated memory
        """
        B, M, K_plus_1, input_dim = hybrid_input.shape
        K = K_plus_1 - 1
        
        # A. Encode features
        x = self.input_encoder(hybrid_input)  # (B, M, K+1, hidden_dim)
        
        # B. Separate into measurement nodes, anchor nodes, and pairwise features
        # Extract pairwise features (excluding dustbin column)
        h_pairs = x[:, :, :K, :]  # (B, M, K, hidden_dim)

        # Initialize measurement node features (aggregate over anchors)
        h_meas = torch.mean(h_pairs, dim=2)  # (B, M, hidden_dim)

        # Initialize anchor node features (aggregate over measurements)
        h_anchor = torch.mean(h_pairs, dim=1)  # (B, K, hidden_dim)
        
        # C. Iterative message passing + Skip Connections
        layer_outputs_meas = []
        layer_outputs_anchor = []
        layer_outputs_pairs = []
        
        for idx, layer in enumerate(self.gnn_layers):
            h_meas, h_anchor, h_pairs = layer(h_meas, h_anchor, h_pairs)
            
            # Apply skip connection
            if idx in self.skip_connections:
                source_idx = self.skip_connections[idx]
                if source_idx < len(layer_outputs_meas):
                    h_meas = h_meas + layer_outputs_meas[source_idx]
                    h_anchor = h_anchor + layer_outputs_anchor[source_idx]
                    h_pairs = h_pairs + layer_outputs_pairs[source_idx]
            
            layer_outputs_meas.append(h_meas)
            layer_outputs_anchor.append(h_anchor)
            layer_outputs_pairs.append(h_pairs)
        
        # D. Reconstruct full feature matrix (including dustbin column)
        # Dustbin column: use measurement features
        h_dustbin = h_meas.unsqueeze(2)  # (B, M, 1, hidden_dim)
        x_full = torch.cat([h_pairs, h_dustbin], dim=2)  # (B, M, K+1, hidden_dim)
        
        # E. Choose different processing based on whether GRU is used
        if self.use_temporal_gru:
            # Use cross-frame GRU memory
            global_feat, _ = torch.max(x_full.view(B, -1, self.hidden_dim), dim=1)
            
            if hidden_state is None:
                hidden_state = torch.zeros_like(global_feat)
            
            new_hidden_state = self.gru(global_feat, hidden_state)
            
            # Feature fusion
            h_expanded = new_hidden_state.view(B, 1, 1, self.hidden_dim).expand(B, M, K_plus_1, self.hidden_dim)
            combined_feat = torch.cat([x_full, h_expanded], dim=-1)
        else:
            combined_feat = x_full
            new_hidden_state = None
        
        # F. Decode to Logits
        logits = self.head(combined_feat).squeeze(-1)  # (B, M, K+1)
        
        return logits, new_hidden_state


class JointDualHeadGNNV3(nn.Module):
    """
    Explicit Factor Node Version of Dual-Head GNN
    
    Uses explicit factor nodes for message passing, outputs quality head and association head
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
        
        # ===== 1. Shared feature extraction backbone =====
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Explicit factor GNN layers
        self.gnn_layers = nn.ModuleList([
            ExplicitFactorGNNLayer(
                hidden_dim,
                use_gru=use_layer_gru,
                aggregation=aggregation,
                gamma=gamma,
                edge_mode=edge_mode
            ) for _ in range(num_layers)
        ])
        
        # [Optional] Cross-frame node-level memory GRU
        if use_temporal_gru:
            self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.gru = None
        
        # ===== 2. Quality head =====
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # ===== 3. Association head =====
        self.association_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, hybrid_input, hidden_state=None):
        """
        Forward propagation
        
        Input:
            hybrid_input: (Batch, M, K+1, 5) hybrid features
            hidden_state: (Batch*M*K, hidden_dim) node-level memory (optional)
            
        Output:
            assoc_logits: (Batch, M, K) association scores
            quality_scores: (Batch, M) quality scores [0, 1]
            new_hidden_state: (Batch*M*K, hidden_dim) updated memory
        """
        B, M, K_plus_1, input_dim = hybrid_input.shape
        K = K_plus_1 - 1
        
        # A. Encode features
        x = self.input_encoder(hybrid_input)  # (B, M, K+1, hidden_dim)
        
        # B. Separate into measurement nodes, anchor nodes, and pairwise features
        h_pairs = x[:, :, :K, :]  # (B, M, K, hidden_dim)
        h_meas = torch.mean(h_pairs, dim=2)  # (B, M, hidden_dim)
        h_anchor = torch.mean(h_pairs, dim=1)  # (B, K, hidden_dim)
        
        # C. Iterative message passing + Skip Connections
        layer_outputs_meas = []
        layer_outputs_anchor = []
        layer_outputs_pairs = []
        
        for idx, layer in enumerate(self.gnn_layers):
            h_meas, h_anchor, h_pairs = layer(h_meas, h_anchor, h_pairs)
            
            # Apply skip connection
            if idx in self.skip_connections:
                source_idx = self.skip_connections[idx]
                if source_idx < len(layer_outputs_meas):
                    h_meas = h_meas + layer_outputs_meas[source_idx]
                    h_anchor = h_anchor + layer_outputs_anchor[source_idx]
                    h_pairs = h_pairs + layer_outputs_pairs[source_idx]
            
            layer_outputs_meas.append(h_meas)
            layer_outputs_anchor.append(h_anchor)
            layer_outputs_pairs.append(h_pairs)
        
        # D. Quality head inference (use measurement features)
        quality_scores = self.quality_head(h_meas).squeeze(-1)  # (B, M)
        
        # E. Association head inference (use pairwise features)
        # If using cross-frame GRU
        if self.use_temporal_gru:
            x_flat = h_pairs.reshape(-1, self.hidden_dim)
            
            if hidden_state is None:
                hidden_state = torch.zeros_like(x_flat)
            
            new_hidden_state = self.gru(x_flat, hidden_state)
            x_mem = new_hidden_state.view(B, M, K, self.hidden_dim)
            combined_feat = x_mem
        else:
            combined_feat = h_pairs
            new_hidden_state = None
        
        # Association head output
        assoc_logits = self.association_head(combined_feat).squeeze(-1)  # (B, M, K)
        
        return assoc_logits, quality_scores, new_hidden_state
