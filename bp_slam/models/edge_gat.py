# -*- coding: utf-8 -*-
"""
Edge-Conditioned Bipartite Graph Attention Network for Radio SLAM Data Association.

This network is specifically designed for multipath-assisted SLAM, where:
- Left nodes: Measurements (range observations)
- Right nodes: Anchors (physical/virtual landmarks)
- Edges: Fully connected bipartite graph with geometric residuals

The network outputs:
1. Matching probabilities: P(measurement_i -> anchor_j)
2. New anchor probabilities: P(measurement_i is from a new anchor)

Architecture:
    Input Encoders -> Edge-Conditioned GAT Layers -> Dual Output Heads
                                                     +-- Matching Head (M x N)
                                                     +-- New Anchor Head (M x 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MLPEncoder(nn.Module):
    """MLP encoder to project input features to hidden dimension."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class EdgeConditionedGATLayer(nn.Module):
    """
    Edge-Conditioned Graph Attention Layer for Bipartite Graphs.
    
    Attention coefficient computation:
        alpha_ij = LeakyReLU(a^T [W1*h_i || W2*h_j || W3*e_ij])
    
    This layer incorporates edge features (geometric residuals) directly
    into the attention mechanism, which is crucial for SLAM data association.
    
    Args:
        meas_dim: Dimension of measurement node features
        anchor_dim: Dimension of anchor node features
        edge_dim: Dimension of edge features
        out_dim: Output dimension for updated node features
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        meas_dim: int,
        anchor_dim: int,
        edge_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        # Linear projections for attention computation
        # W1 for measurement nodes (source)
        self.W_meas = nn.Linear(meas_dim, out_dim, bias=False)
        # W2 for anchor nodes (target)
        self.W_anchor = nn.Linear(anchor_dim, out_dim, bias=False)
        # W3 for edge features
        self.W_edge = nn.Linear(edge_dim, out_dim, bias=False)
        
        # Attention vector a (per head)
        # Concatenated dimension: head_dim * 3 (meas + anchor + edge)
        self.attention = nn.Parameter(torch.zeros(num_heads, 3 * self.head_dim))
        nn.init.xavier_uniform_(self.attention.view(num_heads, -1))
        
        # Output projection
        self.out_proj = nn.Linear(out_dim, out_dim)
        
        # Normalization and dropout
        self.layer_norm_meas = nn.LayerNorm(out_dim)
        self.layer_norm_anchor = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(
        self,
        h_meas: torch.Tensor,      # (M, meas_dim)
        h_anchor: torch.Tensor,    # (N, anchor_dim)
        e_attr: torch.Tensor,      # (M*N, edge_dim)
        edge_index: torch.Tensor   # (2, M*N) - [src_idx, tgt_idx]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of Edge-Conditioned GAT Layer.
        
        Args:
            h_meas: Measurement node features (M, meas_dim)
            h_anchor: Anchor node features (N, anchor_dim)
            e_attr: Edge attributes (M*N, edge_dim)
            edge_index: Edge indices [source, target] (2, M*N)
        
        Returns:
            h_meas_out: Updated measurement features (M, out_dim)
            h_anchor_out: Updated anchor features (N, out_dim)
            attention_weights: Attention coefficients (M*N,) or (num_heads, M*N)
        """
        M = h_meas.size(0)
        N = h_anchor.size(0)
        num_edges = edge_index.size(1)
        
        # Project node features
        h_meas_proj = self.W_meas(h_meas)      # (M, out_dim)
        h_anchor_proj = self.W_anchor(h_anchor)  # (N, out_dim)
        e_proj = self.W_edge(e_attr)            # (M*N, out_dim)
        
        # Gather features for each edge
        src_idx = edge_index[0]  # (M*N,) - measurement indices
        tgt_idx = edge_index[1]  # (M*N,) - anchor indices
        
        h_src = h_meas_proj[src_idx]      # (M*N, out_dim)
        h_tgt = h_anchor_proj[tgt_idx]    # (M*N, out_dim)
        
        # Reshape for multi-head attention
        h_src = h_src.view(num_edges, self.num_heads, self.head_dim)  # (E, H, D)
        h_tgt = h_tgt.view(num_edges, self.num_heads, self.head_dim)  # (E, H, D)
        e_proj = e_proj.view(num_edges, self.num_heads, self.head_dim)  # (E, H, D)
        
        # Concatenate for attention: [h_src || h_tgt || e_proj]
        concat_features = torch.cat([h_src, h_tgt, e_proj], dim=-1)  # (E, H, 3*D)
        
        # Compute attention scores: a^T [W1*h_i || W2*h_j || W3*e_ij]
        # attention shape: (H, 3*D) -> expand to (1, H, 3*D) for broadcasting
        attn_scores = (concat_features * self.attention.unsqueeze(0)).sum(dim=-1)  # (E, H)
        attn_scores = self.leaky_relu(attn_scores)
        
        # Softmax over all edges from same source (measurement)
        # Need to normalize per measurement node
        attn_weights = self._edge_softmax(attn_scores, src_idx, M)  # (E, H)
        attn_weights = self.dropout(attn_weights)
        
        # ================================================================
        # Update measurement nodes (aggregate from anchors)
        # Each measurement aggregates information from all connected anchors
        # ================================================================
        # Weighted anchor features
        weighted_anchor = h_tgt * attn_weights.unsqueeze(-1)  # (E, H, D)
        weighted_anchor = weighted_anchor.view(num_edges, self.out_dim)  # (E, out_dim)
        
        # Aggregate: sum over edges for each measurement
        h_meas_agg = torch.zeros(M, self.out_dim, device=h_meas.device)
        h_meas_agg.scatter_add_(0, src_idx.unsqueeze(-1).expand(-1, self.out_dim), weighted_anchor)
        
        # Residual connection and normalization
        h_meas_out = self.layer_norm_meas(h_meas_proj + self.out_proj(h_meas_agg))
        
        # ================================================================
        # Update anchor nodes (aggregate from measurements)
        # Each anchor aggregates information from all connected measurements
        # ================================================================
        # Recompute attention for anchor update (reverse direction)
        attn_weights_rev = self._edge_softmax(attn_scores, tgt_idx, N)  # (E, H)
        
        weighted_meas = h_src * attn_weights_rev.unsqueeze(-1)  # (E, H, D)
        weighted_meas = weighted_meas.view(num_edges, self.out_dim)  # (E, out_dim)
        
        h_anchor_agg = torch.zeros(N, self.out_dim, device=h_anchor.device)
        h_anchor_agg.scatter_add_(0, tgt_idx.unsqueeze(-1).expand(-1, self.out_dim), weighted_meas)
        
        h_anchor_out = self.layer_norm_anchor(h_anchor_proj + self.out_proj(h_anchor_agg))
        
        # Return mean attention weights across heads for output
        attn_weights_mean = attn_weights.mean(dim=-1)  # (E,)
        
        return h_meas_out, h_anchor_out, attn_weights_mean
    
    def _edge_softmax(
        self,
        scores: torch.Tensor,   # (E, H)
        index: torch.Tensor,    # (E,) - node indices for grouping
        num_nodes: int
    ) -> torch.Tensor:
        """
        Compute softmax over edges grouped by node index.
        
        For each node i, compute softmax over all edges connected to i.
        """
        # Subtract max for numerical stability (per node)
        max_scores = torch.zeros(num_nodes, scores.size(1), device=scores.device)
        max_scores.scatter_reduce_(
            0, 
            index.unsqueeze(-1).expand(-1, scores.size(1)), 
            scores, 
            reduce='amax',
            include_self=False
        )
        scores = scores - max_scores[index]
        
        # Compute exp
        exp_scores = torch.exp(scores)
        
        # Sum per node
        sum_exp = torch.zeros(num_nodes, scores.size(1), device=scores.device)
        sum_exp.scatter_add_(0, index.unsqueeze(-1).expand(-1, scores.size(1)), exp_scores)
        
        # Normalize
        return exp_scores / (sum_exp[index] + 1e-10)


class MatchingHead(nn.Module):
    """
    Output head for matching probabilities.
    
    Takes updated edge features and outputs P(meas_i -> anchor_j).
    Output shape: (M, N) matrix where each row sums to ~1 (soft assignment).
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),  # h_meas + h_anchor + e_attr
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)  # Output single score per edge
        )
    
    def forward(
        self,
        h_meas: torch.Tensor,      # (M, hidden_dim)
        h_anchor: torch.Tensor,    # (N, hidden_dim)
        e_attr: torch.Tensor,      # (M*N, hidden_dim)
        edge_index: torch.Tensor,  # (2, M*N)
        M: int,
        N: int,
        use_sigmoid: bool = True
    ) -> torch.Tensor:
        """
        Compute matching scores.
        
        Args:
            h_meas: Measurement node features (M, hidden_dim)
            h_anchor: Anchor node features (N, hidden_dim)
            e_attr: Edge attributes (M*N, hidden_dim)
            edge_index: Edge indices [source, target] (2, M*N)
            M: Number of measurements
            N: Number of anchors
            use_sigmoid: If True, use sigmoid (independent probabilities per edge).
                        If False, use softmax (each measurement must match one anchor).
        
        Returns:
            match_scores: (M, N) matrix of matching probabilities
        """
        src_idx = edge_index[0]
        tgt_idx = edge_index[1]
        
        # Gather node features for each edge
        h_src = h_meas[src_idx]      # (M*N, hidden_dim)
        h_tgt = h_anchor[tgt_idx]    # (M*N, hidden_dim)
        
        # Concatenate all features
        edge_features = torch.cat([h_src, h_tgt, e_attr], dim=-1)  # (M*N, hidden_dim*3)
        
        # Compute scores
        scores = self.mlp(edge_features).squeeze(-1)  # (M*N,)
        
        # Reshape to (M, N) matrix
        match_matrix = scores.view(M, N)
        
        # Apply activation function
        if use_sigmoid:
            # Sigmoid: Independent probability per edge
            # Allows measurement to match none, one, or potentially multiple anchors
            # Better for handling clutter (measurements from no anchor)
            match_probs = torch.sigmoid(match_matrix)
        else:
            # Softmax: Each measurement must match exactly one anchor
            # P(anchor_j | measurement_i) sums to 1 over j
            match_probs = F.softmax(match_matrix, dim=-1)
        
        return match_probs


class NewAnchorHead(nn.Module):
    """
    Output head for new anchor probabilities.
    
    Takes updated measurement features (which have "seen" all anchors)
    and outputs P(measurement_i is from a NEW anchor not in current map).
    
    Physical intuition: If a measurement doesn't match any existing anchor
    well (after GAT aggregation), it's likely a new landmark.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output probability in [0, 1]
        )
    
    def forward(self, h_meas: torch.Tensor) -> torch.Tensor:
        """
        Compute new anchor probabilities.
        
        Args:
            h_meas: Updated measurement features (M, hidden_dim)
        
        Returns:
            new_anchor_probs: (M,) vector of probabilities
        """
        return self.mlp(h_meas).squeeze(-1)


class EdgeConditionedBipartiteGAT(nn.Module):
    """
    Edge-Conditioned Bipartite Graph Attention Network for SLAM Data Association.
    
    Architecture:
        1. Input Encoders: Project raw features to hidden dimension
           - Measurement encoder: (z, R) -> hidden_dim
           - Anchor encoder: (z_hat, P, p_exist) -> hidden_dim
           - Edge encoder: |z - z_hat| -> hidden_dim
        
        2. GAT Backbone: Multiple Edge-Conditioned GAT layers
           - Attention incorporates edge features (geometric residuals)
           - Message passing updates both measurement and anchor nodes
        
        3. Output Heads:
           - Matching Head: P(meas_i -> anchor_j) as (M, N) matrix
           - New Anchor Head: P(meas_i is new) as (M,) vector
    
    Args:
        meas_input_dim: Input dimension for measurements (default: 2 for [z, R])
        anchor_input_dim: Input dimension for anchors (default: 3 for [z_hat, P, p_exist])
        edge_input_dim: Input dimension for edges (default: 1 for |z - z_hat|)
        hidden_dim: Hidden dimension for all layers (default: 64)
        num_layers: Number of GAT layers (default: 2)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        meas_input_dim: int = 2,
        anchor_input_dim: int = 3,
        edge_input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # ================================================================
        # Input Encoders
        # ================================================================
        self.meas_encoder = MLPEncoder(meas_input_dim, hidden_dim, hidden_dim, dropout)
        self.anchor_encoder = MLPEncoder(anchor_input_dim, hidden_dim, hidden_dim, dropout)
        self.edge_encoder = MLPEncoder(edge_input_dim, hidden_dim, hidden_dim, dropout)
        
        # ================================================================
        # GAT Backbone
        # ================================================================
        self.gat_layers = nn.ModuleList([
            EdgeConditionedGATLayer(
                meas_dim=hidden_dim,
                anchor_dim=hidden_dim,
                edge_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # ================================================================
        # Output Heads
        # ================================================================
        self.matching_head = MatchingHead(hidden_dim, dropout)
        self.new_anchor_head = NewAnchorHead(hidden_dim, dropout)
    
    def forward(
        self,
        x_meas: torch.Tensor,      # (M, 2) - [z, R]
        x_anchor: torch.Tensor,    # (N, 3) - [z_hat, P, p_exist]
        e_attr: torch.Tensor,      # (M*N, 1) - [|z - z_hat|]
        edge_index: torch.Tensor,  # (2, M*N) - [src, tgt]
        use_sigmoid: bool = True   # Use sigmoid (independent) or softmax (exclusive)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Edge-Conditioned Bipartite GAT.
        
        Args:
            x_meas: Measurement features (M, 2)
                    [range_measurement, measurement_variance]
            x_anchor: Anchor features (N, 3)
                      [predicted_range, prediction_variance, existence_probability]
            e_attr: Edge attributes (M*N, 1)
                    [geometric_residual |z - z_hat|]
            edge_index: Edge connectivity (2, M*N)
                        [[meas_indices], [anchor_indices]]
            use_sigmoid: If True, use sigmoid activation (independent probabilities).
                        If False, use softmax (each measurement matches one anchor).
                        Default: True (recommended for handling clutter)
        
        Returns:
            match_probs: (M, N) matrix of matching probabilities
                         P(measurement_i comes from anchor_j)
            new_anchor_probs: (M,) vector of new anchor probabilities
                              P(measurement_i is from a new anchor)
            attention_weights: (M*N,) attention weights from last GAT layer
        """
        M = x_meas.size(0)
        N = x_anchor.size(0)
        
        # ================================================================
        # Encode inputs to hidden dimension
        # ================================================================
        h_meas = self.meas_encoder(x_meas)        # (M, hidden_dim)
        h_anchor = self.anchor_encoder(x_anchor)  # (N, hidden_dim)
        h_edge = self.edge_encoder(e_attr)        # (M*N, hidden_dim)
        
        # ================================================================
        # Pass through GAT layers
        # ================================================================
        attn_weights = None
        for gat_layer in self.gat_layers:
            h_meas, h_anchor, attn_weights = gat_layer(
                h_meas, h_anchor, h_edge, edge_index
            )
        
        # ================================================================
        # Output heads
        # ================================================================
        # Matching probabilities: P(meas_i -> anchor_j)
        match_probs = self.matching_head(
            h_meas, h_anchor, h_edge, edge_index, M, N, use_sigmoid=use_sigmoid
        )
        
        # New anchor probabilities: P(meas_i is new)
        new_anchor_probs = self.new_anchor_head(h_meas)
        
        return match_probs, new_anchor_probs, attn_weights
    
    @staticmethod
    def build_bipartite_edge_index(M: int, N: int, device: torch.device = None) -> torch.Tensor:
        """
        Build fully-connected bipartite edge index.
        
        Creates edges from every measurement to every anchor.
        
        Args:
            M: Number of measurements
            N: Number of anchors
            device: Target device
        
        Returns:
            edge_index: (2, M*N) tensor
                        edge_index[0] = source (measurement) indices
                        edge_index[1] = target (anchor) indices
        """
        # Create mesh grid
        src_idx = torch.arange(M, device=device).repeat_interleave(N)  # [0,0,0,1,1,1,2,2,2,...]
        tgt_idx = torch.arange(N, device=device).repeat(M)            # [0,1,2,0,1,2,0,1,2,...]
        
        return torch.stack([src_idx, tgt_idx], dim=0)
    
    @staticmethod
    def compute_edge_attr(
        x_meas: torch.Tensor,      # (M, 2) - [z, R]
        x_anchor: torch.Tensor,    # (N, 3) - [z_hat, P, p_exist]
        edge_index: torch.Tensor   # (2, M*N)
    ) -> torch.Tensor:
        """
        Compute edge attributes (geometric residuals).
        
        Args:
            x_meas: Measurement features (M, 2)
            x_anchor: Anchor features (N, 3)
            edge_index: Edge connectivity (2, M*N)
        
        Returns:
            e_attr: Edge attributes (M*N, 1) containing |z - z_hat|
        """
        src_idx = edge_index[0]  # (M*N,)
        tgt_idx = edge_index[1]  # (M*N,)
        
        z = x_meas[src_idx, 0]        # (M*N,) measured range
        z_hat = x_anchor[tgt_idx, 0]  # (M*N,) predicted range
        
        residual = torch.abs(z - z_hat)  # (M*N,)
        
        return residual.unsqueeze(-1)  # (M*N, 1)


# ============================================================================
# Loss Functions
# ============================================================================

class DataAssociationLoss(nn.Module):
    """
    Combined loss for SLAM data association.
    
    Loss = lambda1 * MatchingLoss + lambda2 * NewAnchorLoss
    
    Where:
        - MatchingLoss: Cross-entropy for measurement-anchor matching
        - NewAnchorLoss: Binary cross-entropy for new anchor detection
    """
    
    def __init__(self, matching_weight: float = 1.0, new_anchor_weight: float = 1.0):
        super().__init__()
        self.matching_weight = matching_weight
        self.new_anchor_weight = new_anchor_weight
    
    def forward(
        self,
        match_probs: torch.Tensor,       # (M, N) predicted matching probs
        new_anchor_probs: torch.Tensor,  # (M,) predicted new anchor probs
        match_labels: torch.Tensor,      # (M,) ground truth anchor indices (-1 for clutter)
        num_anchors: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            match_probs: Predicted matching probabilities (M, N)
            new_anchor_probs: Predicted new anchor probabilities (M,)
            match_labels: Ground truth labels (M,)
                          label >= 0: measurement from anchor[label]
                          label == -1: clutter (not from any known anchor)
            num_anchors: Number of anchors N
        
        Returns:
            total_loss: Combined loss
            matching_loss: Cross-entropy loss for matching
            new_anchor_loss: BCE loss for new anchor detection
        """
        M = match_probs.size(0)
        device = match_probs.device
        
        # ================================================================
        # Matching Loss (only for non-clutter measurements)
        # ================================================================
        valid_mask = match_labels >= 0  # Non-clutter measurements
        
        if valid_mask.sum() > 0:
            valid_probs = match_probs[valid_mask]  # (num_valid, N)
            valid_labels = match_labels[valid_mask]  # (num_valid,)
            
            # Cross-entropy loss
            matching_loss = F.cross_entropy(
                torch.log(valid_probs + 1e-10),
                valid_labels
            )
        else:
            matching_loss = torch.tensor(0.0, device=device)
        
        # ================================================================
        # New Anchor Loss (clutter detection as proxy for "new anchor")
        # In simulation, clutter = measurement not from any known anchor
        # ================================================================
        # Label: 1 if clutter (new), 0 if from known anchor
        new_anchor_labels = (match_labels == -1).float()  # (M,)
        
        new_anchor_loss = F.binary_cross_entropy(
            new_anchor_probs,
            new_anchor_labels
        )
        
        # ================================================================
        # Combined loss
        # ================================================================
        total_loss = (
            self.matching_weight * matching_loss +
            self.new_anchor_weight * new_anchor_loss
        )
        
        return total_loss, matching_loss, new_anchor_loss


# ============================================================================
# Utility Functions
# ============================================================================

def prepare_gat_input(
    measurements: torch.Tensor,    # (2, M) - [distances, variances]
    anchor_predictions: dict,      # {'z_hat': (N,), 'P': (N,), 'p_exist': (N,)}
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare input tensors for EdgeConditionedBipartiteGAT.
    
    Args:
        measurements: Measurement data (2, M) where row0=distance, row1=variance
        anchor_predictions: Dictionary with anchor predictions
            - 'z_hat': Predicted distances (N,)
            - 'P': Prediction variances (N,)
            - 'p_exist': Existence probabilities (N,)
        device: Target device
    
    Returns:
        x_meas: (M, 2) measurement features
        x_anchor: (N, 3) anchor features
        e_attr: (M*N, 1) edge attributes
        edge_index: (2, M*N) edge connectivity
    """
    if device is None:
        device = measurements.device if torch.is_tensor(measurements) else torch.device('cpu')
    
    # Convert to tensors if needed
    if not torch.is_tensor(measurements):
        measurements = torch.tensor(measurements, dtype=torch.float32, device=device)
    
    M = measurements.size(1)
    
    # Measurement features: [z, R]
    x_meas = measurements.T  # (M, 2)
    
    # Anchor features: [z_hat, P, p_exist]
    z_hat = torch.tensor(anchor_predictions['z_hat'], dtype=torch.float32, device=device)
    P = torch.tensor(anchor_predictions['P'], dtype=torch.float32, device=device)
    p_exist = torch.tensor(anchor_predictions['p_exist'], dtype=torch.float32, device=device)
    N = z_hat.size(0)
    
    x_anchor = torch.stack([z_hat, P, p_exist], dim=-1)  # (N, 3)
    
    # Build edge index
    edge_index = EdgeConditionedBipartiteGAT.build_bipartite_edge_index(M, N, device)
    
    # Compute edge attributes
    e_attr = EdgeConditionedBipartiteGAT.compute_edge_attr(x_meas, x_anchor, edge_index)
    
    return x_meas, x_anchor, e_attr, edge_index


if __name__ == '__main__':
    # Quick test
    print("=" * 60)
    print("Testing EdgeConditionedBipartiteGAT")
    print("=" * 60)
    
    # Create model
    model = EdgeConditionedBipartiteGAT(
        meas_input_dim=2,
        anchor_input_dim=3,
        edge_input_dim=1,
        hidden_dim=64,
        num_layers=2,
        num_heads=4
    )
    
    print(f"\nModel architecture:")
    print(model)
    
    # Test input
    M, N = 5, 3  # 5 measurements, 3 anchors
    
    x_meas = torch.randn(M, 2)      # [z, R]
    x_anchor = torch.randn(N, 3)    # [z_hat, P, p_exist]
    x_anchor[:, 2] = torch.sigmoid(x_anchor[:, 2])  # p_exist in [0, 1]
    
    edge_index = EdgeConditionedBipartiteGAT.build_bipartite_edge_index(M, N)
    e_attr = EdgeConditionedBipartiteGAT.compute_edge_attr(x_meas, x_anchor, edge_index)
    
    print(f"\nInput shapes:")
    print(f"  x_meas: {x_meas.shape}")
    print(f"  x_anchor: {x_anchor.shape}")
    print(f"  e_attr: {e_attr.shape}")
    print(f"  edge_index: {edge_index.shape}")
    
    # Forward pass
    match_probs, new_anchor_probs, attn_weights = model(x_meas, x_anchor, e_attr, edge_index)
    
    print(f"\nOutput shapes:")
    print(f"  match_probs: {match_probs.shape} (should be {M}x{N})")
    print(f"  new_anchor_probs: {new_anchor_probs.shape} (should be {M})")
    print(f"  attn_weights: {attn_weights.shape} (should be {M*N})")
    
    print(f"\nMatch probabilities (rows should sum to ~1):")
    print(match_probs)
    print(f"Row sums: {match_probs.sum(dim=1)}")
    
    print(f"\nNew anchor probabilities (should be in [0, 1]):")
    print(new_anchor_probs)
    
    # Test loss computation
    print("\n" + "=" * 60)
    print("Testing DataAssociationLoss")
    print("=" * 60)
    
    loss_fn = DataAssociationLoss()
    
    # Ground truth labels: [0, 1, -1, 2, -1] means:
    #   meas 0 -> anchor 0
    #   meas 1 -> anchor 1
    #   meas 2 -> clutter
    #   meas 3 -> anchor 2
    #   meas 4 -> clutter
    labels = torch.tensor([0, 1, -1, 2, -1])
    
    total_loss, match_loss, new_loss = loss_fn(match_probs, new_anchor_probs, labels, N)
    
    print(f"\nLabels: {labels}")
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"  Matching loss: {match_loss.item():.4f}")
    print(f"  New anchor loss: {new_loss.item():.4f}")
    
    print("\nAll tests passed!")
