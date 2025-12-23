# -*- coding: utf-8 -*-
"""
Data preparation utilities for GAT-based SLAM data association.

This module provides functions to convert SLAM simulation data into
the format required by EdgeConditionedBipartiteGAT.

Key functions:
    - prepare_training_sample: Convert one time step to GAT input format
    - SLAMDataset: PyTorch Dataset for training
    - create_data_loader: Create DataLoader for training
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any


def prepare_training_sample(
    measurements: np.ndarray,      # (2, M) - [distances, variances]
    anchor_predictions: Dict,      # Predictions for each anchor
    labels: np.ndarray,            # (M,) - source_id for each measurement
    device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """
    Prepare a single training sample for EdgeConditionedBipartiteGAT.
    
    Args:
        measurements: Measurement data (2, M)
                     row 0: measured distances
                     row 1: measurement variances
        anchor_predictions: Dictionary with anchor predictions
            - 'z_hat': Predicted distances to each anchor (N,)
            - 'P': Prediction variances (N,)
            - 'p_exist': Existence probabilities (N,)
        labels: Ground truth labels (M,)
                label >= 0: measurement from anchor[label]
                label == -1: clutter
        device: Target device
    
    Returns:
        sample: Dictionary containing:
            - 'x_meas': (M, 2) measurement features
            - 'x_anchor': (N, 3) anchor features
            - 'e_attr': (M*N, 1) edge attributes
            - 'edge_index': (2, M*N) edge connectivity
            - 'labels': (M,) ground truth labels
            - 'M': number of measurements
            - 'N': number of anchors
    """
    if device is None:
        device = torch.device('cpu')
    
    M = measurements.shape[1]
    
    # ================================================================
    # Node A: Measurement features [z, R]
    # ================================================================
    x_meas = torch.tensor(measurements.T, dtype=torch.float32, device=device)  # (M, 2)
    
    # ================================================================
    # Node B: Anchor features [z_hat, P, p_exist]
    # ================================================================
    z_hat = np.array(anchor_predictions['z_hat'], dtype=np.float32)
    P = np.array(anchor_predictions['P'], dtype=np.float32)
    p_exist = np.array(anchor_predictions['p_exist'], dtype=np.float32)
    N = len(z_hat)
    
    x_anchor = torch.tensor(
        np.stack([z_hat, P, p_exist], axis=-1),
        dtype=torch.float32,
        device=device
    )  # (N, 3)
    
    # ================================================================
    # Edge index: Fully connected bipartite graph
    # ================================================================
    # src_idx: [0,0,0,...,1,1,1,...,M-1,M-1,M-1]
    # tgt_idx: [0,1,2,...,0,1,2,...,0,1,2,...]
    src_idx = torch.arange(M, device=device).repeat_interleave(N)
    tgt_idx = torch.arange(N, device=device).repeat(M)
    edge_index = torch.stack([src_idx, tgt_idx], dim=0)  # (2, M*N)
    
    # ================================================================
    # Edge attributes: Geometric residual |z - z_hat|
    # ================================================================
    z = x_meas[src_idx, 0]          # (M*N,) measured distances
    z_hat_edges = x_anchor[tgt_idx, 0]  # (M*N,) predicted distances
    residual = torch.abs(z - z_hat_edges)  # (M*N,)
    e_attr = residual.unsqueeze(-1)  # (M*N, 1)
    
    # ================================================================
    # Labels
    # ================================================================
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    
    return {
        'x_meas': x_meas,
        'x_anchor': x_anchor,
        'e_attr': e_attr,
        'edge_index': edge_index,
        'labels': labels_tensor,
        'M': M,
        'N': N
    }


def compute_anchor_predictions_from_particles(
    agent_particles: np.ndarray,    # (4, num_particles) or (2, num_particles)
    anchor_particles: Dict,         # Anchor particle data
    anchor_idx: int,
    parameters: Dict
) -> Dict[str, float]:
    """
    Compute anchor predictions (z_hat, P, p_exist) from particle representation.
    
    This is what the SLAM algorithm computes before data association.
    
    Args:
        agent_particles: Agent state particles (4, num_particles) [x, y, vx, vy]
        anchor_particles: Dictionary with anchor particle data
            - 'x': (2, num_particles) anchor position particles
            - 'w': (num_particles,) weights
            - 'posteriorExistence': existence probability
        anchor_idx: Index of the anchor
        parameters: Parameter dictionary
    
    Returns:
        predictions: Dictionary with z_hat, P, p_exist
    """
    # Agent position (mean estimate)
    if agent_particles.shape[0] == 4:
        agent_pos = agent_particles[:2, :].mean(axis=1)  # (2,)
    else:
        agent_pos = agent_particles.mean(axis=1)  # (2,)
    
    # Anchor position particles
    anchor_pos = anchor_particles['x']  # (2, num_particles)
    weights = anchor_particles['w']     # (num_particles,)
    
    # Compute predicted distances for all particles
    dx = anchor_pos[0, :] - agent_pos[0]
    dy = anchor_pos[1, :] - agent_pos[1]
    distances = np.sqrt(dx**2 + dy**2)
    
    # Weighted mean and variance
    z_hat = np.average(distances, weights=weights)
    P = np.average((distances - z_hat)**2, weights=weights)
    
    # Existence probability
    p_exist = anchor_particles.get('posteriorExistence', 1.0)
    
    return {
        'z_hat': z_hat,
        'P': P,
        'p_exist': p_exist
    }


class SLAMDataset(Dataset):
    """
    PyTorch Dataset for SLAM data association training.
    
    Each sample corresponds to one time step of the SLAM simulation,
    containing measurements, anchor predictions, and ground truth labels.
    """
    
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        device: torch.device = None
    ):
        """
        Initialize dataset.
        
        Args:
            samples: List of sample dictionaries, each containing:
                - 'measurements': (2, M) array
                - 'anchor_predictions': dict with z_hat, P, p_exist for each anchor
                - 'labels': (M,) array of source_ids
            device: Target device
        """
        self.samples = samples
        self.device = device if device else torch.device('cpu')
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return prepare_training_sample(
            measurements=sample['measurements'],
            anchor_predictions=sample['anchor_predictions'],
            labels=sample['labels'],
            device=self.device
        )


def collate_variable_size(batch: List[Dict]) -> List[Dict]:
    """
    Collate function for variable-size graphs.
    
    Since M and N can vary across samples, we don't batch them together.
    Instead, return list of samples for manual iteration.
    """
    return batch


def create_data_loader(
    samples: List[Dict],
    batch_size: int = 1,
    shuffle: bool = True,
    device: torch.device = None
) -> DataLoader:
    """
    Create DataLoader for SLAM training.
    
    Note: Due to variable graph sizes (M, N differ per sample),
    batch_size=1 is recommended. For larger batch training,
    use padding or graph batching techniques.
    
    Args:
        samples: List of sample dictionaries
        batch_size: Batch size (default 1 for variable sizes)
        shuffle: Whether to shuffle data
        device: Target device
    
    Returns:
        DataLoader instance
    """
    dataset = SLAMDataset(samples, device)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_variable_size
    )


def generate_training_data_from_simulation(
    target_trajectory: np.ndarray,    # (2, num_steps)
    data_va: List[Dict],              # Virtual anchor data
    parameters: Dict,
    num_samples: int = None
) -> List[Dict]:
    """
    Generate training data from SLAM simulation.
    
    This function runs the simulation and collects training samples
    with ground truth labels.
    
    Args:
        target_trajectory: Agent trajectory (2, num_steps)
        data_va: Virtual anchor data list
        parameters: SLAM parameters
        num_samples: Number of samples to generate (default: all steps)
    
    Returns:
        samples: List of training sample dictionaries
    """
    from bp_slam.utils.label_generator import generate_measurements_with_labels
    
    # Generate measurements with labels
    measurements, labels = generate_measurements_with_labels(
        target_trajectory, data_va, parameters
    )
    
    num_steps = len(measurements)
    num_sensors = len(measurements[0])
    
    if num_samples is None:
        num_samples = num_steps
    
    samples = []
    
    for step in range(min(num_samples, num_steps)):
        for sensor in range(num_sensors):
            meas = measurements[step][sensor]
            lab = labels[step][sensor]
            
            if meas.shape[1] == 0:
                continue  # Skip empty measurements
            
            # Get anchor positions for this sensor
            anchor_positions = data_va[sensor]['positions']  # (2, N)
            visibility = data_va[sensor]['visibility']       # (N, num_steps)
            
            # Agent position at this step
            agent_pos = target_trajectory[:, step]
            
            # Compute "predictions" for each anchor
            # In real SLAM, this comes from particle filter
            # For training data generation, we use ground truth + noise
            anchor_predictions = {
                'z_hat': [],
                'P': [],
                'p_exist': []
            }
            
            num_anchors = anchor_positions.shape[1]
            for anchor_idx in range(num_anchors):
                # True distance
                dx = anchor_positions[0, anchor_idx] - agent_pos[0]
                dy = anchor_positions[1, anchor_idx] - agent_pos[1]
                true_dist = np.sqrt(dx**2 + dy**2)
                
                # Add some prediction noise (simulate particle filter uncertainty)
                pred_noise = np.random.randn() * 0.1
                z_hat = true_dist + pred_noise
                
                # Prediction variance (simplified)
                P = parameters['measurementVariance'] * 2
                
                # Existence probability (1 if visible, 0.5 otherwise)
                p_exist = 1.0 if visibility[anchor_idx, step] else 0.5
                
                anchor_predictions['z_hat'].append(z_hat)
                anchor_predictions['P'].append(P)
                anchor_predictions['p_exist'].append(p_exist)
            
            samples.append({
                'measurements': meas,
                'anchor_predictions': anchor_predictions,
                'labels': lab,
                'step': step,
                'sensor': sensor
            })
    
    return samples


# ============================================================================
# Inference Function (Drop-in replacement for BP)
# ============================================================================

def data_association_gat(
    model,
    measurements: np.ndarray,           # (2, M) - [distances, variances]
    predicted_measurements: np.ndarray, # (N,) - predicted distances
    predicted_uncertainties: np.ndarray,# (N,) - prediction variances
    existence_probs: np.ndarray,        # (N,) - anchor existence probabilities
    device: torch.device = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GAT-based data association (drop-in replacement for BP).
    
    This function has the same interface as the BP data association,
    allowing direct substitution in the SLAM pipeline.
    
    Comparison with BP:
    
        BP Flow:
            measurements + predictions ¡ú Gaussian likelihoods ¡ú BP iterations ¡ú probs
        
        GAT Flow:
            measurements + predictions ¡ú Graph features ¡ú Single forward pass ¡ú probs
    
    Args:
        model: Trained EdgeConditionedBipartiteGAT model
        measurements: Measurement data (2, M)
                     row 0: measured distances z
                     row 1: measurement variances R
        predicted_measurements: Predicted distances z_hat for each anchor (N,)
        predicted_uncertainties: Prediction variances P for each anchor (N,)
        existence_probs: Existence probabilities p_exist for each anchor (N,)
        device: Computation device (default: model's device)
    
    Returns:
        match_probs: (M, N) matrix of association probabilities
                     P(measurement_i comes from anchor_j)
        new_anchor_probs: (M,) vector of new anchor probabilities
                          P(measurement_i is from a new/unknown anchor)
    
    Usage (replacing BP in SLAM):
        # Before (BP):
        assoc_prob, assoc_prob_new, _, _ = calculate_association_probabilities_ga(
            measurements, predicted_meas, predicted_uncert, weights, new_input, params
        )
        
        # After (GAT):
        match_probs, new_probs = data_association_gat(
            model, measurements, predicted_meas, predicted_uncert, exist_probs
        )
    """
    from bp_slam.models.edge_gat import EdgeConditionedBipartiteGAT
    
    if device is None:
        device = next(model.parameters()).device
    
    # Handle empty inputs
    if measurements is None or measurements.size == 0:
        M = 0
    else:
        M = measurements.shape[1]
    
    N = len(predicted_measurements)
    
    if M == 0 or N == 0:
        return np.zeros((M, N)), np.zeros(M)
    
    # ================================================================
    # Convert to GAT input format
    # ================================================================
    
    # Node A: Measurements [z, R]
    x_meas = torch.tensor(measurements.T, dtype=torch.float32, device=device)  # (M, 2)
    
    # Node B: Anchors [z_hat, P, p_exist]
    x_anchor = torch.tensor(
        np.stack([
            predicted_measurements,
            predicted_uncertainties,
            existence_probs
        ], axis=-1),
        dtype=torch.float32,
        device=device
    )  # (N, 3)
    
    # Build fully-connected bipartite graph
    edge_index = EdgeConditionedBipartiteGAT.build_bipartite_edge_index(M, N, device)
    
    # Edge attributes: geometric residuals |z - z_hat|
    e_attr = EdgeConditionedBipartiteGAT.compute_edge_attr(x_meas, x_anchor, edge_index)
    
    # ================================================================
    # Forward pass (no gradients needed for inference)
    # ================================================================
    model.eval()
    with torch.no_grad():
        match_probs, new_anchor_probs, _ = model(x_meas, x_anchor, e_attr, edge_index)
    
    # ================================================================
    # Convert back to numpy
    # ================================================================
    match_probs_np = match_probs.cpu().numpy()        # (M, N)
    new_anchor_probs_np = new_anchor_probs.cpu().numpy()  # (M,)
    
    return match_probs_np, new_anchor_probs_np


def convert_gat_output_to_bp_format(
    match_probs: np.ndarray,      # (M, N) from GAT
    new_anchor_probs: np.ndarray, # (M,) from GAT
    detection_probability: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert GAT output to BP output format for compatibility.
    
    BP returns:
        - assoc_prob_existing: (M+1, N) where row 0 is undetected prob
        - assoc_prob_new: (M,) 
        - message_lhf_ratios: (M, N)
        - message_lhf_ratios_new: (M,)
    
    This function converts GAT output to match that format.
    
    Args:
        match_probs: GAT matching probabilities (M, N)
        new_anchor_probs: GAT new anchor probabilities (M,)
        detection_probability: P_d for computing undetected row
    
    Returns:
        assoc_prob_existing: (M+1, N) with undetected row added
        assoc_prob_new: (M,) same as input
        message_lhf_ratios: (M, N) placeholder (ones)
        message_lhf_ratios_new: (M,) placeholder (ones)
    """
    M, N = match_probs.shape
    
    # Create (M+1, N) matrix with undetected probability in row 0
    assoc_prob_existing = np.zeros((M + 1, N))
    assoc_prob_existing[0, :] = 1 - detection_probability  # Undetected prob
    assoc_prob_existing[1:, :] = match_probs * detection_probability
    
    # Normalize columns
    for col in range(N):
        col_sum = assoc_prob_existing[:, col].sum()
        if col_sum > 0:
            assoc_prob_existing[:, col] /= col_sum
    
    # Placeholders for message ratios (not used in GAT)
    message_lhf_ratios = np.ones((M, N))
    message_lhf_ratios_new = np.ones(M)
    
    return assoc_prob_existing, new_anchor_probs, message_lhf_ratios, message_lhf_ratios_new


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Data Preparation Module")
    print("=" * 60)
    
    # Create dummy data
    M, N = 5, 3  # 5 measurements, 3 anchors
    
    measurements = np.random.randn(2, M).astype(np.float32)
    measurements[0, :] = np.abs(measurements[0, :]) * 10  # Positive distances
    measurements[1, :] = 0.01  # Small variances
    
    anchor_predictions = {
        'z_hat': np.abs(np.random.randn(N)) * 10,
        'P': np.ones(N) * 0.02,
        'p_exist': np.random.rand(N) * 0.5 + 0.5  # [0.5, 1.0]
    }
    
    labels = np.array([0, 1, -1, 2, -1])  # 3 real, 2 clutter
    
    print(f"\nInput:")
    print(f"  measurements shape: {measurements.shape}")
    print(f"  measurements:\n{measurements}")
    print(f"  anchor z_hat: {anchor_predictions['z_hat']}")
    print(f"  labels: {labels}")
    
    # Prepare sample
    sample = prepare_training_sample(measurements, anchor_predictions, labels)
    
    print(f"\nPrepared sample:")
    print(f"  x_meas shape: {sample['x_meas'].shape}")
    print(f"  x_anchor shape: {sample['x_anchor'].shape}")
    print(f"  e_attr shape: {sample['e_attr'].shape}")
    print(f"  edge_index shape: {sample['edge_index'].shape}")
    print(f"  labels shape: {sample['labels'].shape}")
    print(f"  M={sample['M']}, N={sample['N']}")
    
    # Test with GAT model
    print("\n" + "=" * 60)
    print("Testing with EdgeConditionedBipartiteGAT")
    print("=" * 60)
    
    from bp_slam.models.edge_gat import EdgeConditionedBipartiteGAT, DataAssociationLoss
    
    model = EdgeConditionedBipartiteGAT()
    loss_fn = DataAssociationLoss()
    
    # Forward pass
    match_probs, new_anchor_probs, attn = model(
        sample['x_meas'],
        sample['x_anchor'],
        sample['e_attr'],
        sample['edge_index']
    )
    
    print(f"\nModel output:")
    print(f"  match_probs: {match_probs.shape}")
    print(f"  new_anchor_probs: {new_anchor_probs.shape}")
    
    # Compute loss
    total_loss, match_loss, new_loss = loss_fn(
        match_probs, new_anchor_probs, sample['labels'], N
    )
    
    print(f"\nLoss:")
    print(f"  Total: {total_loss.item():.4f}")
    print(f"  Matching: {match_loss.item():.4f}")
    print(f"  New anchor: {new_loss.item():.4f}")
    
    # Test Dataset
    print("\n" + "=" * 60)
    print("Testing SLAMDataset")
    print("=" * 60)
    
    # Create multiple samples
    samples = []
    for i in range(10):
        m = np.random.randint(3, 8)  # Random number of measurements
        meas = np.random.randn(2, m).astype(np.float32)
        meas[0, :] = np.abs(meas[0, :]) * 10
        meas[1, :] = 0.01
        
        lab = np.random.randint(-1, N, size=m)
        
        samples.append({
            'measurements': meas,
            'anchor_predictions': anchor_predictions,
            'labels': lab
        })
    
    loader = create_data_loader(samples, batch_size=1, shuffle=True)
    
    print(f"Created DataLoader with {len(loader)} batches")
    
    for batch in loader:
        sample = batch[0]  # batch_size=1
        print(f"  Sample: M={sample['M']}, N={sample['N']}, "
              f"x_meas={sample['x_meas'].shape}, labels={sample['labels'].tolist()}")
        break
    
    print("\n? All tests passed!")
