# -*- coding: utf-8 -*-
"""
Training script for Edge-Conditioned Bipartite GAT for SLAM Data Association.

This script trains the GAT network to replace BP-based data association
in the SLAM algorithm.

Usage:
    python train_gat.py [options]

Options:
    --epochs: Number of training epochs (default: 100)
    --lr: Learning rate (default: 1e-3)
    --hidden_dim: Hidden dimension (default: 64)
    --num_layers: Number of GAT layers (default: 2)
    --num_heads: Number of attention heads (default: 4)
    --seed: Random seed (default: 42)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from typing import Dict, List, Tuple
import scipy.io as sio

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bp_slam.models.edge_gat import EdgeConditionedBipartiteGAT, DataAssociationLoss
from bp_slam.models.data_preparation import (
    prepare_training_sample,
    generate_training_data_from_simulation,
    create_data_loader
)
from bp_slam.utils.label_generator import generate_measurements_with_labels


def load_mat_data(mat_file: str) -> Tuple[np.ndarray, List[Dict], Dict]:
    """
    Load data from MATLAB .mat file.
    
    Returns:
        trajectory: Agent trajectory (2, num_steps)
        data_va: Virtual anchor data list
        mat_data: Raw MATLAB data
    """
    print(f"Loading data from {mat_file}...")
    mat_data = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    
    # Extract trajectory
    if 'targetTrajectory' in mat_data:
        trajectory = mat_data['targetTrajectory'][:2, :]
    else:
        raise KeyError("No trajectory data found in .mat file")
    
    # Extract virtual anchor data
    data_va = []
    if 'dataVA' in mat_data:
        va_raw = mat_data['dataVA']
        if not isinstance(va_raw, np.ndarray):
            va_raw = [va_raw]
        
        for sensor_data in va_raw:
            positions = sensor_data.positions if hasattr(sensor_data, 'positions') else sensor_data['positions']
            visibility = sensor_data.visibility if hasattr(sensor_data, 'visibility') else sensor_data['visibility']
            
            data_va.append({
                'positions': np.array(positions),
                'visibility': np.array(visibility).astype(bool)
            })
    
    print(f"  Trajectory: {trajectory.shape}")
    print(f"  Sensors: {len(data_va)}")
    for i, va in enumerate(data_va):
        print(f"    Sensor {i}: {va['positions'].shape[1]} anchors")
    
    return trajectory, data_va, mat_data


def get_default_parameters() -> Dict:
    """Get default SLAM parameters for training data generation."""
    return {
        'measurementVariance': 0.01,
        'measurementVarianceLHF': 0.01,
        'detectionProbability': 0.95,
        'meanNumberOfClutter': 2.0,
        'regionOfInterestSize': 30.0,
        'numParticles': 1000,  # Smaller for training
    }


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_match_loss = 0.0
    total_new_anchor_loss = 0.0
    num_samples = 0
    
    for batch in loader:
        for sample in batch:
            # Move to device
            x_meas = sample['x_meas'].to(device)
            x_anchor = sample['x_anchor'].to(device)
            e_attr = sample['e_attr'].to(device)
            edge_index = sample['edge_index'].to(device)
            labels = sample['labels'].to(device)
            N = sample['N']
            
            # Forward pass
            match_probs, new_anchor_probs, _ = model(x_meas, x_anchor, e_attr, edge_index)
            
            # Compute loss
            loss, match_loss, new_loss = loss_fn(match_probs, new_anchor_probs, labels, N)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_match_loss += match_loss.item()
            total_new_anchor_loss += new_loss.item()
            num_samples += 1
    
    return {
        'total_loss': total_loss / num_samples,
        'match_loss': total_match_loss / num_samples,
        'new_anchor_loss': total_new_anchor_loss / num_samples
    }


def evaluate(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    
    total_loss = 0.0
    total_match_loss = 0.0
    total_new_anchor_loss = 0.0
    
    correct_matches = 0
    total_matches = 0
    correct_new_anchor = 0
    total_new_anchor = 0
    
    num_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            for sample in batch:
                x_meas = sample['x_meas'].to(device)
                x_anchor = sample['x_anchor'].to(device)
                e_attr = sample['e_attr'].to(device)
                edge_index = sample['edge_index'].to(device)
                labels = sample['labels'].to(device)
                N = sample['N']
                
                match_probs, new_anchor_probs, _ = model(x_meas, x_anchor, e_attr, edge_index)
                
                loss, match_loss, new_loss = loss_fn(match_probs, new_anchor_probs, labels, N)
                
                total_loss += loss.item()
                total_match_loss += match_loss.item()
                total_new_anchor_loss += new_loss.item()
                num_samples += 1
                
                # Compute accuracy
                # Matching accuracy (for non-clutter measurements)
                valid_mask = labels >= 0
                if valid_mask.sum() > 0:
                    pred_anchors = match_probs[valid_mask].argmax(dim=1)
                    true_anchors = labels[valid_mask]
                    correct_matches += (pred_anchors == true_anchors).sum().item()
                    total_matches += valid_mask.sum().item()
                
                # New anchor detection accuracy
                pred_new = (new_anchor_probs > 0.5).long()
                true_new = (labels == -1).long()
                correct_new_anchor += (pred_new == true_new).sum().item()
                total_new_anchor += len(labels)
    
    metrics = {
        'total_loss': total_loss / num_samples,
        'match_loss': total_match_loss / num_samples,
        'new_anchor_loss': total_new_anchor_loss / num_samples,
        'match_accuracy': correct_matches / total_matches if total_matches > 0 else 0.0,
        'new_anchor_accuracy': correct_new_anchor / total_new_anchor if total_new_anchor > 0 else 0.0
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train GAT for SLAM Data Association')
    parser.add_argument('--mat_file', type=str, default='scenarioCleanM2_new.mat',
                        help='Path to MATLAB data file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GAT layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of training samples')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    trajectory, data_va, _ = load_mat_data(args.mat_file)
    
    # Parameters
    parameters = get_default_parameters()
    
    # Generate training data
    print(f"\nGenerating {args.num_samples} training samples...")
    samples = generate_training_data_from_simulation(
        trajectory, data_va, parameters, num_samples=args.num_samples
    )
    print(f"Generated {len(samples)} samples")
    
    # Split train/val
    np.random.shuffle(samples)
    split_idx = int(len(samples) * (1 - args.val_split))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    # Create data loaders
    train_loader = create_data_loader(train_samples, batch_size=1, shuffle=True, device=device)
    val_loader = create_data_loader(val_samples, batch_size=1, shuffle=False, device=device)
    
    # Create model
    model = EdgeConditionedBipartiteGAT(
        meas_input_dim=2,
        anchor_input_dim=3,
        edge_input_dim=1,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=0.1
    ).to(device)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    loss_fn = DataAssociationLoss(matching_weight=1.0, new_anchor_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        
        # Update scheduler
        scheduler.step(val_metrics['total_loss'])
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_metrics['total_loss']:.4f} | "
                  f"Val Loss: {val_metrics['total_loss']:.4f} | "
                  f"Match Acc: {val_metrics['match_accuracy']:.2%} | "
                  f"New Acc: {val_metrics['new_anchor_accuracy']:.2%}")
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            save_path = os.path.join(args.save_dir, 'best_gat_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'args': vars(args)
            }, save_path)
    
    # Save final model
    final_path = os.path.join(args.save_dir, 'final_gat_model.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args)
    }, final_path)
    
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {args.save_dir}/")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    final_metrics = evaluate(model, val_loader, loss_fn, device)
    print(f"  Loss: {final_metrics['total_loss']:.4f}")
    print(f"  Match Accuracy: {final_metrics['match_accuracy']:.2%}")
    print(f"  New Anchor Accuracy: {final_metrics['new_anchor_accuracy']:.2%}")


if __name__ == '__main__':
    main()
