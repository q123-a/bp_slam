# -*- coding: utf-8 -*-
"""
Training script for Edge-Conditioned Bipartite GAT (Version 2)
Uses pre-collected training data from collect_training_data_v2.py

Usage:
    python train_gat_v2.py --data_file training_data/training_data.pt --epochs 100
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bp_slam.models.edge_gat import EdgeConditionedBipartiteGAT


class SLAMDataset(Dataset):
    """Dataset for SLAM data association training samples."""
    
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        """
        pred: (M, N) match probabilities
        target: (M, N) binary labels
        """
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1 - eps)
        
        # Binary cross entropy
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        
        # Focal weight
        pt = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        
        loss = alpha_weight * focal_weight * bce
        return loss.mean()


class DataAssociationLoss(nn.Module):
    """Combined loss for data association."""
    
    def __init__(self, match_weight=1.0, new_anchor_weight=1.0):
        super().__init__()
        self.match_weight = match_weight
        self.new_anchor_weight = new_anchor_weight
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.bce_loss = nn.BCELoss()
    
    def forward(self, match_probs, new_anchor_probs, y_match, y_new):
        """
        match_probs: (M, N) predicted match probabilities
        new_anchor_probs: (M,) predicted new anchor probabilities
        y_match: (M, N) ground truth match labels
        y_new: (M,) ground truth new anchor labels
        """
        # Match loss using Focal Loss
        if y_match.numel() > 0 and match_probs.shape == y_match.shape:
            match_loss = self.focal_loss(match_probs, y_match)
        else:
            match_loss = torch.tensor(0.0, device=match_probs.device)
        
        # New anchor loss using BCE
        if y_new.numel() > 0:
            new_anchor_loss = self.bce_loss(new_anchor_probs, y_new)
        else:
            new_anchor_loss = torch.tensor(0.0, device=new_anchor_probs.device)
        
        total_loss = self.match_weight * match_loss + self.new_anchor_weight * new_anchor_loss
        
        return total_loss, match_loss, new_anchor_loss


def collate_fn(batch):
    """Custom collate function - just return list of samples."""
    return batch


def train_epoch(model, loader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_match_loss = 0.0
    total_new_loss = 0.0
    num_samples = 0
    
    for batch in loader:
        for sample in batch:
            # Get data
            x_meas = sample['x_meas'].float().to(device)
            x_anchor = sample['x_anchor'].float().to(device)
            edge_index = sample['edge_index'].long().to(device)
            edge_attr = sample['edge_attr'].float().to(device)
            y_match = sample['y_match'].float().to(device)
            y_new = sample['y_new'].float().to(device)
            
            M = x_meas.shape[0]
            N = x_anchor.shape[0]
            
            # Skip if no measurements or no anchors
            if M == 0 or N == 0:
                continue
            
            # Forward pass (use sigmoid for independent edge probabilities)
            match_probs, new_anchor_probs, _ = model(x_meas, x_anchor, edge_attr, edge_index, use_sigmoid=True)
            
            # Ensure shapes match
            if match_probs.shape != y_match.shape:
                # Resize y_match if needed
                if match_probs.shape[1] != y_match.shape[1]:
                    continue  # Skip if shape mismatch
            
            # Compute loss
            loss, match_loss, new_loss = loss_fn(match_probs, new_anchor_probs, y_match, y_new)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_match_loss += match_loss.item()
            total_new_loss += new_loss.item()
            num_samples += 1
    
    if num_samples == 0:
        return {'loss': 0, 'match_loss': 0, 'new_loss': 0}
    
    return {
        'loss': total_loss / num_samples,
        'match_loss': total_match_loss / num_samples,
        'new_loss': total_new_loss / num_samples
    }


def evaluate(model, loader, loss_fn, device):
    """Evaluate model."""
    model.eval()
    
    total_loss = 0.0
    total_match_loss = 0.0
    total_new_loss = 0.0
    
    # Accuracy metrics
    correct_new = 0
    total_new = 0
    correct_match = 0
    total_match = 0
    
    num_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            for sample in batch:
                x_meas = sample['x_meas'].float().to(device)
                x_anchor = sample['x_anchor'].float().to(device)
                edge_index = sample['edge_index'].long().to(device)
                edge_attr = sample['edge_attr'].float().to(device)
                y_match = sample['y_match'].float().to(device)
                y_new = sample['y_new'].float().to(device)
                
                M = x_meas.shape[0]
                N = x_anchor.shape[0]
                
                if M == 0 or N == 0:
                    continue
                
                match_probs, new_anchor_probs, _ = model(x_meas, x_anchor, edge_attr, edge_index, use_sigmoid=True)
                
                if match_probs.shape != y_match.shape:
                    continue
                
                loss, match_loss, new_loss = loss_fn(match_probs, new_anchor_probs, y_match, y_new)
                
                total_loss += loss.item()
                total_match_loss += match_loss.item()
                total_new_loss += new_loss.item()
                num_samples += 1
                
                # New anchor accuracy
                pred_new = (new_anchor_probs > 0.5).float()
                correct_new += (pred_new == y_new).sum().item()
                total_new += y_new.numel()
                
                # Match accuracy (for measurements that have a match)
                has_match = y_match.sum(dim=1) > 0
                if has_match.sum() > 0:
                    pred_match = match_probs[has_match].argmax(dim=1)
                    true_match = y_match[has_match].argmax(dim=1)
                    correct_match += (pred_match == true_match).sum().item()
                    total_match += has_match.sum().item()
    
    if num_samples == 0:
        return {'loss': 0, 'match_loss': 0, 'new_loss': 0, 'new_acc': 0, 'match_acc': 0}
    
    return {
        'loss': total_loss / num_samples,
        'match_loss': total_match_loss / num_samples,
        'new_loss': total_new_loss / num_samples,
        'new_acc': correct_new / total_new if total_new > 0 else 0,
        'match_acc': correct_match / total_match if total_match > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description='Train GAT for SLAM Data Association')
    parser.add_argument('--data_file', type=str, default='training_data/training_data.pt',
                        help='Path to training data file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GAT layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Load data
    print(f'Loading data from {args.data_file}...')
    samples = torch.load(args.data_file, weights_only=False)
    print(f'Loaded {len(samples)} samples')
    
    # Analyze data
    total_clutter = 0
    total_existing = 0
    total_new = 0
    for s in samples:
        y_new = s['y_new'].numpy()
        y_match = s['y_match'].numpy()
        for i in range(len(y_new)):
            if y_new[i] == 1.0:
                total_new += 1
            elif y_match[i].sum() > 0:
                total_existing += 1
            else:
                total_clutter += 1
    
    total = total_clutter + total_existing + total_new
    print(f'\nLabel distribution:')
    print(f'  Clutter: {total_clutter} ({100*total_clutter/total:.1f}%)')
    print(f'  Existing: {total_existing} ({100*total_existing/total:.1f}%)')
    print(f'  New anchor: {total_new} ({100*total_new/total:.1f}%)')
    
    # Create dataset
    dataset = SLAMDataset(samples)
    
    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f'\nTrain samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn)
    
    # Create model
    model = EdgeConditionedBipartiteGAT(
        meas_input_dim=2,      # [distance, variance]
        anchor_input_dim=3,    # [predicted_dist, variance, existence]
        edge_input_dim=1,      # |z - z_hat|
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel parameters: {num_params:,}')
    
    # Loss and optimizer
    loss_fn = DataAssociationLoss(match_weight=1.0, new_anchor_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=10)
    
    # Training loop
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    best_val_loss = float('inf')
    
    # History for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'match_acc': [],
        'new_acc': []
    }
    
    print(f'\nStarting training for {args.epochs} epochs...\n')
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['match_acc'].append(val_metrics['match_acc'])
        history['new_acc'].append(val_metrics['new_acc'])
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Match Acc: {val_metrics['match_acc']:.3f} | "
              f"New Acc: {val_metrics['new_acc']:.3f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'args': vars(args)
            }, save_dir / 'best_model.pt')
            print(f"  -> Saved best model (val_loss: {best_val_loss:.4f})")
    
    print(f'\nTraining complete!')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Model saved to: {save_dir / "best_model.pt"}')
    
    # Plot training curves
    plot_training_curves(history, save_dir)


def plot_training_curves(history, save_dir):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss curves
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    ax2 = axes[1]
    ax2.plot(epochs, history['match_acc'], 'g-', label='Match Accuracy', linewidth=2)
    ax2.plot(epochs, history['new_acc'], 'm-', label='New Anchor Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Matching and New Anchor Detection Accuracy', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    # Save figure
    fig_path = save_dir / 'training_curves.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'Training curves saved to: {fig_path}')
    
    # Show figure
    plt.show()


if __name__ == '__main__':
    main()
