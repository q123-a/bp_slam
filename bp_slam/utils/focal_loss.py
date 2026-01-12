# -*- coding: utf-8 -*-
"""
Focal Loss and Weighted BCE for handling class imbalance.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with class imbalance.
    
    Formula: FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    
    Args:
        alpha: Weight for positive class (1)
        gamma: Focusing parameter, gamma=0 is equivalent to BCE
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits (before sigmoid), shape (N,) or (N, 1)
            targets: Binary labels 0 or 1, shape (N,) or (N, 1)
        
        Returns:
            Focal loss value
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Binary cross entropy (unreduced)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # pt = probability of correct class
        pt = torch.exp(-bce_loss)
        
        # Compute alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal weight: (1-pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Final focal loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCEForAssociation(nn.Module):
    """
    Weighted BCE for Association Head - NOT Focal Loss.
    
    Problem: True matches (1) are rare (~14%), wrong matches (0) are abundant (~86%).
    
    Previous attempts FAILED:
    - Focal Loss: Model oscillates between all-1 and all-0
    - pos_weight=6.0: all-1
    - sqrt + clamp[1,3]: all-0 (too weak)
    
    New solution: Linear ratio with higher cap
    - pos_weight = num_neg / num_pos
    - clamp to [2.0, 8.0] - allow stronger positive weight
    """
    def __init__(self, base_pos_weight=1.0, reduction='mean'):
        super().__init__()
        self.base_pos_weight = base_pos_weight
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Dynamic pos_weight based on batch statistics
        num_pos = targets.sum().clamp(min=1)
        num_neg = (1 - targets).sum().clamp(min=1)
        
        # Linear ratio, clamp to [2.0, 8.0]
        # Example: 36 neg, 6 pos -> weight = 6.0
        # Example: 80 neg, 5 pos -> weight = 8.0 (capped)
        dynamic_weight = (num_neg / num_pos).clamp(min=2.0, max=8.0)
        
        pos_weight = self.base_pos_weight * dynamic_weight
        
        loss = F.binary_cross_entropy_with_logits(
            inputs, targets, 
            pos_weight=pos_weight.unsqueeze(0),
            reduction=self.reduction
        )
        return loss


class FocalLossForAssociation(WeightedBCEForAssociation):
    """Alias for backward compatibility - now uses Dynamic Weighted BCE"""
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__(base_pos_weight=1.0, reduction=reduction)


class FocalLossForQuality(nn.Module):
    """
    Dynamic Weighted BCE for Quality Head.
    
    Problem: True signals (1) are abundant (~85%), clutter (0) is rare (~15%).
    
    Previous issue: alpha=0.25 caused model to predict ALL 1s (overcompensated)
    
    New solution: Dynamic neg_weight based on batch
    - Count samples using binary threshold (0.5)
    - neg_weight = num_signals / num_clutter
    - Apply weight only to clutter samples
    """
    def __init__(self, gamma=1.5, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Binary classification: >0.5 is signal, <0.5 is clutter
        # This works with label smoothing (0.95 -> signal, 0.05 -> clutter)
        is_signal = (targets > 0.5).float()
        is_clutter = (targets < 0.5).float()
        
        # Count samples using binary threshold
        num_signals = is_signal.sum().clamp(min=1)
        num_clutter = is_clutter.sum().clamp(min=1)
        
        # Weight for clutter class (clutter is rare)
        # Example: 5 signals, 1 clutter -> clutter_weight = 5.0
        clutter_weight = (num_signals / num_clutter).clamp(min=1.0, max=6.0)
        
        # Compute BCE loss (with smoothed targets for better gradient)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Apply class weights: signal=1.0, clutter=clutter_weight
        # Use binary mask, not soft targets
        weights = is_signal * 1.0 + is_clutter * clutter_weight
        
        # Apply focal term (mild effect)
        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        loss = weights * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
