# -*- coding: utf-8 -*-
"""
Geometry utility functions for feature initialization
Used for triangulation to provide initial values for GTSAM
"""

import numpy as np
from scipy.optimize import least_squares


def check_geometric_health(poses):
    """
    Check if geometric structure is healthy (not degenerate)
    
    Performs two checks:
    1. Baseline check: Head-to-tail distance must be >= 0.8m
    2. Collinearity check: Lateral displacement must be sufficient
    
    Args:
        poses: Agent positions array, shape (N, 2)
        
    Returns:
        bool: True if geometry is healthy for triangulation
    """
    poses = np.array(poses)
    if len(poses) < 3:
        return False
    
    # 1. Check total baseline (head-to-tail distance)
    # Step size is 0.03m, need ~27 frames to accumulate 0.8m
    # Must have sufficient baseline to resist UWB 10cm noise
    start = poses[0]
    end = poses[-1]
    baseline = np.linalg.norm(end - start)
    
    if baseline < 0.8:  # 0.8m minimum baseline
        return False

    # 2. Check collinearity via PCA
    # Compute covariance eigenvalues to check if points are on a line
    data = poses - np.mean(poses, axis=0)
    cov = np.dot(data.T, data) / len(data)
    eig_vals = np.linalg.eigvals(cov)
    
    # If minimum eigenvalue is too small, lateral displacement is insufficient
    # Threshold 2e-3 means lateral std dev of ~4.5cm, not enough to resolve left/right
    if min(np.abs(eig_vals)) < 2e-3:
        return False
        
    return True


def triangulate_feature(poses, ranges, min_observations=10, rmse_threshold=0.5):
    """
    Estimate feature position using least squares from multiple (position, range) pairs
    
    Args:
        poses: Agent positions [[x1, y1], [x2, y2], ...]
        ranges: Corresponding range measurements [r1, r2, ...]
        min_observations: Minimum number of observations required (default 10)
        rmse_threshold: RMSE threshold for valid initialization (default 0.5)
        
    Returns:
        (is_valid, estimated_pos): Whether valid, estimated position
    """
    if len(poses) < min_observations:
        return False, None

    poses = np.array(poses)
    ranges = np.array(ranges)
    
    # Pass the geometric health check first (strict gate)
    if not check_geometric_health(poses):
        return False, None

    # Residual function: predicted range - observed range
    def residuals(x):
        return np.linalg.norm(poses - x, axis=1) - ranges

    # Initial guess using weighted centroid
    x0 = _compute_initial_guess(poses, ranges)

    try:
        # Use soft_l1 loss for robustness
        res = least_squares(residuals, x0, loss='soft_l1', f_scale=0.1)
        
        # Validate convergence quality (RMSE)
        rmse = np.sqrt(np.mean(res.fun**2))
        
        if res.success and rmse < rmse_threshold:
            return True, res.x
        else:
            return False, None
    except Exception:
        return False, None


def _compute_initial_guess(poses, ranges):
    """
    Compute initial guess for feature position using weighted centroid
    
    Args:
        poses: Observation positions array
        ranges: Range measurements array
        
    Returns:
        Initial guess position
    """
    # Weights inversely proportional to range
    weights = 1.0 / (ranges + 1e-6)
    weights = weights / np.sum(weights)
    
    # Weighted centroid
    centroid = np.sum(poses * weights[:, np.newaxis], axis=0)
    
    # Average range
    avg_range = np.mean(ranges)
    
    # Offset if centroid is too close to first observation
    if np.linalg.norm(centroid - poses[0]) < avg_range * 0.5:
        direction = centroid - poses[0]
        if np.linalg.norm(direction) > 1e-6:
            direction = direction / np.linalg.norm(direction)
        else:
            direction = np.array([1.0, 0.0])
        centroid = poses[0] + direction * avg_range
    
    return centroid


def estimate_feature_from_particles(particles, weights=None):
    """
    Estimate feature position from particle set
    
    Args:
        particles: Particle positions, shape (2, num_particles)
        weights: Particle weights, shape (num_particles,)
        
    Returns:
        estimated_pos: Estimated position [x, y]
    """
    if weights is None:
        return np.mean(particles, axis=1)
    else:
        weights = weights / np.sum(weights)
        return np.sum(particles * weights, axis=1)


def compute_distance(pos1, pos2):
    """
    Compute Euclidean distance between two points
    
    Args:
        pos1: First point [x, y]
        pos2: Second point [x, y]
        
    Returns:
        Distance value
    """
    return np.linalg.norm(np.array(pos1) - np.array(pos2))
