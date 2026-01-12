"""
bp_slam/core/reid_utils.py
ReID Fingerprint Utility Functions

Compute and manage anchor ReID fingerprints.
Based on Friis formula: F = RSS + 20*log10(d)
"""

import numpy as np


# ============ Normalization Parameters ============
FINGERPRINT_CENTER = 0.0    # Fingerprint center value (dB)
FINGERPRINT_SCALE = 10.0    # Fingerprint scale factor (dB)
DELTA_F_SCALE = 5.0         # Fingerprint difference scale factor (dB)
MIN_DISTANCE = 0.5          # Minimum distance threshold (m), prevent log(0)

# EMA Update Parameters
EMA_ALPHA = 0.05            # Update rate: 95% trust history, 5% accept new
MIN_EXIST_PROB = 0.9        # Minimum existence probability for update
MIN_ASSOC_PROB = 0.8        # Minimum association probability for update


def compute_fingerprint(rss_dbm, distance):
    """
    Compute ReID fingerprint.
    
    Based on Friis formula: F = RSS + 20*log10(d)
    Physical meaning: Compensate distance attenuation, restore anchor reflection characteristics.
    
    Args:
        rss_dbm: Signal strength (dBm)
        distance: Measured distance (m)
    
    Returns:
        F: Fingerprint value (dB)
    """
    d_safe = max(distance, MIN_DISTANCE)
    F = rss_dbm + 20.0 * np.log10(d_safe)
    return F


def normalize_fingerprint(F):
    """
    Normalize fingerprint value.
    
    Args:
        F: Raw fingerprint value (dB)
    
    Returns:
        F_norm: Normalized fingerprint, expected range [-2, 2]
    """
    if F is None:
        return 0.0  # Preset anchor without history returns neutral value
    return (F - FINGERPRINT_CENTER) / FINGERPRINT_SCALE


def compute_delta_f_norm(F_instant, F_history):
    """
    Compute normalized fingerprint difference (edge feature).
    
    Args:
        F_instant: Measurement instant fingerprint (dB)
        F_history: Anchor history fingerprint (dB), may be None
    
    Returns:
        delta_F_norm: Normalized fingerprint difference, range [0, 3]
    """
    if F_history is None:
        # Preset anchor without history, return neutral value
        # Let GNN rely on geometric features only
        return 1.0  # Neutral value, means "unknown match"
    
    delta_F = abs(F_instant - F_history)
    delta_F_norm = delta_F / DELTA_F_SCALE
    # Clamp to [0, 3]
    delta_F_norm = min(delta_F_norm, 3.0)
    return delta_F_norm


def update_fingerprint_ema(F_old, F_instant, alpha=EMA_ALPHA):
    """
    Update fingerprint using EMA.
    
    Formula: F_new = alpha * F_instant + (1 - alpha) * F_old
    
    Args:
        F_old: Old history fingerprint, may be None
        F_instant: Current instant fingerprint
        alpha: Update rate, default 0.05
    
    Returns:
        F_new: Updated fingerprint
    """
    if F_old is None:
        # First initialization
        return F_instant
    
    F_new = alpha * F_instant + (1 - alpha) * F_old
    return F_new


def compute_fingerprints_batch(distances, rss_values):
    """
    Batch compute fingerprints.
    
    Args:
        distances: (M,) Distance array
        rss_values: (M,) RSS array (dBm)
    
    Returns:
        fingerprints: (M,) Fingerprint array
    """
    M = len(distances)
    fingerprints = np.zeros(M)
    
    for m in range(M):
        fingerprints[m] = compute_fingerprint(rss_values[m], distances[m])
    
    return fingerprints


def get_anchor_fingerprints(posterior_particles_anchors):
    """
    Extract all fingerprints from anchor states.
    
    Args:
        posterior_particles_anchors: Anchor particle list
    
    Returns:
        fingerprints: (K,) Fingerprint list, None means no history
    """
    K = len(posterior_particles_anchors)
    fingerprints = []
    
    for k in range(K):
        anchor = posterior_particles_anchors[k]
        F = anchor.get('reid_fingerprint', None)
        fingerprints.append(F)
    
    return fingerprints
