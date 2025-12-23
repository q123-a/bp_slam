# -*- coding: utf-8 -*-
"""
Data Collection Script for GNN Training (Offline Supervised Learning - Phase 1)

Core idea:
    Run simulator with "cheat mode" (ground truth association) to drive SLAM,
    while recording GNN inputs (X) and labels (Y) at each frame.

    This ensures:
    1. SLAM keeps running without losing track
    2. All collected data is based on "correct historical state"
    3. Data quality is highest

Output:
    training_data/frame_XXXX.pt files, each containing:
    - x_meas: (M, 2) measurement features
    - x_anchor: (N, 3) anchor features
    - edge_index: (2, M*N) edge indices
    - edge_attr: (M*N, 1) edge features
    - labels: (M,) association labels
    - new_anchor_labels: (M,) new anchor labels
"""

import os
import sys
import numpy as np
import scipy.io as sio
import torch
import argparse
from pathlib import Path

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bp_slam.utils.label_generator import generate_measurements_with_labels
from bp_slam.utils.motion_model import perform_prediction
from bp_slam.utils.sampling import draw_samples_uniformly_circ, resample_systematic
from bp_slam.utils.distance import calc_distance
from bp_slam.core.anchors import (init_anchors, predict_anchors, predict_measurements,
                                   generate_new_anchors, delete_unreliable_va)


def create_gnn_training_sample(
    measurements,           # (2, M) current measurements [distance, variance]
    labels,                 # (M,) ground truth labels (truth_anchor_id or -1 for clutter)
    predicted_measurements, # (N,) predicted distances for existing anchors
    predicted_uncertainties,# (N,) predicted variances
    anchor_existence,       # (N,) anchor existence probabilities
    existing_truth_ids,     # list: truth IDs of existing anchors in SLAM state
    step,                   # int: time step
    sensor                  # int: sensor index
):
    """
    Convert SLAM intermediate data to GNN training sample.
    
    Key logic for labels:
        - Clutter (label=-1): y_new=0, y_match=all zeros (it's garbage, not new anchor!)
        - Existing anchor: y_new=0, y_match[matched_idx]=1
        - True new anchor: y_new=1, y_match=all zeros (it's in truth but not in SLAM state yet)
    
    Returns:
        sample: dict with GNN inputs and labels
        new_anchor_ids: list of truth IDs that are true new anchors (to be added to SLAM state)
    """
    if measurements is None or measurements.size == 0:
        return None, []
    
    M = measurements.shape[1]  # number of measurements
    N = len(predicted_measurements)  # number of existing anchors in SLAM state
    
    # ================================================================
    # Build measurement node features x_meas: (M, 2)
    # ================================================================
    x_meas = np.zeros((M, 2), dtype=np.float32)
    x_meas[:, 0] = measurements[0, :]  # distance
    x_meas[:, 1] = measurements[1, :]  # variance
    
    # ================================================================
    # Build anchor node features x_anchor: (N, 3)
    # If N=0 (no anchors yet), we still need to create the sample
    # ================================================================
    if N > 0:
        x_anchor = np.zeros((N, 3), dtype=np.float32)
        x_anchor[:, 0] = predicted_measurements
        x_anchor[:, 1] = predicted_uncertainties
        x_anchor[:, 2] = anchor_existence
    else:
        # No anchors yet - create dummy anchor node for graph structure
        # All measurements should be "new anchor" or "clutter"
        x_anchor = np.zeros((1, 3), dtype=np.float32)
        x_anchor[0, :] = [0.0, 1.0, 0.0]  # dummy: dist=0, var=1, exist=0
        N = 1  # for edge index calculation
    
    # ================================================================
    # Build bipartite graph edge indices edge_index: (2, M*N)
    # ================================================================
    meas_indices = np.repeat(np.arange(M), N)
    anchor_indices = np.tile(np.arange(N), M)
    edge_index = np.stack([meas_indices, anchor_indices], axis=0)
    
    # ================================================================
    # Build edge features edge_attr: (M*N, 1)
    # ================================================================
    z = measurements[0, :]
    z_hat = x_anchor[:, 0]  # Use x_anchor's predicted distance
    residuals = np.abs(z[:, np.newaxis] - z_hat[np.newaxis, :])
    edge_attr = residuals.flatten()[:, np.newaxis]
    
    # ================================================================
    # Build labels with CORRECT logic
    # ================================================================
    # Create truth_id to index mapping for existing anchors
    id_to_index = {tid: idx for idx, tid in enumerate(existing_truth_ids)}
    
    # y_match: (M, N) binary matrix for matching
    # y_new: (M,) binary vector for new anchor detection
    actual_N = len(existing_truth_ids)  # actual number of anchors (may be 0)
    y_match = np.zeros((M, max(actual_N, 1)), dtype=np.float32)
    y_new = np.zeros(M, dtype=np.float32)
    match_labels = np.zeros(M, dtype=np.int64)  # for CrossEntropy backup
    
    new_anchor_ids = []  # IDs of true new anchors discovered this frame
    
    for i, truth_id in enumerate(labels):
        if truth_id == -1:
            # ====== CLUTTER ======
            # Key fix: Clutter is NOT a new anchor! It's garbage.
            # y_match: all zeros (no match)
            # y_new: 0 (NOT a new anchor!)
            match_labels[i] = actual_N  # "no match" category
            y_new[i] = 0.0  # CRITICAL FIX: clutter is NOT new anchor
            
        elif truth_id in id_to_index:
            # ====== EXISTING ANCHOR ======
            # This measurement matches an anchor already in SLAM state
            idx = id_to_index[truth_id]
            y_match[i, idx] = 1.0
            match_labels[i] = idx
            y_new[i] = 0.0
            
        else:
            # ====== TRUE NEW ANCHOR ======
            # This measurement comes from a real anchor NOT yet in SLAM state
            # This is what we want the "new anchor head" to learn!
            match_labels[i] = actual_N  # "no match" category
            y_new[i] = 1.0  # TRUE new anchor!
            new_anchor_ids.append(truth_id)  # Remember to add this to SLAM state later
    
    # ================================================================
    # Assemble sample
    # ================================================================
    sample = {
        'x_meas': torch.from_numpy(x_meas),
        'x_anchor': torch.from_numpy(x_anchor[:actual_N] if actual_N > 0 else x_anchor),
        'edge_index': torch.from_numpy(edge_index.astype(np.int64)),
        'edge_attr': torch.from_numpy(edge_attr.astype(np.float32)),
        # Labels
        'y_match': torch.from_numpy(y_match[:, :actual_N] if actual_N > 0 else y_match),  # (M, N) for Focal Loss
        'y_new': torch.from_numpy(y_new),  # (M,) for BCE Loss
        'match_labels': torch.from_numpy(match_labels),  # (M,) for CrossEntropy backup
        # Metadata
        'metadata': {
            'step': step,
            'sensor': sensor,
            'num_measurements': M,
            'num_anchors': actual_N,
            'existing_truth_ids': existing_truth_ids.copy(),
            'new_anchor_ids': new_anchor_ids.copy()
        }
    }
    
    return sample, new_anchor_ids


def collect_training_data_with_ground_truth(
    data_va,
    cluttered_measurements,
    labels_all,
    parameters,
    true_trajectory,
    output_dir='training_data'
):
    """
    Use ground truth association to drive SLAM while collecting GNN training data.
    
    KEY IMPROVEMENTS:
        1. Start with EMPTY anchor list (not all anchors)
        2. Dynamically discover and add new anchors
        3. Correctly label: clutter=0, existing=match, new_anchor=1
    
    This ensures the GNN learns:
        - To reject clutter (y_new=0 when no match and it's garbage)
        - To match existing anchors (y_match has correct entry)
        - To discover new anchors (y_new=1 when it's a real new anchor)
    
    Args:
        data_va: virtual anchor data (contains true anchor positions)
        cluttered_measurements: measurements with clutter
        labels_all: ground truth labels [step][sensor] = (M,) truth_anchor_ids (-1 for clutter)
        parameters: parameter dictionary
        true_trajectory: true trajectory
        output_dir: output directory
    
    Returns:
        num_samples: number of collected samples
        all_samples: list of all samples
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get dimensions
    num_steps = min(len(cluttered_measurements), parameters['maxSteps'])
    num_sensors = len(cluttered_measurements[0])
    num_particles = parameters['numParticles']
    known_track = parameters['known_track']
    prior_covariance = parameters['priorCovarianceAnchor']
    
    # Initialize agent particles
    if known_track:
        posterior_particles_agent = np.tile(
            np.vstack([true_trajectory[:2, 0:1], np.zeros((2, 1))]),
            (1, num_particles)
        )
    else:
        posterior_particles_agent = np.zeros((4, num_particles))
        posterior_particles_agent[0:2, :] = draw_samples_uniformly_circ(
            parameters['priorMean'][0:2], parameters['UniformRadius_pos'], num_particles
        )
        posterior_particles_agent[2:4, :] = (
            np.tile(parameters['priorMean'][2:4].reshape(-1, 1), (1, num_particles)) +
            2 * parameters['UniformRadius_vel'] * np.random.rand(2, num_particles) -
            parameters['UniformRadius_vel']
        )
    
    # ================================================================
    # KEY CHANGE: Start with EMPTY anchor lists for each sensor
    # We will dynamically add anchors as we "discover" them
    # ================================================================
    # Each sensor has its own list of discovered anchors
    # Each anchor entry: {truth_id, particles, existence_prob}
    discovered_anchors = [[] for _ in range(num_sensors)]
    
    # Store true anchor positions for initializing new anchors
    true_anchor_positions = [data_va[s]['positions'] for s in range(num_sensors)]
    
    # Collect all samples
    all_samples = []
    sample_count = 0
    
    # Statistics
    stats = {'clutter': 0, 'existing': 0, 'new_anchor': 0}
    
    # Main loop
    for step in range(1, num_steps):
        # Predict agent state
        if known_track:
            predicted_particles_agent = np.tile(
                np.vstack([true_trajectory[:2, step:step+1], np.zeros((2, 1))]),
                (1, num_particles)
            )
        else:
            predicted_particles_agent = perform_prediction(posterior_particles_agent, parameters)
        
        # Process each sensor
        for sensor in range(num_sensors):
            measurements = cluttered_measurements[step][sensor]
            labels = labels_all[step][sensor]
            
            if measurements is None or measurements.size == 0:
                continue
            
            # ================================================================
            # Compute predicted measurements for existing discovered anchors
            # ================================================================
            num_anchors = len(discovered_anchors[sensor])
            
            if num_anchors > 0:
                # Stack all anchor particles: (2, num_particles, num_anchors)
                anchor_particles_stack = np.stack([
                    a['particles'] for a in discovered_anchors[sensor]
                ], axis=2)
                
                # Compute distances from agent to each anchor
                agent_pos = predicted_particles_agent[:2, :]  # (2, num_particles)
                
                # predicted_measurements[j] = mean distance to anchor j
                predicted_measurements = np.zeros(num_anchors)
                predicted_uncertainties = np.zeros(num_anchors)
                
                for j in range(num_anchors):
                    anchor_pos = anchor_particles_stack[:, :, j]  # (2, num_particles)
                    distances = np.sqrt(np.sum((agent_pos - anchor_pos)**2, axis=0))
                    predicted_measurements[j] = np.mean(distances)
                    predicted_uncertainties[j] = np.var(distances) + parameters['measurementVariance']
                
                anchor_existence = np.array([a['existence_prob'] for a in discovered_anchors[sensor]])
                existing_truth_ids = [a['truth_id'] for a in discovered_anchors[sensor]]
            else:
                # No anchors yet
                predicted_measurements = np.array([])
                predicted_uncertainties = np.array([])
                anchor_existence = np.array([])
                existing_truth_ids = []
            
            # ================================================================
            # Create GNN training sample
            # ================================================================
            sample, new_anchor_ids = create_gnn_training_sample(
                measurements=measurements,
                labels=labels,
                predicted_measurements=predicted_measurements,
                predicted_uncertainties=predicted_uncertainties,
                anchor_existence=anchor_existence,
                existing_truth_ids=existing_truth_ids,
                step=step,
                sensor=sensor
            )
            
            if sample is not None:
                all_samples.append(sample)
                sample_count += 1
                
                # Update statistics
                y_new = sample['y_new'].numpy()
                y_match = sample['y_match'].numpy() if sample['y_match'].numel() > 0 else np.zeros((len(y_new), 1))
                for i in range(len(y_new)):
                    if y_new[i] == 1.0:
                        stats['new_anchor'] += 1
                    elif y_match.shape[1] > 0 and np.sum(y_match[i, :]) > 0:
                        stats['existing'] += 1
                    else:
                        stats['clutter'] += 1
            
            # ================================================================
            # CHEAT MODE: Add newly discovered anchors to SLAM state
            # ================================================================
            for new_id in new_anchor_ids:
                if new_id not in [a['truth_id'] for a in discovered_anchors[sensor]]:
                    # Get true position of this anchor
                    true_pos = true_anchor_positions[sensor][:, new_id]
                    
                    # Initialize particles around true position
                    new_particles = np.random.multivariate_normal(
                        true_pos, prior_covariance, num_particles
                    ).T  # (2, num_particles)
                    
                    # Add to discovered anchors
                    discovered_anchors[sensor].append({
                        'truth_id': new_id,
                        'particles': new_particles,
                        'existence_prob': 0.9  # High initial probability
                    })
            
            # Update existence probability for observed existing anchors
            id_to_idx = {a['truth_id']: idx for idx, a in enumerate(discovered_anchors[sensor])}
            for truth_id in labels:
                if truth_id >= 0 and truth_id in id_to_idx:
                    idx = id_to_idx[truth_id]
                    # Increase existence probability when observed
                    discovered_anchors[sensor][idx]['existence_prob'] = min(
                        0.999, 
                        discovered_anchors[sensor][idx]['existence_prob'] * 1.1
                    )
        
        # Update agent particles
        posterior_particles_agent = predicted_particles_agent
        
        # Print progress
        if step % 50 == 0 or step == num_steps - 1:
            total_anchors = sum(len(discovered_anchors[s]) for s in range(num_sensors))
            print(f'Step {step}/{num_steps-1}, Samples: {sample_count}, Discovered anchors: {total_anchors}')
    
    # Save all samples
    output_file = output_path / 'training_data.pt'
    torch.save({
        'samples': all_samples,
        'num_samples': sample_count,
        'num_steps': num_steps,
        'num_sensors': num_sensors,
        'statistics': stats,
        'parameters': {
            'max_steps': parameters['maxSteps'],
            'num_particles': parameters['numParticles'],
            'detection_probability': parameters['detectionProbability'],
            'measurement_variance': parameters['measurementVariance'],
        }
    }, output_file)
    
    print(f'\nData collection complete!')
    print(f'Total samples: {sample_count}')
    print(f'Saved to: {output_file}')
    print(f'\n=== Label Distribution ===')
    total = stats['clutter'] + stats['existing'] + stats['new_anchor']
    print(f"Clutter (y_new=0, no match): {stats['clutter']} ({100*stats['clutter']/total:.1f}%)")
    print(f"Existing match: {stats['existing']} ({100*stats['existing']/total:.1f}%)")
    print(f"True new anchor (y_new=1): {stats['new_anchor']} ({100*stats['new_anchor']/total:.1f}%)")
    
    return sample_count, all_samples


def main():
    parser = argparse.ArgumentParser(description='Collect training data for GNN')
    parser.add_argument('--mat_file', type=str, default='scenarioCleanM2_new.mat',
                        help='MATLAB data file')
    parser.add_argument('--output_dir', type=str, default='training_data',
                        help='Output directory for training samples')
    parser.add_argument('--max_steps', type=int, default=900,
                        help='Maximum number of time steps')
    parser.add_argument('--num_particles', type=int, default=10000,
                        help='Number of particles (can be smaller for data collection)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load data (same format as testbed.py)
    print(f'Loading data from {args.mat_file}...')
    mat_data = sio.loadmat(args.mat_file)
    
    # Extract trajectory
    true_trajectory = mat_data['trueTrajectory']
    print(f'True trajectory shape: {true_trajectory.shape}')
    
    # Extract virtual anchor data (same format as testbed.py)
    data_va_raw = mat_data['dataVA'][:, 0]
    num_sensors = len(data_va_raw)
    
    # Convert to Python format
    data_va = []
    for sensor in range(num_sensors):
        sensor_data = {
            'positions': data_va_raw[sensor]['positions'][0, 0],
            'visibility': np.ones((data_va_raw[sensor]['positions'][0, 0].shape[1],
                                  true_trajectory.shape[1])).astype(bool)
        }
        data_va.append(sensor_data)
    
    print(f'Number of sensors: {num_sensors}')
    for i, va in enumerate(data_va):
        print(f'  Sensor {i}: {va["positions"].shape[1]} anchors')
    
    # Set parameters (same as testbed.py)
    num_steps = true_trajectory.shape[1]
    
    parameters = {}
    parameters['maxSteps'] = min(args.max_steps, num_steps)
    parameters['lengthStep'] = 0.03
    parameters['scanTime'] = 1
    
    v_max = parameters['lengthStep'] / parameters['scanTime']
    parameters['drivingNoiseVariance'] = (v_max / 3 / parameters['scanTime'])**2
    
    parameters['measurementVariance'] = 0.1**2
    parameters['measurementVarianceLHF'] = 0.15**2
    parameters['detectionProbability'] = 0.95
    parameters['regionOfInterestSize'] = 30
    parameters['meanNumberOfClutter'] = 1
    parameters['clutterIntensity'] = parameters['meanNumberOfClutter'] / parameters['regionOfInterestSize']
    
    parameters['meanNumberOfBirth'] = 1e-4
    parameters['birthIntensity'] = parameters['meanNumberOfBirth'] / (2 * parameters['regionOfInterestSize'])**2
    
    parameters['meanNumberOfUndetectedAnchors'] = 6
    parameters['undetectedAnchorsIntensity'] = parameters['meanNumberOfUndetectedAnchors'] / (2 * parameters['regionOfInterestSize'])**2
    
    parameters['numParticles'] = args.num_particles
    parameters['survivalProbability'] = 0.999
    parameters['unreliabilityThreshold'] = 1e-4
    
    # KEY CHANGE: Start with EMPTY anchor list for dynamic discovery
    # The data collection will dynamically add anchors as they are discovered
    # This enables learning "new anchor detection" capability
    parameters['priorKnownAnchors'] = [[], []]  # Empty for both sensors
    print(f'Starting with EMPTY anchor lists (dynamic discovery mode)')
    print(f'  True anchors available: Sensor0={data_va[0]["positions"].shape[1]}, Sensor1={data_va[1]["positions"].shape[1]}')
    
    parameters['priorCovarianceAnchor'] = 0.001**2 * np.eye(2)
    parameters['anchorRegularNoiseVariance'] = 1e-4**2
    parameters['UniformRadius_pos'] = 0.5
    parameters['UniformRadius_vel'] = 0.05
    parameters['known_track'] = True  # Use known track mode for data collection
    parameters['priorMean'] = np.vstack([true_trajectory[0:2, 0:1], np.zeros((2, 1))])
    
    # Truncate trajectory to max_steps
    true_trajectory = true_trajectory[:, :parameters['maxSteps']]
    
    # Generate measurements with labels
    print('Generating measurements with ground truth labels...')
    cluttered_measurements, labels_all = generate_measurements_with_labels(
        true_trajectory, data_va, parameters
    )
    
    # Collect training data
    print('Collecting training data with ground truth association...')
    num_samples, all_samples = collect_training_data_with_ground_truth(
        data_va=data_va,
        cluttered_measurements=cluttered_measurements,
        labels_all=labels_all,
        parameters=parameters,
        true_trajectory=true_trajectory,
        output_dir=args.output_dir
    )
    
    print(f'\n=== Data Collection Statistics ===')
    print(f'Time steps: {parameters["maxSteps"]}')
    print(f'Sensors: {num_sensors}')
    print(f'Total samples: {num_samples}')
    print(f'Samples per step: {num_samples / parameters["maxSteps"]:.1f}')
    
    # Print sample structure info
    if len(all_samples) > 0:
        sample = all_samples[0]
        print(f'\n=== Sample Structure ===')
        print(f'x_meas: {sample["x_meas"].shape}')
        print(f'x_anchor: {sample["x_anchor"].shape}')
        print(f'edge_index: {sample["edge_index"].shape}')
        print(f'edge_attr: {sample["edge_attr"].shape}')
        print(f'match_labels: {sample["match_labels"].shape}')


if __name__ == '__main__':
    main()
