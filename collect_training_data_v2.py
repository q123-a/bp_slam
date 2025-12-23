# -*- coding: utf-8 -*-
"""
Data Collection Script for GNN Training (Simplified Version)

This version directly uses bp_slam.core.slam.bp_based_mint_slam() with
association_mode='ground_truth' to collect training data.

Core idea:
    Run SLAM with "ground truth association" mode to drive state updates,
    while collecting GNN training samples at each frame.

Output:
    training_data/training_data.pt - PyTorch file containing all samples
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
from bp_slam.core.slam import bp_based_mint_slam


def main():
    parser = argparse.ArgumentParser(description='Collect GNN training data using ground truth association')
    parser.add_argument('--mat_file', type=str, default='scenarioCleanM2_new.mat',
                        help='Path to MATLAB data file')
    parser.add_argument('--output_dir', type=str, default='training_data',
                        help='Output directory for training data')
    parser.add_argument('--max_steps', type=int, default=900,
                        help='Maximum number of time steps')
    parser.add_argument('--num_particles', type=int, default=100000,
                        help='Number of particles')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print('=' * 60)
    print('GNN Training Data Collection (Simplified Version)')
    print('Using bp_based_mint_slam with ground_truth association mode')
    print('=' * 60)
    
    # Load MATLAB data (same format as testbed.py and collect_training_data.py)
    print(f'Loading data from: {args.mat_file}')
    mat_data = sio.loadmat(args.mat_file)
    
    # Extract true trajectory
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
    
    num_sensors = len(data_va)
    print(f'Number of sensors: {num_sensors}')
    for i, va in enumerate(data_va):
        print(f'  Sensor {i}: {va["positions"].shape[1]} anchors')
    
    # Set parameters
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
    parameters['upSamplingFactor'] = 1  # Particle upsampling factor
    parameters['detectionThreshold'] = 0.5
    parameters['survivalProbability'] = 0.999
    parameters['unreliabilityThreshold'] = 1e-4
    
    # KEY: Empty anchor list for dynamic discovery
    parameters['priorKnownAnchors'] = [[], []]
    print(f'Starting with EMPTY anchor lists (dynamic discovery mode)')
    
    parameters['priorCovarianceAnchor'] = 0.001**2 * np.eye(2)
    parameters['anchorRegularNoiseVariance'] = 1e-4**2
    parameters['UniformRadius_pos'] = 0.5
    parameters['UniformRadius_vel'] = 0.05
    parameters['known_track'] = True  # Use known track for data collection
    parameters['priorMean'] = np.vstack([true_trajectory[0:2, 0:1], np.zeros((2, 1))])
    
    # Truncate trajectory
    true_trajectory_trunc = true_trajectory[:, :parameters['maxSteps']]
    
    # Generate measurements with ground truth labels
    print('Generating measurements with ground truth labels...')
    cluttered_measurements, labels_all = generate_measurements_with_labels(
        true_trajectory_trunc, data_va, parameters
    )
    
    # Run SLAM with ground truth association and collect training data
    print('Running SLAM with ground_truth association mode...')
    print(f'Collecting training data...')
    
    results = bp_based_mint_slam(
        data_va=data_va,
        cluttered_measurements=cluttered_measurements,
        parameters=parameters,
        true_trajectory=true_trajectory_trunc,
        association_mode='ground_truth',
        ground_truth_labels=labels_all,
        collect_training_data=True
    )
    
    # Unpack results
    estimated_trajectory, estimated_anchors, posterior_particles_anchors_storage, num_estimated_anchors, training_samples = results
    
    # Save training data
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / 'training_data.pt'
    
    torch.save(training_samples, output_file)
    print(f'\nSaved {len(training_samples)} training samples to {output_file}')
    
    # Print statistics
    print(f'\n=== Data Collection Statistics ===')
    print(f'Time steps: {parameters["maxSteps"]}')
    print(f'Sensors: {num_sensors}')
    print(f'Total samples: {len(training_samples)}')
    print(f'Samples per step: {len(training_samples) / parameters["maxSteps"]:.1f}')
    
    # Analyze label distribution
    total_clutter = 0
    total_existing = 0
    total_new = 0
    
    for sample in training_samples:
        y_new = sample['y_new'].numpy()
        y_match = sample['y_match'].numpy()
        
        for i in range(len(y_new)):
            if y_new[i] == 1.0:
                total_new += 1
            elif y_match[i].sum() > 0:
                total_existing += 1
            else:
                total_clutter += 1
    
    total = total_clutter + total_existing + total_new
    if total > 0:
        print(f'\n=== Label Distribution ===')
        print(f'Clutter: {total_clutter} ({100*total_clutter/total:.1f}%)')
        print(f'Existing anchor: {total_existing} ({100*total_existing/total:.1f}%)')
        print(f'New anchor: {total_new} ({100*total_new/total:.1f}%)')
    
    # Print sample structure
    if len(training_samples) > 0:
        sample = training_samples[0]
        print(f'\n=== Sample Structure ===')
        print(f'x_meas: {sample["x_meas"].shape}')
        print(f'x_anchor: {sample["x_anchor"].shape}')
        print(f'edge_index: {sample["edge_index"].shape}')
        print(f'edge_attr: {sample["edge_attr"].shape}')
        print(f'y_match: {sample["y_match"].shape}')
        print(f'y_new: {sample["y_new"].shape}')


if __name__ == '__main__':
    main()
