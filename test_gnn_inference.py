# -*- coding: utf-8 -*-
"""
Test GNN inference mode in SLAM.

This script runs SLAM with the trained GNN model for data association
and compares with BP mode.
"""

import os
import sys
import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bp_slam.core.slam import bp_based_mint_slam
from bp_slam.utils.measurements import generate_measurements, generate_cluttered_measurements
from bp_slam.models.edge_gat import EdgeConditionedBipartiteGAT


def load_gnn_model(checkpoint_path, device='cuda'):
    """Load trained GNN model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model args
    args = checkpoint.get('args', {})
    hidden_dim = args.get('hidden_dim', 64)
    num_layers = args.get('num_layers', 2)
    num_heads = args.get('num_heads', 4)
    
    # Create model
    model = EdgeConditionedBipartiteGAT(
        meas_input_dim=2,
        anchor_input_dim=3,
        edge_input_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='scen_semroom_new.mat',
                        choices=['scenarioCleanM2_new.mat', 'scen_semroom_new.mat'],
                        help='Dataset to use for testing (default: scen_semroom_new.mat for unseen data)')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps to run')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load GNN model
    model_path = 'checkpoints/best_model.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first: python train_gat_v2.py")
        return
    
    gnn_model = load_gnn_model(model_path, device)
    
    # Load data
    print(f'\nLoading data from {args.dataset}...')
    print('NOTE: Training was done on scenarioCleanM2_new.mat')
    if args.dataset == 'scen_semroom_new.mat':
        print('      Testing on UNSEEN dataset (scen_semroom_new.mat) - proper generalization test!')
    else:
        print('      Testing on SAME dataset as training - may overestimate performance!')
    
    mat_file = args.dataset
    mat_data = sio.loadmat(mat_file)
    true_trajectory = mat_data['trueTrajectory']
    data_va_raw = mat_data['dataVA'][:, 0]
    
    data_va = []
    for sensor in range(len(data_va_raw)):
        sensor_data = {
            'positions': data_va_raw[sensor]['positions'][0, 0],
            'visibility': np.ones((data_va_raw[sensor]['positions'][0, 0].shape[1],
                                  true_trajectory.shape[1])).astype(bool)
        }
        data_va.append(sensor_data)
    
    # Parameters (same as testbed.py)
    parameters = {}
    parameters['maxSteps'] = 100  # Quick test
    true_trajectory = true_trajectory[:, :parameters['maxSteps']]
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
    parameters['numParticles'] = 10000  # Reduced for testing
    parameters['upSamplingFactor'] = 1
    parameters['detectionThreshold'] = 0.5
    parameters['survivalProbability'] = 0.999
    parameters['unreliabilityThreshold'] = 1e-4
    parameters['priorKnownAnchors'] = [[0], [0]]
    parameters['priorCovarianceAnchor'] = 0.001**2 * np.eye(2)
    parameters['anchorRegularNoiseVariance'] = 1e-4**2
    parameters['UniformRadius_pos'] = 0.5
    parameters['UniformRadius_vel'] = 0.05
    parameters['known_track'] = False  # Unknown track mode
    parameters['priorMean'] = np.vstack([true_trajectory[0:2, 0:1], np.zeros((2, 1))])
    
    np.random.seed(1)
    
    # Generate measurements
    print('Generating measurements...')
    measurements = generate_measurements(true_trajectory, data_va, parameters)
    cluttered_measurements = generate_cluttered_measurements(measurements, parameters)
    
    # Run SLAM with GNN mode
    print('\n' + '=' * 60)
    print('Running SLAM with GNN data association...')
    print('=' * 60)
    
    (estimated_trajectory_gnn, estimated_anchors_gnn,
     _, num_estimated_anchors_gnn) = bp_based_mint_slam(
        data_va, cluttered_measurements, parameters, true_trajectory,
        association_mode='gnn',
        gnn_model=gnn_model
    )
    
    # Calculate final error
    from bp_slam.utils.distance import calc_distance
    errors_gnn = calc_distance(true_trajectory[0:2, :], estimated_trajectory_gnn[0:2, :])
    mean_error_gnn = np.mean(errors_gnn)
    
    print(f'\n=== GNN Mode Results ===')
    print(f'Mean position error: {mean_error_gnn:.4f} m')
    print(f'Final anchors: Sensor1={num_estimated_anchors_gnn[0, -1]}, Sensor2={num_estimated_anchors_gnn[1, -1]}')
    
    # Optionally compare with BP mode
    print('\n' + '=' * 60)
    print('Running SLAM with BP data association for comparison...')
    print('=' * 60)
    
    np.random.seed(1)  # Same seed for fair comparison
    cluttered_measurements_bp = generate_cluttered_measurements(measurements, parameters)
    
    (estimated_trajectory_bp, estimated_anchors_bp,
     _, num_estimated_anchors_bp) = bp_based_mint_slam(
        data_va, cluttered_measurements_bp, parameters, true_trajectory,
        association_mode='bp'
    )
    
    errors_bp = calc_distance(true_trajectory[0:2, :], estimated_trajectory_bp[0:2, :])
    mean_error_bp = np.mean(errors_bp)
    
    print(f'\n=== BP Mode Results ===')
    print(f'Mean position error: {mean_error_bp:.4f} m')
    print(f'Final anchors: Sensor1={num_estimated_anchors_bp[0, -1]}, Sensor2={num_estimated_anchors_bp[1, -1]}')
    
    # Summary
    print('\n' + '=' * 60)
    print('COMPARISON SUMMARY')
    print('=' * 60)
    print(f'{"Mode":<10} {"Mean Error (m)":<15} {"Sensor1 Anchors":<18} {"Sensor2 Anchors"}')
    print(f'{"GNN":<10} {mean_error_gnn:<15.4f} {num_estimated_anchors_gnn[0, -1]:<18} {num_estimated_anchors_gnn[1, -1]}')
    print(f'{"BP":<10} {mean_error_bp:<15.4f} {num_estimated_anchors_bp[0, -1]:<18} {num_estimated_anchors_bp[1, -1]}')


if __name__ == '__main__':
    main()
