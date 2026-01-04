# -*- coding: utf-8 -*-
"""
BP-GTSAM Fusion SLAM Test Script
Test script for BP-GTSAM fusion SLAM algorithm

Based on testbed.py, using GTSAM graph optimization to replace particle filter position update
"""

import numpy as np
import scipy.io as sio
from bp_slam.utils.measurements import generate_measurements, generate_cluttered_measurements
from bp_slam.core.slam_gtsam import bp_gtsam_slam


def main():
    """Main test function"""

    # ---------------------------
    # 1. Load data and parameters
    # ---------------------------
    parameters = {}
    parameters['known_track'] = 0  # 0 = unknown trajectory

    # Load scenario data
    mat_data = sio.loadmat('scenarioCleanM2_new.mat')
    data_va_raw = mat_data['dataVA'][:, 0]
    true_trajectory = mat_data['trueTrajectory']

    # Set all anchors to visible
    num_sensors = len(data_va_raw)
    data_va = []
    for sensor in range(num_sensors):
        sensor_data = {
            'positions': data_va_raw[sensor]['positions'][0, 0],
            'visibility': np.ones((data_va_raw[sensor]['positions'][0, 0].shape[1],
                                  true_trajectory.shape[1]))
        }
        data_va.append(sensor_data)

    # ---------------------------
    # 2. Algorithm parameters
    # ---------------------------
    parameters['maxSteps'] = 200  # Extended test for turn detection
    true_trajectory = true_trajectory[:, :parameters['maxSteps']]
    parameters['lengthStep'] = 0.03
    parameters['scanTime'] = 1

    # Max velocity and process noise
    v_max = parameters['lengthStep'] / parameters['scanTime']
    parameters['drivingNoiseVariance'] = (v_max / 3 / parameters['scanTime'])**2

    # Measurement noise
    parameters['measurementVariance'] = 0.1**2
    parameters['measurementVarianceLHF'] = 0.15**2

    # Detection probability
    parameters['detectionProbability'] = 0.95

    # Region size and clutter
    parameters['regionOfInterestSize'] = 30
    parameters['meanNumberOfClutter'] = 1
    parameters['clutterIntensity'] = (parameters['meanNumberOfClutter'] /
                                     parameters['regionOfInterestSize'])

    # Birth rate
    parameters['meanNumberOfBirth'] = 1e-4
    parameters['birthIntensity'] = (parameters['meanNumberOfBirth'] /
                                   (2 * parameters['regionOfInterestSize'])**2)

    # Undetected anchors
    parameters['meanNumberOfUndetectedAnchors'] = 6
    parameters['undetectedAnchorsIntensity'] = (parameters['meanNumberOfUndetectedAnchors'] /
                                               (2 * parameters['regionOfInterestSize'])**2)

    # Particle filter parameters (still needed for BP)
    parameters['numParticles'] = 100000
    parameters['upSamplingFactor'] = 1

    # SLAM thresholds and priors
    parameters['detectionThreshold'] = 0.5
    parameters['survivalProbability'] = 0.999
    parameters['unreliabilityThreshold'] = 1e-4
    parameters['priorKnownAnchors'] = [[0], [0]]
    parameters['priorCovarianceAnchor'] = 0.001**2 * np.eye(2)
    parameters['anchorRegularNoiseVariance'] = 1e-4**2

    # Agent parameters
    parameters['UniformRadius_pos'] = 0.5
    parameters['UniformRadius_vel'] = 0.05

    # ---------------------------
    # 3. Random seed
    # ---------------------------
    np.random.seed(1)

    # ---------------------------
    # 4. Initial position
    # ---------------------------
    parameters['priorMean'] = np.vstack([true_trajectory[0:2, 0:1], np.zeros((2, 1))])

    # ---------------------------
    # 5. Generate ideal measurements
    # ---------------------------
    print("Generating ideal measurements...")
    measurements = generate_measurements(true_trajectory, data_va, parameters)

    # ---------------------------
    # 6. Add clutter and missed detections
    # ---------------------------
    print("Generating cluttered measurements...")
    cluttered_measurements = generate_cluttered_measurements(measurements, parameters)

    # ---------------------------
    # 7. Run BP-GTSAM Fusion SLAM
    # ---------------------------
    print("\nStarting BP-GTSAM Fusion SLAM...\n")
    print("=" * 50)
    print("Using BP for data association, GTSAM for optimization")
    print("=" * 50 + "\n")
    
    (estimated_trajectory, estimated_anchors,
     posterior_particles_anchors, num_estimated_anchors) = bp_gtsam_slam(
        data_va, cluttered_measurements, parameters, true_trajectory
    )

    print("\n" + "=" * 50)
    print("Algorithm completed!")
    print(f"Final anchor count - Sensor 1: {num_estimated_anchors[0, -1]}")
    if num_sensors > 1:
        print(f"Final anchor count - Sensor 2: {num_estimated_anchors[1, -1]}")

    # ---------------------------
    # 8. Compute statistics
    # ---------------------------
    from bp_slam.utils.distance import calc_distance
    
    position_errors = []
    for step in range(parameters['maxSteps']):
        err = calc_distance(
            true_trajectory[0:2, step:step+1],
            estimated_trajectory[0:2, step:step+1]
        )
        if isinstance(err, np.ndarray):
            err = err.item() if err.size == 1 else err[0]
        position_errors.append(err)
    
    position_errors = np.array(position_errors)
    print(f"\nPosition error statistics:")
    print(f"  Mean error: {np.mean(position_errors):.4f} m")
    print(f"  Max error: {np.max(position_errors):.4f} m")
    print(f"  Min error: {np.min(position_errors):.4f} m")
    print(f"  Std dev: {np.std(position_errors):.4f} m")
    print(f"  RMSE: {np.sqrt(np.mean(position_errors**2)):.4f} m")

    # ---------------------------
    # 9. Save results
    # ---------------------------
    from pathlib import Path
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    print("\nSaving results to results/results_gtsam.npz...")
    np.savez(results_dir / 'results_gtsam.npz',
             estimated_trajectory=estimated_trajectory,
             true_trajectory=true_trajectory,
             num_estimated_anchors=num_estimated_anchors,
             estimated_anchors=np.array(estimated_anchors, dtype=object),
             posterior_particles_anchors=np.array(posterior_particles_anchors, dtype=object),
             parameters=parameters,
             position_errors=position_errors,
             allow_pickle=True)

    print("Done! Results saved to results/results_gtsam.npz")

    # ---------------------------
    # 10. Visualization
    # ---------------------------
    print("\nGenerating visualization...")
    try:
        from bp_slam.visualization.visualizer import visualize_online

        last_particles = posterior_particles_anchors[-1] if len(posterior_particles_anchors) > 0 and posterior_particles_anchors[-1] is not None else None

        stats = visualize_online(
            true_trajectory, estimated_trajectory, estimated_anchors,
            last_particles, data_va, parameters,
            scene_file='scen_semroom_new.mat',
            output_dir='results',
            save=True,
            show=True,
            filename_suffix='_gtsam'
        )
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Using simple plotting...")
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Trajectory comparison
        ax1 = axes[0]
        ax1.plot(true_trajectory[0, :], true_trajectory[1, :], 'b-', label='True Trajectory', linewidth=2)
        ax1.plot(estimated_trajectory[0, :], estimated_trajectory[1, :], 'r--', label='BP-GTSAM Estimated', linewidth=2)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Trajectory Comparison')
        ax1.legend()
        ax1.grid(True)
        ax1.axis('equal')
        
        # Plot 2: Position error
        ax2 = axes[1]
        ax2.plot(position_errors, 'b-', linewidth=1)
        ax2.axhline(y=np.mean(position_errors), color='r', linestyle='--', label=f'Mean: {np.mean(position_errors):.4f} m')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Position Error (m)')
        ax2.set_title('Position Error over Time')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'results_gtsam.png', dpi=150)
        print(f"Figure saved to results/results_gtsam.png")
        plt.show()

    return estimated_trajectory, estimated_anchors, num_estimated_anchors


if __name__ == '__main__':
    main()
