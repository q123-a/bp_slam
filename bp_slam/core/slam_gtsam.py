# -*- coding: utf-8 -*-
"""
BP-GTSAM Fusion SLAM: combines BP data association with GTSAM optimization

This is the core implementation of Plan A:
- Keep BP for data association, get association probabilities
- Use GTSAM to replace particle filter for position optimization
"""

import numpy as np
import time
import copy
from ..utils.sampling import draw_samples_uniformly_circ, resample_systematic
from ..utils.motion_model import perform_prediction
from ..utils.distance import calc_distance
from .anchors import (init_anchors, predict_anchors, predict_measurements,
                     generate_new_anchors, delete_unreliable_va)
from .association import calculate_association_probabilities_ga
from .graph_backend import GraphBackend, GTSAM_AVAILABLE


def bp_gtsam_slam(data_va, cluttered_measurements, parameters, true_trajectory):
    """
    BP-GTSAM Fusion SLAM algorithm
    
    BP handles data association, GTSAM handles position optimization

    Args:
        data_va: Virtual anchor data list
        cluttered_measurements: Cluttered measurements (range + variance)
        parameters: Algorithm parameters dict
        true_trajectory: True trajectory for error calculation

    Returns:
        estimated_trajectory: Estimated agent trajectory (position + velocity)
        estimated_anchors: Estimated anchor positions and existence probabilities
        posterior_particles_anchors_storage: Stored anchor particles for analysis
        num_estimated_anchors: Number of estimated anchors per time step
    """
    if not GTSAM_AVAILABLE:
        raise RuntimeError("GTSAM is not installed. Please run: pip install gtsam")
    
    # Get measurement steps and sensor count
    num_steps = len(cluttered_measurements)
    num_sensors = len(cluttered_measurements[0])
    num_steps = min(num_steps, parameters['maxSteps'])

    # Read parameters
    num_particles = parameters['numParticles']
    detection_probability = parameters['detectionProbability']
    prior_mean = parameters['priorMean']
    survival_probability = parameters['survivalProbability']
    undetected_anchors_intensity = parameters['undetectedAnchorsIntensity'] * np.ones(num_sensors)
    birth_intensity = parameters['birthIntensity']
    clutter_intensity = parameters['clutterIntensity']
    unreliability_threshold = parameters['unreliabilityThreshold']
    exec_time_per_step = np.zeros(num_steps)
    known_track = parameters['known_track']
    scan_time = parameters.get('scanTime', 1.0)

    # Pre-allocate storage
    estimated_trajectory = np.zeros((4, num_steps))
    num_estimated_anchors = np.zeros((num_sensors, num_steps), dtype=int)
    storing_idx = list(range(29, num_steps, 30))
    posterior_particles_anchors_storage = [None] * len(storing_idx)

    # ====== Initialize GTSAM backend ======
    graph_backend = GraphBackend(parameters)
    initial_pose = prior_mean[:2, 0]
    initial_velocity = prior_mean[2:4, 0] if prior_mean.shape[0] >= 4 else np.zeros(2)
    graph_backend.initialize(initial_pose, initial_velocity)
    
    # Record initial state
    estimated_trajectory[:2, 0] = initial_pose
    estimated_trajectory[2:4, 0] = initial_velocity

    # ====== Initialize particle filter (only for BP data association) ======
    if known_track:
        posterior_particles_agent = np.tile(
            np.vstack([true_trajectory[:2, 0:1], np.zeros((2, 1))]),
            (1, num_particles)
        )
    else:
        posterior_particles_agent = np.zeros((4, num_particles))
        posterior_particles_agent[0:2, :] = draw_samples_uniformly_circ(
            prior_mean[0:2], parameters['UniformRadius_pos'], num_particles
        )
        posterior_particles_agent[2:4, :] = (
            np.tile(prior_mean[2:4].reshape(-1, 1), (1, num_particles)) +
            2 * parameters['UniformRadius_vel'] * np.random.rand(2, num_particles) -
            parameters['UniformRadius_vel']
        )

    # Initialize anchor states
    estimated_anchors, posterior_particles_anchors = init_anchors(
        parameters, data_va, num_steps, num_sensors
    )
    for sensor in range(num_sensors):
        num_estimated_anchors[sensor, 0] = len(estimated_anchors[sensor][0])

    # ====== Main loop ======
    for step in range(1, num_steps):
        start_time = time.time()
        timestamp = step * scan_time

        # --- A. Predict agent state (for BP computation) ---
        if known_track:
            predicted_particles_agent = np.tile(
                np.vstack([true_trajectory[:2, step:step+1], np.zeros((2, 1))]),
                (1, num_particles)
            )
        else:
            predicted_particles_agent = perform_prediction(posterior_particles_agent, parameters)

        # Particle mean as predicted position
        predicted_pose = np.mean(predicted_particles_agent, axis=1)

        # --- B. Collect sensor observations for GTSAM ---
        sensor_data_for_gtsam = []
        anchor_particles_for_gtsam = {}  # {(sensor_id, anchor_idx): particles array}
        
        # Initialize agent particle weights (for PF update)
        weights_sensors = np.full((num_particles, num_sensors), np.nan)

        for sensor in range(num_sensors):
            # Inherit anchor states from previous step
            estimated_anchors[sensor][step] = copy.deepcopy(estimated_anchors[sensor][step - 1])
            measurements = cluttered_measurements[step][sensor]

            if measurements is None or measurements.size == 0:
                num_measurements = 0
            else:
                num_measurements = measurements.shape[1]

            # Predict undetected anchor intensity
            undetected_anchors_intensity[sensor] = (
                undetected_anchors_intensity[sensor] * survival_probability + birth_intensity
            )

            # Predict anchor particles and weights
            predicted_particles_anchors, weights_anchor = predict_anchors(
                posterior_particles_anchors[sensor], parameters
            )

            # Generate new anchor particles
            new_particles_anchors, new_input_bp = generate_new_anchors(
                measurements, undetected_anchors_intensity[sensor],
                predicted_particles_agent, parameters
            )

            # Predict measurements
            predicted_measurements_bp, predicted_uncertainties, predicted_range = predict_measurements(
                predicted_particles_agent, predicted_particles_anchors, weights_anchor
            )

            # ====== BP Data Association ======
            (association_probabilities, association_probabilities_new,
             message_lhf_ratios, messages_new) = calculate_association_probabilities_ga(
                measurements, predicted_measurements_bp, predicted_uncertainties,
                weights_anchor, new_input_bp, parameters
            )

            # --- C. Process existing anchors, collect GTSAM observations ---
            num_anchors = predicted_particles_anchors.shape[2]
            weights = np.zeros((num_particles, num_anchors))

            if num_measurements > 0:
                weights[:, :] = (1 - detection_probability)
                measurement_variances = measurements[1, :]
                factors = (1 / np.sqrt(2 * np.pi * measurement_variances) *
                          detection_probability / clutter_intensity)
                range_diff = measurements[0, :][np.newaxis, np.newaxis, :] - predicted_range[:, :, np.newaxis]
                weight_contributions = (
                    factors[np.newaxis, np.newaxis, :] *
                    message_lhf_ratios.T[np.newaxis, :, :] *
                    np.exp(-0.5 / measurement_variances[np.newaxis, np.newaxis, :] * range_diff**2)
                )
                weights += np.sum(weight_contributions, axis=2)
            else:
                weights[:, :] = (1 - detection_probability)

            # Update anchors and collect GTSAM observations
            for anchor in range(num_anchors):
                predicted_existence = np.sum(weights_anchor[:, anchor])
                
                # Compute posterior existence probability
                alive_update = np.sum(predicted_existence * (1 / num_particles) * weights[:, anchor])
                dead_update = 1 - predicted_existence
                posterior_existence = alive_update / (alive_update + dead_update + 1e-10)
                posterior_particles_anchors[sensor][anchor]['posteriorExistence'] = posterior_existence

                # Resample particles
                weight_sum = np.sum(weights[:, anchor])
                if weight_sum > 0:
                    idx_resampling = resample_systematic(weights[:, anchor] / weight_sum, num_particles)
                else:
                    idx_resampling = np.arange(num_particles)

                posterior_particles_anchors[sensor][anchor]['x'] = (
                    predicted_particles_anchors[:, idx_resampling, anchor]
                )
                posterior_particles_anchors[sensor][anchor]['w'] = (
                    posterior_existence / num_particles * np.ones(num_particles)
                )

                # Anchor position estimate
                anchor_pos = np.mean(posterior_particles_anchors[sensor][anchor]['x'], axis=1)
                estimated_anchors[sensor][step][anchor]['x'] = anchor_pos
                estimated_anchors[sensor][step][anchor]['posteriorExistence'] = posterior_existence

                # ====== Prepare observations for GTSAM ======
                if num_measurements > 0 and posterior_existence > 0.5:
                    # Get measurement with highest association probability
                    assoc_probs = association_probabilities[1:, anchor]
                    best_meas_idx = np.argmax(assoc_probs)
                    best_prob = assoc_probs[best_meas_idx]
                    
                    if best_prob > 0.5:
                        # Use generatedAt as stable anchor ID (not affected by deletions)
                        generated_at = estimated_anchors[sensor][step][anchor].get('generatedAt', 0)
                        stable_anchor_id = generated_at * 1000 + anchor  # Unique ID
                        
                        sensor_data_for_gtsam.append({
                            'sensor_id': sensor,
                            'anchor_idx': stable_anchor_id,  # Use stable ID
                            'range': measurements[0, best_meas_idx],
                            'prob': best_prob,
                            'existence': posterior_existence,
                            'position': anchor_pos
                        })
                        
                        # Collect PF particles for this anchor (for convergence check)
                        anchor_particles_for_gtsam[(sensor, stable_anchor_id)] = (
                            posterior_particles_anchors[sensor][anchor]['x'].copy()
                        )

                # Update weight log
                weights[:, anchor] = predicted_existence * weights[:, anchor] + dead_update
                weights[:, anchor] = np.log(weights[:, anchor] + 1e-10)
                weights[:, anchor] = weights[:, anchor] - np.max(weights[:, anchor])

            # Accumulate agent particle weights from this sensor
            weights_sensors[:, sensor] = np.sum(weights, axis=1)
            weights_sensors[:, sensor] = weights_sensors[:, sensor] - np.max(weights_sensors[:, sensor])
            
            num_estimated_anchors[sensor, step] = len(estimated_anchors[sensor][step])

            # Update undetected anchor intensity
            undetected_anchors_intensity[sensor] = (
                undetected_anchors_intensity[sensor] * (1 - detection_probability)
            )

            # Process new anchors
            for measurement in range(num_measurements):
                new_anchor_idx = num_anchors + measurement
                constant = new_particles_anchors[measurement]['constant']
                posterior_existence = (
                    messages_new[measurement] * constant /
                    (messages_new[measurement] * constant + 1)
                )

                if new_anchor_idx >= len(posterior_particles_anchors[sensor]):
                    posterior_particles_anchors[sensor].append({
                        'x': new_particles_anchors[measurement]['x'],
                        'w': posterior_existence / num_particles,
                        'posteriorExistence': posterior_existence
                    })
                    estimated_anchors[sensor][step].append({
                        'x': np.mean(new_particles_anchors[measurement]['x'], axis=1),
                        'posteriorExistence': posterior_existence,
                        'generatedAt': step
                    })
                else:
                    posterior_particles_anchors[sensor][new_anchor_idx]['posteriorExistence'] = posterior_existence
                    posterior_particles_anchors[sensor][new_anchor_idx]['x'] = new_particles_anchors[measurement]['x']
                    posterior_particles_anchors[sensor][new_anchor_idx]['w'] = posterior_existence / num_particles
                    estimated_anchors[sensor][step][new_anchor_idx]['x'] = np.mean(
                        new_particles_anchors[measurement]['x'], axis=1
                    )
                    estimated_anchors[sensor][step][new_anchor_idx]['posteriorExistence'] = posterior_existence
                    estimated_anchors[sensor][step][new_anchor_idx]['generatedAt'] = step

            # Delete unreliable anchors
            estimated_anchors[sensor][step], posterior_particles_anchors[sensor] = delete_unreliable_va(
                estimated_anchors[sensor][step], posterior_particles_anchors[sensor],
                unreliability_threshold
            )
            num_estimated_anchors[sensor, step] = len(estimated_anchors[sensor][step])

        # --- Normalize agent particle weights and compute PF estimate ---
        weights_sensors = np.sum(weights_sensors, axis=1)
        weights_sensors = weights_sensors - np.max(weights_sensors)
        weights_sensors = np.exp(weights_sensors)
        weights_sensors = weights_sensors / (np.sum(weights_sensors) + 1e-10)
        
        # PF weighted position estimate
        pf_position = predicted_particles_agent @ weights_sensors
        
        # --- D. GTSAM graph optimization update (with two-phase support) ---
        # Pass PF particles for convergence-based initialization ("handoff" strategy)
        gtsam_pose, gtsam_velocity, current_phase = graph_backend.update(
            timestamp, 
            sensor_data_for_gtsam, 
            predicted_pose[:2],
            anchor_particles=anchor_particles_for_gtsam,
            pf_position=pf_position[:2]
        )

        # Store anchor particles at key time steps
        if step in storing_idx:
            idx = storing_idx.index(step)
            posterior_particles_anchors_storage[idx] = [
                [anchor.copy() for anchor in sensor_anchors]
                for sensor_anchors in posterior_particles_anchors
            ]

        # --- E. Update agent particles based on current phase ---
        if known_track:
            estimated_trajectory[:, step] = np.mean(predicted_particles_agent, axis=1)
            posterior_particles_agent = predicted_particles_agent
        else:
            if current_phase == 0:  # PHASE_PF
                # Phase 1: Pure PF mode - use weighted estimate and resample
                estimated_trajectory[:, step] = pf_position
                # Resample particles based on weights (proper PF update)
                posterior_particles_agent = predicted_particles_agent[
                    :, resample_systematic(weights_sensors, num_particles)
                ]
            else:  # PHASE_GTSAM
                # Phase 2: Use GTSAM optimized result
                estimated_trajectory[0:2, step] = gtsam_pose
                estimated_trajectory[2:4, step] = gtsam_velocity
                
                # Update particle cloud (redistribute around GTSAM estimate)
                posterior_particles_agent[0:2, :] = (
                    gtsam_pose[:, np.newaxis] + 
                    0.05 * np.random.randn(2, num_particles)
                )
                posterior_particles_agent[2:4, :] = (
                    gtsam_velocity[:, np.newaxis] + 
                    0.01 * np.random.randn(2, num_particles)
                )

        # Compute and print error
        exec_time_per_step[step] = time.time() - start_time
        error_agent = calc_distance(
            true_trajectory[0:2, step:step+1],
            estimated_trajectory[0:2, step:step+1]
        )
        if isinstance(error_agent, np.ndarray):
            error_agent = error_agent.item() if error_agent.size == 1 else error_agent[0]

        phase_str = "PF" if current_phase == 0 else "GTSAM"
        print(f'Time instance: {step + 1} [{phase_str}]')
        for sensor in range(num_sensors):
            print(f'Number of Anchors Sensor {sensor + 1}: {num_estimated_anchors[sensor, step]}')
        print(f'Position error agent: {error_agent:.6f}')
        print(f'Active landmarks in graph: {len(graph_backend.feat_mgr.active_features)}')
        print(f'Execution Time: {exec_time_per_step[step]:.4f}')
        print('---------------------------------------------------\n')

    return estimated_trajectory, estimated_anchors, posterior_particles_anchors_storage, num_estimated_anchors
