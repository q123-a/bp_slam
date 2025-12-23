# -*- coding: utf-8 -*-
"""
Label generation for GAT-based data association training.

Core concept (God's View / Factory Analogy):
    
    In simulation, we are the "factory" that manufactures measurements.
    We don't need to guess which anchor a measurement comes from - 
    we STAMP the source_id on it at the moment of creation!
    
    Real SLAM (Inference): Like a detective finding a bullet casing and 
                           trying to match it to a gun by rifling marks.
    
    Simulation (Training): Like a factory stamping serial numbers on bullets
                           at production time. We KNOW which gun it belongs to.

Two types of measurements:
    1. Real measurements: source_id = anchor_index (0, 1, 2, ...)
    2. Clutter (fake):    source_id = -1 (marked as "garbage" at birth)

Usage:
    # Generate measurements WITH labels (recommended)
    measurements, labels = generate_measurements_with_labels(...)
    
    # labels[step][sensor] = array([4, 2, -1, 0, 1])
    #   4, 2, 0, 1 = anchor indices (real measurements)
    #   -1 = clutter (fake measurement)
"""

import numpy as np


def generate_measurements_with_labels(target_trajectory, data_va, parameters):
    """
    Generate measurements with ground truth labels (source_id stamped at creation).
    
    This implementation matches the EXACT random number sequence of the original
    two-step implementation (generate_measurements + generate_cluttered_measurements)
    for reproducibility with the same random seed.
    
    Args:
        target_trajectory: Agent trajectory (2, num_steps)
        data_va: Virtual anchor data list [sensor] = {positions, visibility}
        parameters: Parameter dictionary
    
    Returns:
        measurements: [step][sensor] = (2, M) array, row0=distance, row1=variance
        labels: [step][sensor] = (M,) array of source_ids
            - label >= 0: real measurement from anchor[label]
            - label == -1: clutter (fake measurement)
    """
    measurement_variance = parameters['measurementVariance']
    measurement_variance_lhf = parameters.get('measurementVarianceLHF', measurement_variance)
    detection_probability = parameters['detectionProbability']
    mean_number_of_clutter = parameters['meanNumberOfClutter']
    max_range = parameters['regionOfInterestSize']
    
    num_steps = target_trajectory.shape[1]
    num_sensors = len(data_va)
    
    # ================================================================
    # PHASE 1: Generate all true measurements (same as generate_measurements)
    # Random calls: randn() for each visible anchor
    # ================================================================
    true_measurements_cell = [[None for _ in range(num_sensors)] for _ in range(num_steps)]
    true_labels_cell = [[None for _ in range(num_sensors)] for _ in range(num_steps)]
    
    for sensor in range(num_sensors):
        anchor_positions = data_va[sensor]['positions']  # (2, num_anchors)
        visibility = data_va[sensor]['visibility']       # (num_anchors, num_steps)
        num_anchors = anchor_positions.shape[1]
        
        for step in range(num_steps):
            agent_pos = target_trajectory[:, step]
            
            measurements_list = []
            labels_list = []
            
            for anchor_idx in range(num_anchors):
                if visibility[anchor_idx, step]:
                    # Calculate TRUE distance
                    true_distance = np.sqrt(
                        (anchor_positions[0, anchor_idx] - agent_pos[0])**2 +
                        (anchor_positions[1, anchor_idx] - agent_pos[1])**2
                    )
                    
                    # Add Gaussian noise (same random call order as original)
                    noise = np.sqrt(measurement_variance) * np.random.randn()
                    measured_distance = true_distance + noise
                    
                    measurements_list.append([measured_distance, measurement_variance_lhf])
                    labels_list.append(anchor_idx)  # Stamp source_id!
            
            if len(measurements_list) > 0:
                true_measurements_cell[step][sensor] = np.array(measurements_list).T
                true_labels_cell[step][sensor] = np.array(labels_list, dtype=int)
            else:
                true_measurements_cell[step][sensor] = np.zeros((2, 0))
                true_labels_cell[step][sensor] = np.array([], dtype=int)
    
    # ================================================================
    # PHASE 2: Add detection + clutter (same as generate_cluttered_measurements)
    # Random calls: rand() for detection, poisson() for clutter count,
    #               rand() for clutter distances, permutation() for shuffle
    # ================================================================
    measurements = [[None for _ in range(num_sensors)] for _ in range(num_steps)]
    labels = [[None for _ in range(num_sensors)] for _ in range(num_steps)]
    
    for sensor in range(num_sensors):
        for step in range(num_steps):
            true_meas = true_measurements_cell[step][sensor]
            true_lab = true_labels_cell[step][sensor]
            
            if true_meas is None or true_meas.size == 0:
                num_anchors = 0
                detected_meas = np.zeros((2, 0))
                detected_lab = np.array([], dtype=int)
            else:
                num_anchors = true_meas.shape[1]
                
                # Detection mask (same random call as original)
                detection_indicator = np.random.rand(num_anchors) < detection_probability
                
                detected_meas = true_meas[:, detection_indicator]
                detected_lab = true_lab[detection_indicator]
            
            # Generate clutter count (same random call as original)
            num_clutter = np.random.poisson(mean_number_of_clutter)
            
            # Generate clutter measurements
            if num_clutter > 0:
                clutter_distances = max_range * np.random.rand(num_clutter)
                clutter_meas = np.zeros((2, num_clutter))
                clutter_meas[0, :] = clutter_distances
                clutter_meas[1, :] = measurement_variance
                clutter_lab = np.full(num_clutter, -1, dtype=int)  # -1 = clutter
            else:
                clutter_meas = np.zeros((2, 0))
                clutter_lab = np.array([], dtype=int)
            
            # Combine: clutter first, then detected (same as original hstack order)
            if detected_meas.size > 0:
                combined_meas = np.hstack([clutter_meas, detected_meas])
                combined_lab = np.concatenate([clutter_lab, detected_lab])
            else:
                combined_meas = clutter_meas
                combined_lab = clutter_lab
            
            # Shuffle (same random call as original)
            if combined_meas.shape[1] > 0:
                perm = np.random.permutation(combined_meas.shape[1])
                combined_meas = combined_meas[:, perm]
                combined_lab = combined_lab[perm]
            
            measurements[step][sensor] = combined_meas
            labels[step][sensor] = combined_lab
    
    return measurements, labels
    
    return measurements, labels


def labels_to_association_matrix(labels, num_anchors):
    """
    Convert source_id labels to association matrix format.
    
    Args:
        labels: (M,) array of source_ids (-1 for clutter, >=0 for anchor index)
        num_anchors: Total number of anchors N
    
    Returns:
        association_matrix: (M, N+1) binary matrix
            - association_matrix[m, n] = 1: measurement m from anchor n
            - association_matrix[m, N] = 1: measurement m is clutter
        detection_labels: (N,) binary vector, 1 if anchor was detected
    """
    num_measurements = len(labels)
    association_matrix = np.zeros((num_measurements, num_anchors + 1))
    detection_labels = np.zeros(num_anchors)
    
    for m, source_id in enumerate(labels):
        if source_id == -1:
            association_matrix[m, -1] = 1  # Clutter column
        else:
            association_matrix[m, source_id] = 1
            detection_labels[source_id] = 1
    
    return association_matrix, detection_labels


def labels_to_edge_format(labels, num_anchors):
    """
    Convert source_id labels to edge format for GNN training.
    
    Args:
        labels: (M,) array of source_ids
        num_anchors: Number of anchors N
    
    Returns:
        edge_index: (2, M*N) edge indices [measurement_idx; anchor_idx]
        edge_labels: (M*N,) binary labels, 1 if edge represents true match
        clutter_labels: (M,) binary labels, 1 if measurement is clutter
    """
    num_measurements = len(labels)
    
    # Build all possible edges (M measurements ¡Á N anchors)
    edge_index = []
    edge_labels = []
    
    for m in range(num_measurements):
        for a in range(num_anchors):
            edge_index.append([m, a])
            edge_labels.append(1 if labels[m] == a else 0)
    
    edge_index = np.array(edge_index).T if edge_index else np.zeros((2, 0), dtype=int)
    edge_labels = np.array(edge_labels) if edge_labels else np.array([])
    clutter_labels = (labels == -1).astype(int)
    
    return edge_index, edge_labels, clutter_labels


# ============================================================
# Legacy function for backward compatibility
# (Use this when you only have cluttered measurements without labels)
# ============================================================

from scipy.optimize import linear_sum_assignment


def generate_association_labels(
    agent_position,
    anchor_positions,
    cluttered_measurements,
    measurement_variance,
    threshold_sigma=3.0
):
    """
    Generate data association labels from ground truth (single time step).
    
    Principle:
        1. Compute true distance from agent to each anchor
        2. Match each measurement to true distances
        3. Match success -> associate with anchor
        4. Match failure -> mark as clutter
    
    Args:
        agent_position: Agent true position (2,)
        anchor_positions: Anchor true positions (2, N)
        cluttered_measurements: Cluttered measurements (2, M), row0=distance, row1=variance
        measurement_variance: Measurement variance (for threshold calculation)
        threshold_sigma: Threshold multiplier (default 3 sigma)
    
    Returns:
        association_matrix: (M, N+1) association matrix
            - association_matrix[m, n] = 1: measurement m associates with anchor n
            - association_matrix[m, N] = 1: measurement m is clutter
        detection_labels: (N,) detection labels
            - detection_labels[n] = 1: anchor n is detected
            - detection_labels[n] = 0: anchor n is missed
        match_info: dict with detailed matching information
    """
    num_anchors = anchor_positions.shape[1]
    
    # Compute true distances (agent to each anchor)
    true_distances = np.sqrt(
        (anchor_positions[0, :] - agent_position[0])**2 + 
        (anchor_positions[1, :] - agent_position[1])**2
    )
    
    # Handle empty measurements
    if cluttered_measurements is None or cluttered_measurements.size == 0:
        return (
            np.zeros((0, num_anchors + 1)),
            np.zeros(num_anchors),
            {'true_distances': true_distances, 'num_clutter': 0, 'num_missed': num_anchors}
        )
    
    num_measurements = cluttered_measurements.shape[1]
    measured_distances = cluttered_measurements[0, :]
    
    # Matching threshold = threshold_sigma * std
    threshold = threshold_sigma * np.sqrt(measurement_variance)
    
    # Compute distance difference matrix (M, N)
    distance_diffs = np.abs(
        measured_distances[:, np.newaxis] - true_distances[np.newaxis, :]
    )
    
    # Use Hungarian algorithm for optimal matching
    association_matrix, detection_labels, match_info = _hungarian_matching(
        distance_diffs, threshold, num_measurements, num_anchors
    )
    
    match_info['true_distances'] = true_distances
    match_info['measured_distances'] = measured_distances
    match_info['threshold'] = threshold
    
    return association_matrix, detection_labels, match_info


def _hungarian_matching(distance_diffs, threshold, num_measurements, num_anchors):
    """
    Use Hungarian algorithm for optimal bipartite matching.
    
    Args:
        distance_diffs: (M, N) distance difference matrix
        threshold: matching threshold
        num_measurements: number of measurements M
        num_anchors: number of anchors N
    
    Returns:
        association_matrix: (M, N+1)
        detection_labels: (N,)
        match_info: dict
    """
    # Build cost matrix (set large value for exceeding threshold)
    cost_matrix = distance_diffs.copy()
    cost_matrix[cost_matrix > threshold] = 1e6
    
    # Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Initialize outputs
    association_matrix = np.zeros((num_measurements, num_anchors + 1))
    detection_labels = np.zeros(num_anchors)
    
    matched_measurements = []
    matched_anchors = []
    
    # Process matching results
    for m_idx, a_idx in zip(row_indices, col_indices):
        if a_idx < num_anchors and distance_diffs[m_idx, a_idx] <= threshold:
            # Valid match
            association_matrix[m_idx, a_idx] = 1
            detection_labels[a_idx] = 1
            matched_measurements.append(m_idx)
            matched_anchors.append(a_idx)
        else:
            # Match failed (exceeds threshold) -> clutter
            association_matrix[m_idx, num_anchors] = 1
    
    # Unmatched measurements -> clutter
    for m_idx in range(num_measurements):
        if m_idx not in matched_measurements:
            association_matrix[m_idx, num_anchors] = 1
    
    match_info = {
        'matched_pairs': list(zip(matched_measurements, matched_anchors)),
        'num_clutter': int(association_matrix[:, -1].sum()),
        'num_missed': int(num_anchors - detection_labels.sum()),
        'num_matched': len(matched_measurements)
    }
    
    return association_matrix, detection_labels, match_info


def generate_labels_for_sequence(
    true_trajectory,
    data_va,
    cluttered_measurements,
    parameters,
    num_steps=None
):
    """
    Generate labels for entire time sequence.
    
    Args:
        true_trajectory: True trajectory (2, T)
        data_va: Virtual anchor data list [sensor0, sensor1, ...]
        cluttered_measurements: Cluttered measurements [step][sensor]
        parameters: Parameter dictionary
        num_steps: Number of time steps (None=all)
    
    Returns:
        labels_all: List [step][sensor] = {
            'association_matrix': (M, N+1),
            'detection_labels': (N,),
            'match_info': dict
        }
    """
    if num_steps is None:
        num_steps = len(cluttered_measurements)
    
    num_sensors = len(data_va)
    measurement_variance = parameters['measurementVariance']
    
    labels_all = [[None for _ in range(num_sensors)] for _ in range(num_steps)]
    
    for step in range(num_steps):
        agent_pos = true_trajectory[:, step]
        
        for sensor in range(num_sensors):
            anchor_positions = data_va[sensor]['positions']
            clut_meas = cluttered_measurements[step][sensor]
            
            assoc_matrix, detect_labels, match_info = generate_association_labels(
                agent_pos, anchor_positions, clut_meas, measurement_variance
            )
            
            labels_all[step][sensor] = {
                'association_matrix': assoc_matrix,
                'detection_labels': detect_labels,
                'match_info': match_info
            }
    
    return labels_all


def association_matrix_to_edge_format(association_matrix):
    """
    Convert association matrix to edge format (for GNN).
    (Legacy function - use labels_to_edge_format with source_id labels instead)
    
    Args:
        association_matrix: (M, N+1) association matrix
    
    Returns:
        edge_index: (2, E) edge indices, edge_index[0]=measurement, edge_index[1]=anchor
        edge_labels: (E,) edge labels, 1=match, 0=no match
        clutter_labels: (M,) clutter labels for each measurement
    """
    num_measurements, num_anchors_plus_clutter = association_matrix.shape
    num_anchors = num_anchors_plus_clutter - 1
    
    # Build all possible edges (measurement-anchor pairs)
    edge_index = []
    edge_labels = []
    
    for m in range(num_measurements):
        for a in range(num_anchors):
            edge_index.append([m, a])
            edge_labels.append(association_matrix[m, a])
    
    edge_index = np.array(edge_index).T  # (2, E)
    edge_labels = np.array(edge_labels)  # (E,)
    
    # Clutter labels (whether each measurement is clutter)
    clutter_labels = association_matrix[:, -1]  # (M,)
    
    return edge_index, edge_labels, clutter_labels


def print_label_summary(labels, step, sensor):
    """Print label summary."""
    label = labels[step][sensor]
    assoc = label['association_matrix']
    detect = label['detection_labels']
    info = label['match_info']
    
    num_measurements = assoc.shape[0]
    num_anchors = assoc.shape[1] - 1
    
    print(f"\n{'='*60}")
    print(f"Step {step}, Sensor {sensor} Label Summary")
    print(f"{'='*60}")
    print(f"Measurements: {num_measurements}, Anchors: {num_anchors}")
    print(f"Matched: {info['num_matched']}, Clutter: {info['num_clutter']}, Missed: {info['num_missed']}")
    print(f"Threshold: {info['threshold']:.4f}m")
    
    print(f"\nMatched pairs (measurement -> anchor):")
    for m, a in info['matched_pairs']:
        print(f"  Meas{m} ({info['measured_distances'][m]:.3f}m) -> Anchor{a} ({info['true_distances'][a]:.3f}m)")
    
    print(f"\nClutter measurements:")
    for m in range(num_measurements):
        if assoc[m, -1] == 1:
            print(f"  Meas{m}: {info['measured_distances'][m]:.3f}m")
    
    print(f"\nMissed anchors:")
    for a in range(num_anchors):
        if detect[a] == 0:
            print(f"  Anchor{a}: {info['true_distances'][a]:.3f}m")


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    import scipy.io as sio
    from bp_slam.utils.measurements import generate_measurements, generate_cluttered_measurements
    
    # Load data
    mat_data = sio.loadmat('scenarioCleanM2_new.mat')
    data_va_raw = mat_data['dataVA'][:, 0]
    true_trajectory = mat_data['trueTrajectory']
    
    # Convert data format
    num_sensors = len(data_va_raw)
    data_va = []
    for sensor in range(num_sensors):
        data_va.append({
            'positions': data_va_raw[sensor]['positions'][0, 0],
            'visibility': np.ones((data_va_raw[sensor]['positions'][0, 0].shape[1], 
                                   true_trajectory.shape[1]))
        })
    
    # Parameters
    parameters = {
        'measurementVariance': 0.1**2,
        'measurementVarianceLHF': 0.15**2,
        'detectionProbability': 0.95,
        'meanNumberOfClutter': 1,
        'regionOfInterestSize': 30,
    }
    
    np.random.seed(42)
    
    # Generate measurements
    num_steps = 20
    true_meas = generate_measurements(
        true_trajectory[:, :num_steps],
        [{**d, 'visibility': d['visibility'][:, :num_steps]} for d in data_va],
        parameters
    )
    cluttered_meas = generate_cluttered_measurements(true_meas, parameters)
    
    # Generate labels
    labels = generate_labels_for_sequence(
        true_trajectory[:, :num_steps],
        data_va,
        cluttered_meas,
        parameters
    )
    
    # Print label summary for some time steps
    for step in [0, 5, 10]:
        print_label_summary(labels, step, sensor=0)
    
    # Edge format example
    print("\n" + "="*60)
    print("Edge format example (step 5, sensor 0)")
    print("="*60)
    edge_index, edge_labels, clutter_labels = labels_to_edge_format(
        labels[5][0]['association_matrix']
    )
    print(f"Number of edges: {edge_index.shape[1]}")
    print(f"Positive edges (matched): {int(edge_labels.sum())}")
    print(f"Negative edges (not matched): {int(len(edge_labels) - edge_labels.sum())}")
    print(f"Clutter measurements: {int(clutter_labels.sum())}")
