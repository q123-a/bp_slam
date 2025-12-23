"""
BP-based Multipath-assisted SLAM core algorithm
Supports multiple data association modes: BP, Ground Truth, GNN

Author: Florian Meyer, Erik Leitinger, 20/05/17
Converted to Python: 2025
Extended with multi-mode support: 2025
"""

import numpy as np
import time
import copy
import torch
from ..utils.sampling import draw_samples_uniformly_circ, resample_systematic
from ..utils.motion_model import perform_prediction
from ..utils.distance import calc_distance
from .anchors import (init_anchors, predict_anchors, predict_measurements,
                     generate_new_anchors, delete_unreliable_va)
from .association import calculate_association_probabilities_ga


def _create_training_sample(
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
        sample: dict with GNN inputs and labels, or None if no measurements
    """
    if measurements is None or measurements.size == 0:
        return None
    
    M = measurements.shape[1]  # number of measurements
    N = len(predicted_measurements)  # number of existing anchors in SLAM state
    
    # Ensure N matches existing_truth_ids length
    actual_N = len(existing_truth_ids)
    if N != actual_N:
        # Use actual_N as the true anchor count
        N = actual_N
    
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
        x_anchor[:, 0] = predicted_measurements[:N]
        x_anchor[:, 1] = predicted_uncertainties[:N]
        x_anchor[:, 2] = anchor_existence[:N]
    else:
        # No anchors yet - create dummy anchor node for graph structure
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
    y_match = np.zeros((M, N), dtype=np.float32)
    y_new = np.zeros(M, dtype=np.float32)
    match_labels = np.zeros(M, dtype=np.int64)  # for CrossEntropy backup
    
    new_anchor_ids = []  # IDs of true new anchors discovered this frame
    
    for i, truth_id in enumerate(labels if labels is not None else []):
        if truth_id == -1:
            # ====== CLUTTER ======
            match_labels[i] = actual_N  # "no match" category
            y_new[i] = 0.0  # CRITICAL: clutter is NOT new anchor
            
        elif truth_id in id_to_index:
            # ====== EXISTING ANCHOR ======
            idx = id_to_index[truth_id]
            if idx < N:  # Ensure index is valid
                y_match[i, idx] = 1.0
                match_labels[i] = idx
            y_new[i] = 0.0
            
        else:
            # ====== TRUE NEW ANCHOR ======
            match_labels[i] = actual_N  # "no match" category
            y_new[i] = 1.0  # TRUE new anchor!
            new_anchor_ids.append(truth_id)
    
    # ================================================================
    # Assemble sample - all dimensions should now be consistent
    # ================================================================
    sample = {
        'x_meas': torch.from_numpy(x_meas),
        'x_anchor': torch.from_numpy(x_anchor),
        'edge_index': torch.from_numpy(edge_index.astype(np.int64)),
        'edge_attr': torch.from_numpy(edge_attr.astype(np.float32)),
        # Labels
        'y_match': torch.from_numpy(y_match),
        'y_new': torch.from_numpy(y_new),
        'match_labels': torch.from_numpy(match_labels),
        # Metadata
        'metadata': {
            'step': step,
            'sensor': sensor,
            'num_measurements': M,
            'num_anchors': N if actual_N > 0 else 0,
            'existing_truth_ids': list(existing_truth_ids),
            'new_anchor_ids': new_anchor_ids
        }
    }
    
    return sample


def bp_based_mint_slam(data_va, cluttered_measurements, parameters, true_trajectory,
                       association_mode='bp', ground_truth_labels=None, 
                       gnn_model=None, collect_training_data=False):
    """
    Multi-mode SLAM algorithm supporting BP, Ground Truth, and GNN data association.

    Args:
        data_va: Virtual anchor data list
        cluttered_measurements: Measurements with clutter [num_steps][num_sensors]
        parameters: Algorithm parameters dictionary
        true_trajectory: True trajectory for error calculation, shape (2, num_steps)
        
        association_mode: Data association method
            - 'bp': Belief Propagation (default, for inference)
            - 'ground_truth': Use ground truth labels (for training data collection)
            - 'gnn': Use trained GNN model (for inference after training)
        
        ground_truth_labels: Required when association_mode='ground_truth'
            - Format: [step][sensor] = (M,) array of truth_anchor_ids (-1 for clutter)
        
        gnn_model: Required when association_mode='gnn'
            - Trained EdgeConditionedBipartiteGAT model
        
        collect_training_data: If True, collect and return GNN training samples

    Returns:
        estimated_trajectory: Estimated agent trajectory, shape (4, num_steps)
        estimated_anchors: Estimated anchor positions and existence probabilities
        posterior_particles_anchors_storage: Stored anchor particles for analysis
        num_estimated_anchors: Number of anchors per sensor per step
        
        If collect_training_data=True, also returns:
        training_samples: List of GNN training samples
    """
    # Validate parameters
    if association_mode == 'ground_truth' and ground_truth_labels is None:
        raise ValueError("ground_truth_labels required when association_mode='ground_truth'")
    if association_mode == 'gnn' and gnn_model is None:
        raise ValueError("gnn_model required when association_mode='gnn'")
    
    print(f"SLAM running with association_mode='{association_mode}'")
    if collect_training_data:
        print("Training data collection ENABLED")
    
    # Training data storage
    training_samples = [] if collect_training_data else None
    
    # For ground_truth mode: track discovered anchor truth_ids per sensor
    if association_mode == 'ground_truth':
        discovered_truth_ids = [[] for _ in range(len(cluttered_measurements[0]))]
    
    # 获取测量时间步数和传感器数量
    num_steps = len(cluttered_measurements)
    num_sensors = len(cluttered_measurements[0])

    # 限制最大时间步数
    num_steps = min(num_steps, parameters['maxSteps'])

    # 读取参数
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

    # 预分配存储空间
    estimated_trajectory = np.zeros((4, num_steps))  # 状态空间4维：x,y,vx,vy
    num_estimated_anchors = np.zeros((num_sensors, num_steps), dtype=int)
    storing_idx = list(range(29, num_steps, 30))  # 每30步存储一次锚点粒子状态（Python从0开始）
    posterior_particles_anchors_storage = [None] * len(storing_idx)

    # 初始化移动体粒子
    if known_track:
        # 已知轨迹时，所有粒子初始化为真实轨迹状态，速度为0
        posterior_particles_agent = np.tile(
            np.vstack([true_trajectory[:2, 0:1], np.zeros((2, 1))]),
            (1, num_particles)
        )
    else:
        # 未知轨迹时，均匀采样位置粒子，速度粒子随机采样
        posterior_particles_agent = np.zeros((4, num_particles))
        posterior_particles_agent[0:2, :] = draw_samples_uniformly_circ(
            prior_mean[0:2], parameters['UniformRadius_pos'], num_particles
        )
        posterior_particles_agent[2:4, :] = (
            np.tile(prior_mean[2:4].reshape(-1, 1), (1, num_particles)) +
            2 * parameters['UniformRadius_vel'] * np.random.rand(2, num_particles) -
            parameters['UniformRadius_vel']
        )

    # 记录初始状态估计（粒子均值）
    estimated_trajectory[:, 0] = np.mean(posterior_particles_agent, axis=1)

    # 初始化锚点状态（位置粒子和权重）
    estimated_anchors, posterior_particles_anchors = init_anchors(
        parameters, data_va, num_steps, num_sensors
    )
    for sensor in range(num_sensors):
        num_estimated_anchors[sensor, 0] = len(estimated_anchors[sensor][0])

    # 主循环，遍历每个时间步
    for step in range(1, num_steps):  # Python从0开始，所以从1开始
        start_time = time.time()

        # 预测移动体状态
        if known_track:
            # 已知轨迹，粒子直接用真实轨迹
            predicted_particles_agent = np.tile(
                np.vstack([true_trajectory[:2, step:step+1], np.zeros((2, 1))]),
                (1, num_particles)
            )
        else:
            # 未知轨迹时，基于动力学模型预测粒子状态
            predicted_particles_agent = perform_prediction(posterior_particles_agent, parameters)

        # 初始化存储每个粒子每个传感器权重的矩阵
        weights_sensors = np.full((num_particles, num_sensors), np.nan)

        # 对每个传感器进行锚点估计和数据关联更新
        for sensor in range(num_sensors):
            # 继承上一时刻估计的锚点状态（深拷贝以避免修改历史数据）
            estimated_anchors[sensor][step] = copy.deepcopy(estimated_anchors[sensor][step - 1])
            measurements = cluttered_measurements[step][sensor]  # 当前时刻传感器测量

            if measurements is None or measurements.size == 0:
                num_measurements = 0
            else:
                num_measurements = measurements.shape[1]

            # 预测未检测锚点强度（存活概率衰减+新锚点出生强度）
            undetected_anchors_intensity[sensor] = (
                undetected_anchors_intensity[sensor] * survival_probability + birth_intensity
            )

            # 预测"遗留"锚点粒子状态及权重
            predicted_particles_anchors, weights_anchor = predict_anchors(
                posterior_particles_anchors[sensor], parameters
            )

            # 针对每个测量生成新锚点粒子（新特征）
            new_particles_anchors, new_input_bp = generate_new_anchors(
                measurements, undetected_anchors_intensity[sensor],
                predicted_particles_agent, parameters
            )

            # 预测由锚点到移动体的测量值及其不确定度
            predicted_measurements, predicted_uncertainties, predicted_range = predict_measurements(
                predicted_particles_agent, predicted_particles_anchors, weights_anchor
            )

            # ================================================================
            # DATA ASSOCIATION - Multi-mode support
            # ================================================================
            num_anchors = predicted_particles_anchors.shape[2] if predicted_particles_anchors.size > 0 else 0
            
            if association_mode == 'bp':
                # Original BP-based data association
                (association_probabilities, association_probabilities_new,
                 message_lhf_ratios, messages_new) = calculate_association_probabilities_ga(
                    measurements, predicted_measurements, predicted_uncertainties,
                    weights_anchor, new_input_bp, parameters
                )
                
            elif association_mode == 'ground_truth':
                # Ground truth association (cheat mode for training data collection)
                labels = ground_truth_labels[step][sensor]
                
                # Build mapping from truth_id to anchor index
                id_to_idx = {tid: idx for idx, tid in enumerate(discovered_truth_ids[sensor])}
                
                # Initialize outputs
                if num_measurements > 0 and num_anchors > 0:
                    message_lhf_ratios = np.zeros((num_measurements, num_anchors))
                    for i, truth_id in enumerate(labels):
                        if truth_id >= 0 and truth_id in id_to_idx:
                            # Match to existing anchor
                            idx = id_to_idx[truth_id]
                            message_lhf_ratios[i, idx] = 1.0
                else:
                    message_lhf_ratios = np.zeros((max(num_measurements, 1), max(num_anchors, 1)))
                
                # For new anchors: discover new truth_ids
                messages_new = np.zeros(num_measurements) if num_measurements > 0 else np.array([])
                new_anchor_truth_ids = []
                for i, truth_id in enumerate(labels):
                    if truth_id >= 0 and truth_id not in id_to_idx:
                        # This is a NEW anchor - high message
                        messages_new[i] = 10.0  # High value to create new anchor
                        new_anchor_truth_ids.append(truth_id)
                    elif truth_id == -1:
                        # Clutter - low message (don't create new anchor)
                        messages_new[i] = 0.01
                
                association_probabilities = message_lhf_ratios
                association_probabilities_new = messages_new
                
            elif association_mode == 'gnn':
                # ================================================================
                # GNN-based data association
                # ================================================================
                if num_measurements == 0:
                    message_lhf_ratios = np.zeros((1, max(num_anchors, 1)))
                    messages_new = np.array([])
                else:
                    # Prepare input tensors
                    device = next(gnn_model.parameters()).device
                    
                    # Measurement features: (M, 2) [distance, variance]
                    x_meas = torch.zeros((num_measurements, 2), dtype=torch.float32, device=device)
                    x_meas[:, 0] = torch.from_numpy(measurements[0, :])
                    x_meas[:, 1] = torch.from_numpy(measurements[1, :])
                    
                    # Anchor features: (N, 3) [predicted_dist, variance, existence]
                    if num_anchors > 0:
                        x_anchor = torch.zeros((num_anchors, 3), dtype=torch.float32, device=device)
                        x_anchor[:, 0] = torch.from_numpy(predicted_measurements)
                        x_anchor[:, 1] = torch.from_numpy(predicted_uncertainties)
                        anchor_exist = np.array([posterior_particles_anchors[sensor][a]['posteriorExistence'] 
                                                  for a in range(num_anchors)])
                        x_anchor[:, 2] = torch.from_numpy(anchor_exist)
                        N = num_anchors
                    else:
                        # Dummy anchor for graph structure
                        x_anchor = torch.zeros((1, 3), dtype=torch.float32, device=device)
                        x_anchor[0, :] = torch.tensor([0.0, 1.0, 0.0])
                        N = 1
                    
                    M = num_measurements
                    
                    # Build edge index: (2, M*N)
                    meas_idx = torch.arange(M, device=device).repeat_interleave(N)
                    anchor_idx = torch.arange(N, device=device).repeat(M)
                    edge_index = torch.stack([meas_idx, anchor_idx], dim=0)
                    
                    # Edge features: |z - z_hat|
                    z = x_meas[:, 0]  # (M,)
                    z_hat = x_anchor[:, 0]  # (N,)
                    residuals = torch.abs(z[:, None] - z_hat[None, :])  # (M, N)
                    edge_attr = residuals.flatten()[:, None]  # (M*N, 1)
                    
                    # Forward pass (use sigmoid for independent probabilities)
                    gnn_model.eval()
                    with torch.no_grad():
                        match_probs, new_anchor_probs, _ = gnn_model(x_meas, x_anchor, edge_attr, edge_index, use_sigmoid=True)
                    
                    # Convert to numpy
                    match_probs_np = match_probs.cpu().numpy()  # (M, N)
                    new_probs_np = new_anchor_probs.cpu().numpy()  # (M,)
                    
                    # Use match probabilities as message ratios
                    if num_anchors > 0:
                        message_lhf_ratios = match_probs_np
                    else:
                        message_lhf_ratios = np.zeros((num_measurements, 1))
                    
                    # Use new anchor probabilities as messages_new
                    # Scale to match BP message range
                    messages_new = new_probs_np * 10.0  # Scale up for anchor generation
                
                association_probabilities = message_lhf_ratios
                association_probabilities_new = messages_new
            
            # ================================================================
            # Collect training data if enabled
            # ================================================================
            if collect_training_data and num_measurements > 0:
                sample = _create_training_sample(
                    measurements=measurements,
                    labels=ground_truth_labels[step][sensor] if ground_truth_labels else None,
                    predicted_measurements=predicted_measurements,
                    predicted_uncertainties=predicted_uncertainties,
                    anchor_existence=np.array([posterior_particles_anchors[sensor][a]['posteriorExistence'] 
                                               for a in range(num_anchors)]) if num_anchors > 0 else np.array([]),
                    existing_truth_ids=discovered_truth_ids[sensor] if association_mode == 'ground_truth' else [],
                    step=step,
                    sensor=sensor
                )
                if sample is not None:
                    training_samples.append(sample)
            
            # ================================================================
            # Particle weight update
            # ================================================================
            weights = np.zeros((num_particles, num_anchors)) if num_anchors > 0 else np.zeros((num_particles, 1))

            # 向量化优化：一次性计算所有锚点和测量的权重
            if num_measurements > 0 and num_anchors > 0:
                # 初始化权重为未检测概率
                weights[:, :] = (1 - detection_probability)

                # 预计算所有测量的方差和因子
                measurement_variances = measurements[1, :]  # shape: (num_measurements,)
                factors = (1 / np.sqrt(2 * np.pi * measurement_variances) *
                          detection_probability / clutter_intensity)  # shape: (num_measurements,)

                # 计算距离差: (num_particles, num_anchors, num_measurements)
                # predicted_range: (num_particles, num_anchors)
                # measurements[0, :]: (num_measurements,)
                range_diff = measurements[0, :][np.newaxis, np.newaxis, :] - predicted_range[:, :, np.newaxis]

                # 计算所有权重贡献: (num_particles, num_anchors, num_measurements)
                weight_contributions = (
                    factors[np.newaxis, np.newaxis, :] *
                    message_lhf_ratios.T[np.newaxis, :, :] *  # message_lhf_ratios是(M,N)，转置后是(N,M)
                    np.exp(-0.5 / measurement_variances[np.newaxis, np.newaxis, :] * range_diff**2)
                )

                # 对测量维度求和，得到每个锚点的总权重
                weights += np.sum(weight_contributions, axis=2)
            else:
                # 没有测量或没有锚点时，所有权重为未检测概率
                weights[:, :] = (1 - detection_probability)

            # 对每个锚点进行后续处理（这部分仍需循环，因为涉及重采样等操作）
            for anchor in range(num_anchors):

                # 计算该锚点预测存在概率
                predicted_existence = np.sum(weights_anchor[:, anchor])

                # 计算锚点存在的后验概率
                alive_update = np.sum(predicted_existence * (1 / num_particles) * weights[:, anchor])
                dead_update = 1 - predicted_existence
                posterior_particles_anchors[sensor][anchor]['posteriorExistence'] = (
                    alive_update / (alive_update + dead_update)
                )

                # 重采样粒子，依据权重调整粒子集合
                weight_sum = np.sum(weights[:, anchor])
                if weight_sum > 0:
                    idx_resampling = resample_systematic(
                        weights[:, anchor] / weight_sum, num_particles
                    )
                else:
                    idx_resampling = np.arange(num_particles)

                posterior_particles_anchors[sensor][anchor]['x'] = (
                    predicted_particles_anchors[:, idx_resampling, anchor]
                )
                posterior_particles_anchors[sensor][anchor]['w'] = (
                    posterior_particles_anchors[sensor][anchor]['posteriorExistence'] /
                    num_particles * np.ones(num_particles)
                )

                # 计算锚点位置的均值估计和存在概率
                estimated_anchors[sensor][step][anchor]['x'] = np.mean(
                    posterior_particles_anchors[sensor][anchor]['x'], axis=1
                )
                estimated_anchors[sensor][step][anchor]['posteriorExistence'] = (
                    posterior_particles_anchors[sensor][anchor]['posteriorExistence']
                )

                # 计算归一化权重的对数，防止数值溢出
                weights[:, anchor] = predicted_existence * weights[:, anchor] + dead_update
                weights[:, anchor] = np.log(weights[:, anchor])
                weights[:, anchor] = weights[:, anchor] - np.max(weights[:, anchor])

            # 更新锚点数量
            num_estimated_anchors[sensor, step] = len(estimated_anchors[sensor][step])

            # 汇总所有锚点权重，为移动体粒子加权
            weights_sensors[:, sensor] = np.sum(weights, axis=1)
            weights_sensors[:, sensor] = weights_sensors[:, sensor] - np.max(weights_sensors[:, sensor])

            # 更新未检测锚点强度，乘以未检测概率
            undetected_anchors_intensity[sensor] = (
                undetected_anchors_intensity[sensor] * (1 - detection_probability)
            )

            # 更新新锚点的后验存在概率和粒子集
            for measurement in range(num_measurements):
                new_anchor_idx = num_anchors + measurement
                constant = new_particles_anchors[measurement]['constant']
                posterior_existence = (
                    messages_new[measurement] * constant /
                    (messages_new[measurement] * constant + 1)
                )

                # 扩展列表以容纳新锚点
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
                    
                    # Ground truth mode: track discovered truth IDs
                    if association_mode == 'ground_truth':
                        labels = ground_truth_labels[step][sensor]
                        if measurement < len(labels):
                            truth_id = labels[measurement]
                            if truth_id >= 0 and truth_id not in discovered_truth_ids[sensor]:
                                discovered_truth_ids[sensor].append(truth_id)
                                
                else:
                    posterior_particles_anchors[sensor][new_anchor_idx]['posteriorExistence'] = posterior_existence
                    posterior_particles_anchors[sensor][new_anchor_idx]['x'] = new_particles_anchors[measurement]['x']
                    posterior_particles_anchors[sensor][new_anchor_idx]['w'] = posterior_existence / num_particles
                    estimated_anchors[sensor][step][new_anchor_idx]['x'] = np.mean(
                        new_particles_anchors[measurement]['x'], axis=1
                    )
                    estimated_anchors[sensor][step][new_anchor_idx]['posteriorExistence'] = posterior_existence
                    estimated_anchors[sensor][step][new_anchor_idx]['generatedAt'] = step

            # 删除存在概率低于阈值的不可靠锚点，控制复杂度
            estimated_anchors[sensor][step], posterior_particles_anchors[sensor], reliable_indices = delete_unreliable_va(
                estimated_anchors[sensor][step], posterior_particles_anchors[sensor],
                unreliability_threshold
            )
            
            # Ground truth mode: sync discovered_truth_ids with anchor deletion
            if association_mode == 'ground_truth' and len(reliable_indices) < len(discovered_truth_ids[sensor]):
                discovered_truth_ids[sensor] = [discovered_truth_ids[sensor][i] for i in reliable_indices 
                                                 if i < len(discovered_truth_ids[sensor])]
            
            num_estimated_anchors[sensor, step] = len(estimated_anchors[sensor][step])

        # 汇总所有传感器权重，归一化移动体粒子权重
        weights_sensors = np.sum(weights_sensors, axis=1)
        weights_sensors = weights_sensors - np.max(weights_sensors)
        weights_sensors = np.exp(weights_sensors)
        weights_sensors = weights_sensors / np.sum(weights_sensors)

        # 保存部分关键时间步的锚点粒子状态，用于分析
        if step in storing_idx:
            idx = storing_idx.index(step)
            posterior_particles_anchors_storage[idx] = [
                [anchor.copy() for anchor in sensor_anchors]
                for sensor_anchors in posterior_particles_anchors
            ]

        # 更新移动体估计轨迹
        if known_track:
            # 已知轨迹时，直接用预测粒子均值
            estimated_trajectory[:, step] = np.mean(predicted_particles_agent, axis=1)
            posterior_particles_agent = predicted_particles_agent
        else:
            # 未知轨迹时，基于权重重采样粒子，更新估计
            estimated_trajectory[:, step] = predicted_particles_agent @ weights_sensors
            posterior_particles_agent = predicted_particles_agent[
                :, resample_systematic(weights_sensors, num_particles)
            ]

        # 计算估计误差（距离误差）并打印输出
        exec_time_per_step[step] = time.time() - start_time
        error_agent = calc_distance(
            true_trajectory[0:2, step:step+1],
            estimated_trajectory[0:2, step:step+1]
        )
        # 如果error_agent是数组，提取标量值
        if isinstance(error_agent, np.ndarray):
            error_agent = error_agent.item() if error_agent.size == 1 else error_agent[0]

        print(f'Time instance: {step + 1}')  # +1 for MATLAB-style output
        # 自动打印所有传感器的锚点数量
        for sensor in range(num_sensors):
            print(f'Number of Anchors Sensor {sensor + 1}: {num_estimated_anchors[sensor, step]}')
        print(f'Position error agent: {error_agent:.6f}')
        print(f'Execution Time: {exec_time_per_step[step]:.4f}')
        print('---------------------------------------------------\n')

    # Return results
    results = (estimated_trajectory, estimated_anchors, posterior_particles_anchors_storage, num_estimated_anchors)
    
    if collect_training_data:
        return results + (training_samples,)
    else:
        return results
