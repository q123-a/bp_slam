"""
基于信念传播的多路径SLAM算法核心函数
BP-based Multipath-assisted SLAM core algorithm

Author: Florian Meyer, Erik Leitinger, 20/05/17
Converted to Python: 2025
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


def bp_based_mint_slam(data_va, cluttered_measurements, parameters, true_trajectory):
    """
    基于信念传播的多路径SLAM算法核心函数

    参数:
        data_va: 虚拟锚点数据列表
        cluttered_measurements: 带误报的测量数据（距离+方差），列表[num_steps][num_sensors]
        parameters: 算法参数字典
        true_trajectory: 真实轨迹，用于误差计算（已知轨迹模式），shape (n_dims, num_steps)

    返回:
        estimated_trajectory: 估计的移动体状态轨迹（位置+速度），shape (4, num_steps)
        estimated_anchors: 估计的锚点位置和存在概率
        posterior_particles_anchors_storage: 存储部分时刻锚点粒子用于分析
        num_estimated_anchors: 每时刻估计的锚点数量，shape (num_sensors, num_steps)
    """
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

            # 计算测量与锚点的数据关联概率
            (association_probabilities, association_probabilities_new,
             message_lhf_ratios, messages_new) = calculate_association_probabilities_ga(
                measurements, predicted_measurements, predicted_uncertainties,
                weights_anchor, new_input_bp, parameters
            )

            # 对每个锚点计算粒子权重，结合检测概率和测量似然
            num_anchors = predicted_particles_anchors.shape[2]
            weights = np.zeros((num_particles, num_anchors))

            # 向量化优化：一次性计算所有锚点和测量的权重
            if num_measurements > 0:
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
                # 没有测量时，所有权重为未检测概率
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
            estimated_anchors[sensor][step], posterior_particles_anchors[sensor] = delete_unreliable_va(
                estimated_anchors[sensor][step], posterior_particles_anchors[sensor],
                unreliability_threshold
            )
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

    return estimated_trajectory, estimated_anchors, posterior_particles_anchors_storage, num_estimated_anchors
