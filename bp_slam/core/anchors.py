"""
锚点管理相关函数
Anchor management functions
"""

import numpy as np
from ..utils.sampling import sample_from_likelihood
from ..utils.measurements import calculate_constants_uniform


def init_anchors(parameters, data_va, num_steps, num_sensors):
    """
    初始化物理锚点和几何锚点的粒子集及状态估计

    参数:
        parameters: 参数字典，包含粒子数、先验协方差、已知锚点索引等
        data_va: 虚拟锚点数据列表，包含锚点位置
        num_steps: 时间步总数
        num_sensors: 传感器数量

    返回:
        estimated_anchors: 估计的锚点状态，列表 [num_sensors][num_steps]
        posterior_particles_anchors: 每个传感器锚点的粒子集合列表
    """
    num_particles = parameters['numParticles']

    # 初始化存储结构
    posterior_particles_anchors = [None] * num_sensors
    estimated_anchors = [[None for _ in range(num_steps)] for _ in range(num_sensors)]

    # 初始化锚点cell结构，分配空间
    for sensor in range(num_sensors):
        # 获取该传感器已知锚点的初始位置
        anchor_positions = data_va[sensor]['positions'][:, parameters['priorKnownAnchors'][sensor]]
        num_anchors = anchor_positions.shape[1]
        posterior_particles_anchors[sensor] = [None] * num_anchors
        estimated_anchors[sensor][0] = [None] * num_anchors

    prior_covariance_anchor = parameters['priorCovarianceAnchor']

    # 遍历每个传感器和锚点，初始化粒子集和估计
    for sensor in range(num_sensors):
        anchor_positions = data_va[sensor]['positions'][:, parameters['priorKnownAnchors'][sensor]].copy()
        num_anchors = anchor_positions.shape[1]

        for anchor in range(num_anchors):
            # 初始化粒子结构
            posterior_particles_anchors[sensor][anchor] = {
                'x': np.zeros((2, num_particles)),
                'w': np.zeros(num_particles),
                'posteriorExistence': 1.0
            }

            # 权重均匀分配
            posterior_particles_anchors[sensor][anchor]['w'][:] = (
                posterior_particles_anchors[sensor][anchor]['posteriorExistence'] / num_particles
            )

            if anchor == 0:  # Python索引从0开始，第一个锚点
                # 第一个锚点为物理锚点（PA）
                # 根据先验均值和协方差采样粒子位置，二维正态分布
                posterior_particles_anchors[sensor][anchor]['x'] = np.random.multivariate_normal(
                    anchor_positions[:, anchor], prior_covariance_anchor, num_particles
                ).T
                # 估计位置为先验均值
                estimated_anchors[sensor][0][anchor] = {
                    'x': anchor_positions[:, anchor].copy(),
                    'posteriorExistence': posterior_particles_anchors[sensor][anchor]['posteriorExistence'],
                    'generatedAt': 0  # Python从0开始
                }
            else:
                # 几何锚点（虚拟锚点VA）
                # 先对先验均值采样一次扰动，增加随机性
                anchor_positions[:, anchor] = np.random.multivariate_normal(
                    anchor_positions[:, anchor], prior_covariance_anchor, 1
                ).T.flatten()
                # 再基于扰动均值采样粒子
                posterior_particles_anchors[sensor][anchor]['x'] = np.random.multivariate_normal(
                    anchor_positions[:, anchor], prior_covariance_anchor, num_particles
                ).T
                # 估计位置为扰动后的均值
                estimated_anchors[sensor][0][anchor] = {
                    'x': anchor_positions[:, anchor].copy(),
                    'posteriorExistence': posterior_particles_anchors[sensor][anchor]['posteriorExistence'],
                    'generatedAt': 0
                }

    return estimated_anchors, posterior_particles_anchors


def predict_anchors(posterior_particles_anchors, parameters):
    """
    对锚点粒子进行预测，加入随机漂移并考虑存活概率

    参数:
        posterior_particles_anchors: 当前锚点粒子集合，列表，每个元素包含字典
                                    'x': (2, num_particles) 锚点粒子位置
                                    'w': (num_particles,) 权重向量
        parameters: 参数字典，包含粒子数、锚点漂移噪声方差、存活概率等

    返回:
        predicted_particles_anchors: 预测的锚点粒子位置，shape (2, num_particles, num_anchors)
        weights_anchor: 预测锚点粒子权重，shape (num_particles, num_anchors)
    """
    num_particles = parameters['numParticles']
    anchor_noise_variance = parameters['anchorRegularNoiseVariance']
    survival_probability = parameters['survivalProbability']

    num_anchors = len(posterior_particles_anchors)

    # 预分配预测粒子位置数组和权重矩阵
    predicted_particles_anchors = np.zeros((2, num_particles, num_anchors))
    weights_anchor = np.zeros((num_particles, num_anchors))

    for anchor in range(num_anchors):
        # 根据存活概率调整粒子权重
        weights_anchor[:, anchor] = survival_probability * posterior_particles_anchors[anchor]['w']

        # 生成二维高斯噪声，模拟锚点位置的随机漂移
        anchor_noise = np.sqrt(anchor_noise_variance) * np.random.randn(2, num_particles)

        # 预测锚点粒子位置为先验位置加上漂移噪声
        predicted_particles_anchors[:, :, anchor] = (
            posterior_particles_anchors[anchor]['x'] + anchor_noise
        )

    return predicted_particles_anchors, weights_anchor


def predict_measurements(predicted_particles, anchor_positions, weights_anchor):
    """
    根据预测的移动体粒子和锚点位置粒子，计算距离测量的预测均值和不确定度

    参数:
        predicted_particles: 预测的移动体粒子状态，shape (4, num_particles)（只用前2维位置）
        anchor_positions: 锚点粒子位置，shape (2, num_particles, num_anchors)
        weights_anchor: 对应锚点粒子的权重矩阵，shape (num_particles, num_anchors)

    返回:
        predicted_means: 每个锚点预测测距的加权平均值，shape (num_anchors,)
        predicted_uncertainties: 每个锚点测距的加权方差，shape (num_anchors,)
        predicted_range: 所有粒子对应的距离矩阵，shape (num_particles, num_anchors)
    """
    num_particles = anchor_positions.shape[1]
    num_anchors = anchor_positions.shape[2]

    # 预分配输出变量
    predicted_means = np.zeros(num_anchors)
    predicted_uncertainties = np.zeros(num_anchors)
    predicted_range = np.zeros((num_particles, num_anchors))

    for anchor in range(num_anchors):
        # 计算每个粒子对应的移动体位置与锚点位置间的欧氏距离
        predicted_range[:, anchor] = np.sqrt(
            (predicted_particles[0, :] - anchor_positions[0, :, anchor])**2 +
            (predicted_particles[1, :] - anchor_positions[1, :, anchor])**2
        )

        # 计算加权平均距离（预测测量均值）
        weight_sum = np.sum(weights_anchor[:, anchor])
        if weight_sum > 0:
            predicted_means[anchor] = (
                np.dot(predicted_range[:, anchor], weights_anchor[:, anchor]) / weight_sum
            )

            # 计算加权方差（预测测量不确定度）
            diff = predicted_range[:, anchor] - predicted_means[anchor]
            predicted_uncertainties[anchor] = (
                np.dot(diff * diff, weights_anchor[:, anchor]) / weight_sum
            )

    return predicted_means, predicted_uncertainties, predicted_range


def generate_new_anchors(new_measurements, undetected_targets_intensity,
                        predicted_particles_agent, parameters):
    """
    根据新测量生成新锚点粒子及对应的信念传播输入消息

    参数:
        new_measurements: 新测量矩阵，shape (2, n_measurements)（距离+测量方差）
        undetected_targets_intensity: 未检测锚点强度（先验出生率）
        predicted_particles_agent: 预测的移动体粒子状态，shape (4, num_particles)
        parameters: 参数字典，含杂波强度、检测概率、粒子数等

    返回:
        new_particles_anchors: 新锚点粒子列表，每个包含 'x', 'w', 'constant'
        input_bp: 对应新锚点的输入消息，用于信念传播数据关联
    """
    clutter_intensity = parameters['clutterIntensity']
    num_particles = parameters['numParticles']
    detection_probability = parameters['detectionProbability']

    if new_measurements is None or new_measurements.size == 0:
        num_measurements = 0
    else:
        num_measurements = new_measurements.shape[1]

    input_bp = np.zeros(num_measurements)
    new_particles_anchors = []

    if num_measurements == 0:
        return new_particles_anchors, input_bp

    # 计算每个新测量的归一化常数，用于权重计算（基于均匀分布蒙特卡洛积分）
    constants = calculate_constants_uniform(predicted_particles_agent, new_measurements, parameters)

    for measurement in range(num_measurements):
        # 计算新锚点的信念传播输入消息
        # BP风格（归一化）：更稳定，数值表现更好
        input_bp[measurement] = 1 + (
            (constants[measurement] * undetected_targets_intensity * detection_probability) /
            clutter_intensity
        )

        # 提取测量距离及方差
        measurement_to_anchor = new_measurements[0, measurement]
        measurement_variance = new_measurements[1, measurement]

        # 根据测量似然采样新锚点粒子状态（只采样2维位置）
        sampled_positions = sample_from_likelihood(
            measurement_to_anchor, measurement_variance,
            predicted_particles_agent, num_particles
        )

        # 创建新锚点粒子结构
        new_anchor = {
            'x': sampled_positions,  # (2, num_particles)
            'w': np.ones(num_particles) / num_particles,
            'constant': (constants[measurement] * undetected_targets_intensity *
                        detection_probability / clutter_intensity)
        }

        new_particles_anchors.append(new_anchor)

    return new_particles_anchors, input_bp


def delete_unreliable_va(estimated_anchors, posterior_particles_anchors,
                        unreliability_threshold):
    """
    删除不可靠的虚拟锚点（存在概率低于阈值）

    参数:
        estimated_anchors: 估计的锚点状态列表
        posterior_particles_anchors: 锚点粒子集合列表
        unreliability_threshold: 锚点存在概率阈值，低于则删除

    返回:
        estimated_anchors: 删除不可靠锚点后的估计
        posterior_particles_anchors: 删除不可靠锚点后的粒子集合
    """
    # 找出存在概率高于阈值的锚点索引
    reliable_indices = []
    for i, anchor in enumerate(posterior_particles_anchors):
        if anchor['posteriorExistence'] >= unreliability_threshold:
            reliable_indices.append(i)

    # 只保留可靠的锚点
    estimated_anchors = [estimated_anchors[i] for i in reliable_indices]
    posterior_particles_anchors = [posterior_particles_anchors[i] for i in reliable_indices]

    return estimated_anchors, posterior_particles_anchors
