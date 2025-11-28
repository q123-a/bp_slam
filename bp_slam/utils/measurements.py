"""
测量生成相关函数
Measurement generation functions
"""

import numpy as np


def generate_measurements(target_trajectory, data_va, parameters):
    """
    根据目标轨迹和虚拟锚点数据生成带噪声的测量值

    参数:
        target_trajectory: 目标（移动体）的轨迹，shape (2, num_steps) 或更高维
        data_va: 虚拟锚点数据列表，长度为传感器数量，每个包含锚点位置和可见性
        parameters: 参数字典，包含测量方差等信息

    返回:
        measurements_cell: 生成的测量数据，shape (num_steps, num_sensors)的列表
                          每个元素是 (2, num_visible_anchors) 的数组（距离和方差）
    """
    measurement_variance_range = parameters['measurementVariance']

    num_steps = target_trajectory.shape[1]
    num_sensors = len(data_va)

    # 初始化测量存储
    measurements_cell = [[None for _ in range(num_sensors)] for _ in range(num_steps)]

    # 遍历每个传感器
    for sensor in range(num_sensors):
        positions = data_va[sensor]['positions']  # 锚点位置 (2, num_anchors)
        visibility = data_va[sensor]['visibility']  # 可见性矩阵 (num_anchors, num_steps)

        # 遍历每个时间步
        for step in range(num_steps):
            k = 0  # 计数当前时刻可见锚点数
            num_anchors = positions.shape[1]
            measurements = np.zeros((2, num_anchors))

            # 遍历所有锚点
            for anchor in range(num_anchors):
                if visibility[anchor, step]:  # 该锚点在当前时刻可见
                    # 赋予测距方差（固定）
                    measurements[1, k] = measurement_variance_range
                    # 计算实际距离（欧氏距离） + 加入高斯噪声
                    distance = np.sqrt((positions[0, anchor] - target_trajectory[0, step])**2 +
                                     (positions[1, anchor] - target_trajectory[1, step])**2)
                    measurements[0, k] = distance + np.sqrt(measurements[1, k]) * np.random.randn()
                    # 更新测量方差为另一个参数（可认为是测量噪声的后验方差）
                    measurements[1, k] = parameters['measurementVarianceLHF']
                    k += 1

            # 只保留当前时刻可见锚点的测量数据
            measurements_cell[step][sensor] = measurements[:, :k]

    return measurements_cell


def generate_cluttered_measurements(true_measurements_cell, parameters):
    """
    生成带有杂波和漏检的测量数据（仿真环境）

    参数:
        true_measurements_cell: 真实测量数据，shape (num_steps, num_sensors)的列表
                               每个元素是 (2, num_anchors) 的数组（距离+方差）
        parameters: 参数字典，包括测量方差、检测概率、杂波均值、区域大小等

    返回:
        cluttered_measurements: 加入误报和漏检后的测量数据，shape同输入
    """
    # 读取参数
    measurement_variance_range = parameters['measurementVariance']
    detection_probability = parameters['detectionProbability']
    mean_number_of_clutter = parameters['meanNumberOfClutter']
    max_range = parameters['regionOfInterestSize']

    num_steps = len(true_measurements_cell)
    num_sensors = len(true_measurements_cell[0])

    # 初始化输出
    cluttered_measurements = [[None for _ in range(num_sensors)] for _ in range(num_steps)]

    # 遍历每个传感器和时间步
    for sensor in range(num_sensors):
        for step in range(num_steps):
            true_measurements = true_measurements_cell[step][sensor]

            if true_measurements is None or true_measurements.size == 0:
                num_anchors = 0
                detected_measurements = np.zeros((2, 0))
            else:
                num_anchors = true_measurements.shape[1]

                # 按检测概率随机决定哪些锚点被检测到（漏检处理）
                detection_indicator = (np.random.rand(num_anchors) < detection_probability)

                # 提取被检测到的测量
                detected_measurements = true_measurements[:, detection_indicator]

            # 生成误报（杂波）数量，符合泊松分布
            num_false_alarms = np.random.poisson(mean_number_of_clutter)

            # 生成误报测量
            false_alarms = np.zeros((2, num_false_alarms))
            if num_false_alarms > 0:
                # 误报距离均匀分布在0到maxRange
                false_alarms[0, :] = max_range * np.random.rand(num_false_alarms)
                # 误报测量方差为测距方差
                false_alarms[1, :] = measurement_variance_range

            # 将误报和真实检测测量拼接
            if detected_measurements.size > 0:
                cluttered_measurement = np.hstack([false_alarms, detected_measurements])
            else:
                cluttered_measurement = false_alarms

            # 随机打乱测量顺序，模拟实际测量的无序性
            if cluttered_measurement.shape[1] > 0:
                perm = np.random.permutation(cluttered_measurement.shape[1])
                cluttered_measurement = cluttered_measurement[:, perm]

            # 保存当前时间步传感器的测量
            cluttered_measurements[step][sensor] = cluttered_measurement

    return cluttered_measurements


def calculate_constants_uniform(predicted_particles_agent, new_measurements, parameters):
    """
    计算新锚点生成时的归一化常数

    参数:
        predicted_particles_agent: 预测的移动体粒子位置，shape (4, n_particles)，只用前2维位置
        new_measurements: 新测量值矩阵（距离及方差），shape (2, n_measurements)
        parameters: 参数字典，包含区域大小、上采样因子、粒子数等

    返回:
        constants: 每个测量对应的归一化常数向量，shape (n_measurements,)
    """
    # 定义感兴趣区域面积（假设是一个边长为2*regionOfInterestSize的正方形）
    region_of_interest = (2 * parameters['regionOfInterestSize'])**2

    # 上采样因子（提升粒子数以提高估计精度）
    up_sampling_factor = parameters['upSamplingFactor']

    # 计算总粒子数
    num_particles = parameters['numParticles'] * up_sampling_factor

    # 复制移动体粒子位置，使其与上采样粒子数匹配
    predicted_particles_agent = np.tile(predicted_particles_agent, up_sampling_factor)

    # 在感兴趣区域内均匀采样随机粒子（2维位置）
    particles = (2 * parameters['regionOfInterestSize'] * np.random.rand(2, num_particles) -
                parameters['regionOfInterestSize'])

    # 均匀分布权重常数（区域面积的倒数）
    constant_weight = 1 / region_of_interest

    # 测量数目
    num_measurements = new_measurements.shape[1]

    # 初始化常数向量
    constants = np.zeros(num_measurements)

    # 计算所有均匀采样点到移动体粒子预测位置的预测距离（欧氏距离）
    predicted_range = np.sqrt((particles[0, :] - predicted_particles_agent[0, :])**2 +
                             (particles[1, :] - predicted_particles_agent[1, :])**2)

    # 遍历每个测量，计算归一化常数
    for measurement in range(num_measurements):
        # 计算测量方差对应的高斯似然常数因子
        constant_likelihood = 1 / np.sqrt(2 * np.pi * new_measurements[1, measurement])

        # 计算该测量与预测距离的高斯似然概率，并对所有粒子求平均（Monte Carlo积分）
        likelihood = constant_likelihood * np.exp(
            (-0.5) * (new_measurements[0, measurement] - predicted_range)**2 /
            new_measurements[1, measurement]
        )
        constants[measurement] = np.sum(likelihood / num_particles)

    # 除以均匀分布权重，完成归一化
    constants = constants / constant_weight

    return constants
