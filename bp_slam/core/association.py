"""
数据关联相关函数
Data association functions
"""

import numpy as np
from ..utils.belief_propagation import get_input_bp


def data_association_bp(legacy, new, check_convergence, threshold, num_iterations):
    """
    使用信念传播（BP）算法计算数据关联概率

    参数:
        legacy: 现有锚点的输入消息矩阵，shape (m+1, n)
                第一行为锚点未检测概率，后m行为测量与锚点的似然消息
        new: 新锚点的输入消息向量，shape (m,) 或标量，对应每个测量
        check_convergence: 迭代次数间隔，用于判断收敛
        threshold: 收敛阈值，消息变化低于该值认为收敛
        num_iterations: 最大迭代次数

    返回:
        assoc_prob_existing: 现有锚点与测量的关联概率矩阵，shape (m+1, n)
        assoc_prob_new: 新锚点与测量的关联概率向量，shape (m,)
        message_lhf_ratios: 现有锚点对应的消息比率矩阵，shape (m, n)
        message_lhf_ratios_new: 新锚点对应的消息比率向量，shape (m,)
    """
    m, n = legacy.shape
    m = m - 1  # 测量数 M （去掉未检测行）

    assoc_prob_new = np.ones(m)
    assoc_prob_existing = np.ones((m + 1, n))
    message_lhf_ratios = np.ones((m, n))
    message_lhf_ratios_new = np.ones(m)

    # 如果无测量或无锚点，直接返回默认值
    if n == 0 or m == 0:
        return assoc_prob_existing, assoc_prob_new, message_lhf_ratios, message_lhf_ratios_new

    # 如果新锚点输入消息为空或标量，转换为向量
    if np.isscalar(new):
        new = np.ones(m) * new
    elif new.size == 0:
        new = np.ones(m)

    # 初始化消息矩阵muba (m×n)，测量到锚点的消息
    muba = np.ones((m, n))

    # BP迭代循环（优化版：使用广播代替tile，减少收敛检查频率）
    for iteration in range(int(num_iterations)):
        muba_old = muba.copy()  # 保存上次消息用于收敛判断

        # 计算每个锚点的消息乘积（测量消息 * 先验似然）
        prodfact = muba * legacy[1:, :]
        sumprod = legacy[0, :] + np.sum(prodfact, axis=0)  # 所有测量和未检测概率之和

        # 归一化因子，使用广播代替tile（更快）
        normalization = sumprod[np.newaxis, :] - prodfact
        normalization[normalization == 0] = np.finfo(float).eps

        # 计算锚点到测量的消息更新
        muab = legacy[1:, :] / normalization

        # 计算测量总和消息（包括新锚点消息）
        summuab = new + np.sum(muab, axis=1)
        # 使用广播代替tile（更快）
        normalization = summuab[:, np.newaxis] - muab
        normalization[normalization == 0] = np.finfo(float).eps

        # 更新测量到锚点的消息
        muba = 1.0 / normalization

        # 每隔check_convergence步判断是否收敛
        if (iteration + 1) % check_convergence == 0:
            # 计算消息变化的最大对数比，判断是否小于阈值
            with np.errstate(divide='ignore', invalid='ignore'):
                distance = np.max(np.abs(np.log(muba / muba_old)))
            if distance < threshold:
                break

    # 计算关联概率，结合输入消息和更新后的消息
    assoc_prob_existing[0, :] = legacy[0, :]  # 未检测概率保持不变
    assoc_prob_existing[1:, :] = legacy[1:, :] * muba  # 关联测量概率更新

    # 对每个锚点列归一化概率和为1
    for target in range(n):
        col_sum = np.sum(assoc_prob_existing[:, target])
        if col_sum > 0:
            assoc_prob_existing[:, target] = assoc_prob_existing[:, target] / col_sum

    # 消息比率输出
    message_lhf_ratios = muba
    assoc_prob_new = new / summuab  # 新锚点关联概率计算

    # 新锚点消息比率计算（归一化）
    message_lhf_ratios_new = np.hstack([np.ones((m, 1)), muab])  # 连接未检测与检测消息
    row_sums = np.sum(message_lhf_ratios_new, axis=1, keepdims=True)
    message_lhf_ratios_new = message_lhf_ratios_new / row_sums
    message_lhf_ratios_new = message_lhf_ratios_new[:, 0]  # 取未检测消息部分作为输出

    return assoc_prob_existing, assoc_prob_new, message_lhf_ratios, message_lhf_ratios_new


def calculate_association_probabilities_ga(measurements, predicted_measurements,
                                          predicted_uncertainties, weights_anchor,
                                          new_input_bp, parameters):
    """
    计算测量与锚点之间的关联概率，采用高斯近似 (GA)

    参数:
        measurements: 当前时刻的测量值矩阵，shape (2, m)，第一行为测距，第二行为测距方差
        predicted_measurements: 预测的锚点对应的测距，shape (n_anchors,)
        predicted_uncertainties: 预测测量的不确定度（方差），shape (n_anchors,)
        weights_anchor: 锚点粒子的权重，shape (num_particles, n_anchors)
        new_input_bp: 新锚点的输入消息，用于处理新锚点关联
        parameters: 参数字典，包含检测概率Pd、杂波强度等

    返回:
        association_probabilities: 现有锚点与测量的关联概率矩阵
        association_probabilities_new: 新锚点与测量的关联概率矩阵
        message_legacy: 发送给锚点的消息（旧锚点）
        messages_new: 发送给新锚点的消息
    """
    # 读取参数
    detection_probability = parameters['detectionProbability']
    clutter_intensity = parameters['clutterIntensity']

    if measurements is None or measurements.size == 0:
        num_measurements = 0
    else:
        num_measurements = measurements.shape[1]

    num_anchors = len(predicted_measurements)

    # 计算每个锚点的存在概率（所有粒子权重之和）
    predicted_existence = np.sum(weights_anchor, axis=0)

    # 初始化输入消息矩阵 input_bp (m+1)×n
    # 第一行对应锚点未被检测（未关联任何测量）
    input_bp = np.zeros((num_measurements + 1, num_anchors))
    input_bp[0, :] = (1 - detection_probability)  # 未检测概率

    # 遍历所有锚点
    for anchor in range(num_anchors):
        # 遍历所有测量
        for measurement in range(num_measurements):
            # 预测测量和实际测量的联合方差（不确定度相加）
            predicted_uncertainty_tmp = (predicted_uncertainties[anchor] +
                                        measurements[1, measurement])

            # 计算高斯似然概率密度（归一化后）
            # BP风格（归一化了杂波强度，效果更稳定）
            factor = (1 / np.sqrt(2 * np.pi * predicted_uncertainty_tmp) *
                     detection_probability / clutter_intensity)

            # 根据距离差计算似然概率，存入消息矩阵
            distance_diff = measurements[0, measurement] - predicted_measurements[anchor]
            input_bp[measurement + 1, anchor] = factor * np.exp(
                -1 / (2 * predicted_uncertainty_tmp) * distance_diff**2
            )

        # 结合锚点存在概率，调用辅助函数get_input_bp，计算该锚点的最终输入消息
        input_bp[:, anchor] = get_input_bp(predicted_existence[anchor], input_bp[:, anchor])

    # 调用信念传播函数，进行迭代推断，得到关联概率和消息
    # 参数：最大迭代次数30，收敛阈值10^-6
    (association_probabilities, association_probabilities_new,
     message_legacy, messages_new) = data_association_bp(input_bp, new_input_bp, 30, 1e-6, 1e6)

    return (association_probabilities, association_probabilities_new,
            message_legacy, messages_new)
