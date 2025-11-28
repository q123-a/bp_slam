"""
运动模型相关函数
Motion model related functions
"""

import numpy as np


def get_transition_matrices(scan_time):
    """
    生成状态转移矩阵和过程噪声矩阵

    参数:
        scan_time: 两次状态更新之间的时间间隔（采样时间）

    返回:
        A: 状态转移矩阵 (4, 4)，用于预测下一状态
        W: 过程噪声输入矩阵 (4, 2)，用于将过程噪声映射到状态空间

    状态变量假设为4维：[位置_x; 位置_y; 速度_x; 速度_y]
    """
    # 初始化为单位矩阵（4×4）
    A = np.eye(4)

    # 位置受速度影响，位置更新方程中包含速度乘以时间间隔
    A[0, 2] = scan_time  # x位置随x速度变化
    A[1, 3] = scan_time  # y位置随y速度变化

    # 过程噪声输入矩阵W，将二维加速度噪声映射到4维状态空间
    W = np.zeros((4, 2))
    W[0, 0] = 0.5 * scan_time**2  # 位置x受加速度x影响，积分关系0.5*t^2
    W[1, 1] = 0.5 * scan_time**2  # 位置y受加速度y影响
    W[2, 0] = scan_time  # 速度x受加速度x影响，积分关系t
    W[3, 1] = scan_time  # 速度y受加速度y影响

    return A, W


def perform_prediction(old_particles, parameters):
    """
    根据运动模型和过程噪声预测下一时刻粒子状态

    参数:
        old_particles: 旧时刻粒子状态矩阵，shape (4, num_particles)
        parameters: 参数字典，包含运动模型参数和噪声方差等

    返回:
        predicted_particles: 预测的粒子状态矩阵，shape (4, num_particles)
    """
    # 采样时间间隔
    scan_time = parameters['scanTime']
    # 过程噪声方差（假设均匀，标量）
    driving_noise_variance = parameters['drivingNoiseVariance']

    num_particles = old_particles.shape[1]

    # 获取状态转移矩阵A和过程噪声输入矩阵W
    A, W = get_transition_matrices(scan_time)

    # 根据运动模型进行状态预测，并加入过程噪声
    # randn(2, num_particles) 产生2维加速度噪声样本（加速度输入）
    predicted_particles = (A @ old_particles +
                          W @ (np.sqrt(driving_noise_variance) *
                               np.random.randn(2, num_particles)))

    return predicted_particles
