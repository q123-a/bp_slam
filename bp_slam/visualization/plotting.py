"""
可视化函数
Visualization functions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def ospa_dist(X, Y, c, p):
    """
    计算两个有限点集的OSPA距离

    参数:
        X, Y: 点集，shape (2, n) 和 (2, m)，列为点坐标
        c: 截断参数，限制单点距离的惩罚
        p: 指数参数，用于计算p阶范数

    返回:
        dist: 计算得到的X和Y之间的OSPA距离
        loc_error: (可选)位置误差
        card_error: (可选)基数误差

    说明:
        OSPA距离统一度量位置误差和基数误差（元素个数差异），
        常用于多目标跟踪性能评估。

    参考文献:
        Schuhmacher et al., IEEE Trans. Signal Processing, 2008.
    """
    # 如果两个集合都为空，距离为0
    if (X is None or X.size == 0) and (Y is None or Y.size == 0):
        return 0, 0, 0

    # 如果一个集合为空，距离为截断参数c乘以惩罚
    if X is None or X.size == 0 or Y is None or Y.size == 0:
        n = 0 if X is None or X.size == 0 else X.shape[1]
        m = 0 if Y is None or Y.size == 0 else Y.shape[1]
        dist = c
        return dist, 0, c

    # 获取两个集合的点数
    n = X.shape[1]
    m = Y.shape[1]

    # 计算所有点对之间的欧氏距离矩阵D，大小 n×m
    # D[i,j] = distance between X[:,i] and Y[:,j]
    # 使用广播计算：X[:, :, np.newaxis] - Y[:, np.newaxis, :]
    # 结果shape: (2, n, m)
    diff = X[:, :, np.newaxis] - Y[:, np.newaxis, :]  # shape: (2, n, m)
    D = np.sqrt(np.sum(diff**2, axis=0))  # shape: (n, m)
    D = np.minimum(c, D)**p  # 截断并取p次方

    # 使用匈牙利算法（最小权匹配）求解最优分配
    row_ind, col_ind = linear_sum_assignment(D)
    cost = D[row_ind, col_ind].sum()

    # 计算总体OSPA距离
    dist = (1 / max(m, n) * (c**p * abs(m - n) + cost))**(1 / p)

    # 如果需要返回位置误差和基数误差
    loc_error = (1 / max(m, n) * cost)**(1 / p)  # 位置误差
    card_error = (1 / max(m, n) * c**p * abs(m - n))**(1 / p)  # 基数误差

    return dist, loc_error, card_error


def plot_scatter_2d(X, color='b', marker='.', markersize=1, alpha=0.3):
    """
    绘制2D散点图

    参数:
        X: 点坐标，shape (2, n)
        color: 颜色
        marker: 标记样式
        markersize: 标记大小
        alpha: 透明度
    """
    if X is not None and X.size > 0:
        plt.scatter(X[0, :], X[1, :], c=color, marker=marker,
                   s=markersize, alpha=alpha)


def plot_all(true_trajectory, estimated_trajectory, estimated_anchors,
            posterior_particles_anchors, num_estimated_anchors,
            data_va, parameters, mode=0, num_steps=None):
    """
    绘制所有结果：真实轨迹、估计轨迹、锚点估计等

    参数:
        true_trajectory: 真实轨迹，shape (n_dims, num_steps)
        estimated_trajectory: 估计轨迹，shape (4, num_steps)
        estimated_anchors: 估计的锚点状态
        posterior_particles_anchors: 锚点粒子集合
        num_estimated_anchors: 每时刻估计的锚点数量
        data_va: 虚拟锚点数据
        parameters: 参数字典
        mode: 绘图模式（0=最终状态，1=动画）
        num_steps: 时间步数
    """
    if num_steps is None:
        num_steps = true_trajectory.shape[1]

    num_sensors = len(data_va)

    # 定义颜色方案
    mycolors = np.array([
        [0.66, 0.00, 0.00],  # 红色
        [0.00, 0.30, 0.70],  # 蓝色
        [0.60, 0.90, 0.16],  # 绿色
        [0.54, 0.80, 0.99],  # 浅蓝
        [0.99, 0.34, 0.00],  # 橙色
        [0.92, 0.75, 0.33],  # 黄色
        [0.00, 0.00, 0.00],  # 黑色
    ])

    plt.figure(figsize=(12, 10))

    # 绘制真实轨迹
    plt.plot(true_trajectory[0, :], true_trajectory[1, :],
            '-', color=[0.5, 0.5, 0.5], linewidth=1.5, label='True Trajectory')

    # 绘制估计轨迹
    plt.plot(estimated_trajectory[0, :num_steps], estimated_trajectory[1, :num_steps],
            '-', color=[0, 0.5, 0], linewidth=1.5, label='Estimated Trajectory')

    # 绘制每个传感器的锚点
    for sensor in range(num_sensors):
        # 绘制真实锚点位置
        true_anchor_positions = data_va[sensor]['positions']
        plt.plot(true_anchor_positions[0, :], true_anchor_positions[1, :],
                linestyle='none', marker='s', markersize=8,
                color=mycolors[sensor], markeredgecolor=mycolors[sensor],
                label=f'True Anchors Sensor {sensor + 1}')

        # 绘制估计的锚点
        if mode == 0:
            tmp = num_steps - 1  # Python索引从0开始
        else:
            tmp = num_steps - 1

        if estimated_anchors[sensor][tmp] is not None:
            num_positions = len(estimated_anchors[sensor][tmp])
            for anchor in range(num_positions):
                if estimated_anchors[sensor][tmp][anchor] is None:
                    continue

                anchor_pos = estimated_anchors[sensor][tmp][anchor]['x']
                anchor_existence = estimated_anchors[sensor][tmp][anchor]['posteriorExistence']

                if anchor_existence > parameters['detectionThreshold']:
                    # 绘制锚点粒子云
                    if (posterior_particles_anchors is not None and
                        sensor < len(posterior_particles_anchors) and
                        anchor < len(posterior_particles_anchors[sensor])):
                        particles = posterior_particles_anchors[sensor][anchor]['x']
                        plot_scatter_2d(particles, color=mycolors[sensor],
                                      marker='.', markersize=1, alpha=0.3)

                    # 绘制锚点估计位置
                    plt.plot(anchor_pos[0], anchor_pos[1],
                            color='k', marker='+', markersize=8)

    # 绘制最终估计位置
    plt.plot(estimated_trajectory[0, -1], estimated_trajectory[1, -1],
            color=[0, 0.5, 0], marker='o', markersize=10,
            markerfacecolor='none', markeredgewidth=2,
            label='Final Estimated Position')

    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.title('BP-SLAM Results', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('bp_slam_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_trajectory_error(true_trajectory, estimated_trajectory):
    """
    绘制轨迹估计误差随时间变化

    参数:
        true_trajectory: 真实轨迹，shape (n_dims, num_steps)
        estimated_trajectory: 估计轨迹，shape (4, num_steps)
    """
    num_steps = true_trajectory.shape[1]
    errors = np.sqrt(np.sum((true_trajectory[0:2, :] - estimated_trajectory[0:2, :])**2, axis=0))

    plt.figure(figsize=(10, 6))
    plt.plot(range(num_steps), errors, 'b-', linewidth=1.5)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Position Error (m)', fontsize=12)
    plt.title('Trajectory Estimation Error', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('trajectory_error.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f'Mean position error: {np.mean(errors):.4f} m')
    print(f'Max position error: {np.max(errors):.4f} m')
    print(f'Final position error: {errors[-1]:.4f} m')
