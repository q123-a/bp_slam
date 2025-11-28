"""
采样和重采样相关工具函数
Sampling and resampling utility functions
"""

import numpy as np
from numba import jit


def draw_samples_uniformly_circ(pos_center, radius, n):
    """
    在二维圆形区域内均匀采样点

    参数:
        pos_center: 采样圆的中心坐标，shape (2,) 或 (2, 1)
        radius: 采样圆的半径
        n: 需要采样的点数

    返回:
        pos: 采样点坐标，shape (2, n)，每列是一个采样点的 [x; y]
    """
    pos = np.zeros((2, n))
    phi = 2 * np.pi * np.random.rand(n)  # 采样角度，均匀分布在[0, 2π)
    r = np.sqrt(np.random.rand(n))  # 采样半径，经过开方确保在圆内均匀分布

    # 根据极坐标转换为笛卡尔坐标，加上圆心坐标得到最终位置
    pos[0, :] = (radius * r) * np.cos(phi) + pos_center[0]
    pos[1, :] = (radius * r) * np.sin(phi) + pos_center[1]

    return pos


@jit(nopython=True)
def _resample_systematic_core(Q, T, n):
    """
    系统重采样的核心循环（Numba JIT编译加速）

    参数:
        Q: 累积权重分布，shape (n_particles,)
        T: 采样点，shape (n+1,)
        n: 采样数量

    返回:
        indx: 重采样索引，shape (n,)
    """
    indx = np.zeros(n, dtype=np.int64)
    i = 0  # T指针
    j = 0  # Q指针

    while i < n:
        if T[i] < Q[j]:
            indx[i] = j  # 采样该粒子索引
            i += 1  # 移动T指针
        else:
            j += 1  # 权重CDF指针前进

    return indx


def resample_systematic(w, n):
    """
    基于系统重采样算法，从权重分布中采样粒子索引

    参数:
        w: 粒子权重向量（已归一化，和为1），shape (n_particles,)
        n: 需要采样的粒子数量

    返回:
        indx: 重采样后粒子的索引向量，shape (n,)，索引从0开始（Python风格）

    说明:
        系统重采样通过在[0,1)区间均匀采样N个点并与累积权重比较，
        实现低方差粒子重采样，是粒子滤波中常用的重采样策略。
        使用Numba JIT编译加速核心循环。
    """
    Q = np.cumsum(w)  # 计算权重的累积分布函数（CDF）

    # 在[0,1)区间生成N个均匀间隔采样点，起点带随机偏移
    T = np.linspace(0, 1 - 1/n, n) + np.random.rand(1)/n
    T = np.append(T, 1)  # 边界条件，方便索引比较

    # 调用JIT编译的核心函数
    indx = _resample_systematic_core(Q, T, n)

    return indx


def sample_from_likelihood(measurement_to_anchor, measurement_variance,
                          agent_position, num_particles):
    """
    从测量似然中采样锚点位置粒子

    参数:
        measurement_to_anchor: 测量值（距离）
        measurement_variance: 测量方差
        agent_position: 移动体位置粒子，shape (2, num_particles)
        num_particles: 粒子数量

    返回:
        samples: 锚点位置采样，shape (2, num_particles)
    """
    # 采样角度，均匀分布
    phi = 2 * np.pi * np.random.rand(num_particles)

    # 采样距离，基于测量值和测量噪声
    r = measurement_to_anchor + np.sqrt(measurement_variance) * np.random.randn(num_particles)

    # 从移动体位置出发，按照采样的距离和角度生成锚点位置
    samples = np.zeros((2, num_particles))
    samples[0, :] = agent_position[0, :] + r * np.cos(phi)
    samples[1, :] = agent_position[1, :] + r * np.sin(phi)

    return samples
