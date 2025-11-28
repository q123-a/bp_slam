"""
距离计算相关函数
Distance calculation functions
"""

import numpy as np


def calc_distance(p1, p2, p=2, mode=1, c=None):
    """
    计算点集之间的p阶距离，支持多种模式

    参数:
        p1: shape (d, n) 的矩阵，列为n个d维点
        p2: shape (d, m) 的矩阵，列为m个d维点
        p: 范数阶数（默认为2，即欧几里得距离）
        mode: 计算模式（可选）：
              1 - 常规p阶距离（默认）
              2 - 截断距离，结果距离最大为c
              3 - 计算欧式距离矩阵，输出 (n, m) 矩阵
        c: 截断距离阈值，仅mode=2时有效

    返回:
        d: 计算得到的距离向量或矩阵

    作者：Paul Meissner, SPSC Lab, 2010/11/13
    转换为Python: 2025
    """
    # 如果任一输入为空，返回NaN
    if p1 is None or p2 is None or p1.size == 0 or p2.size == 0:
        return np.nan

    # 如果mode=2但没给截断值c，报错
    if mode == 2 and c is None:
        raise ValueError('Cutoff mode selected without specifying cutoff value!')

    # 获取点的数量（列数）
    n = p1.shape[1] if len(p1.shape) > 1 else 1
    m = p2.shape[1] if len(p2.shape) > 1 else 1

    # 如果p1和p2点数不匹配且不是单点对多点模式，且不是模式3，报错
    if (n != m) and (n > 1) and mode != 3:
        raise ValueError('Size of input vectors incorrect!')

    # 如果p1只有1列而p2有多列，则复制p1使两者列数一致
    if (n == 1) and (m > n):
        p1 = np.tile(p1, (1, m))

    # 模式3，计算欧式距离矩阵（n×m）
    if mode == 3:
        d = np.zeros((n, m))
        for i in range(n):
            for j in range(i + 1, m):
                # 计算p阶范数距离
                d[i, j] = np.linalg.norm(p1[:, i] - p2[:, j], ord=p)
        # 补全对称矩阵
        d = d + d.T
        # 如果给了截断值c，执行截断
        if c is not None:
            d = np.minimum(c, d)
        return d

    # 常规距离计算，按列计算范数
    d = np.sum(np.abs(p1 - p2)**p, axis=0)**(1/p)

    # 如果模式为2，执行截断距离
    if mode == 2:
        d = np.minimum(c, d)

    return d
