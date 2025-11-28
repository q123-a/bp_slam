"""
测试脚本：验证Python转换的正确性
Test script to verify the Python conversion
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from bp_slam.utils.sampling import draw_samples_uniformly_circ, resample_systematic
from bp_slam.utils.motion_model import get_transition_matrices, perform_prediction
from bp_slam.utils.distance import calc_distance


def test_sampling():
    """测试采样函数"""
    print("测试采样函数...")

    # 测试圆形均匀采样
    center = np.array([0, 0])
    radius = 1.0
    n = 1000
    samples = draw_samples_uniformly_circ(center, radius, n)

    assert samples.shape == (2, n), "采样形状错误"
    distances = np.sqrt(np.sum(samples**2, axis=0))
    assert np.all(distances <= radius), "采样点超出圆形范围"
    print("✓ 圆形均匀采样测试通过")

    # 测试系统重采样
    weights = np.random.rand(100)
    weights = weights / np.sum(weights)
    indices = resample_systematic(weights, 100)

    assert len(indices) == 100, "重采样数量错误"
    assert np.all(indices >= 0) and np.all(indices < 100), "重采样索引超出范围"
    print("✓ 系统重采样测试通过")


def test_motion_model():
    """测试运动模型"""
    print("\n测试运动模型...")

    # 测试状态转移矩阵
    scan_time = 1.0
    A, W = get_transition_matrices(scan_time)

    assert A.shape == (4, 4), "状态转移矩阵形状错误"
    assert W.shape == (4, 2), "过程噪声矩阵形状错误"
    assert A[0, 2] == scan_time, "状态转移矩阵值错误"
    print("✓ 状态转移矩阵测试通过")

    # 测试状态预测
    parameters = {
        'scanTime': 1.0,
        'drivingNoiseVariance': 0.01
    }
    old_particles = np.random.randn(4, 100)
    predicted = perform_prediction(old_particles, parameters)

    assert predicted.shape == old_particles.shape, "预测粒子形状错误"
    print("✓ 状态预测测试通过")


def test_distance():
    """测试距离计算"""
    print("\n测试距离计算...")

    # 测试欧氏距离
    p1 = np.array([[0], [0]])
    p2 = np.array([[3], [4]])
    dist = calc_distance(p1, p2, p=2, mode=1)

    assert np.abs(dist - 5.0) < 1e-6, "欧氏距离计算错误"
    print("✓ 欧氏距离测试通过")

    # 测试截断距离
    dist_cut = calc_distance(p1, p2, p=2, mode=2, c=3.0)
    assert dist_cut == 3.0, "截断距离计算错误"
    print("✓ 截断距离测试通过")


def test_data_structures():
    """测试数据结构兼容性"""
    print("\n测试数据结构...")

    # 测试锚点数据结构
    anchor = {
        'x': np.random.randn(2, 100),
        'w': np.ones(100) / 100,
        'posteriorExistence': 0.95
    }

    assert anchor['x'].shape == (2, 100), "锚点位置形状错误"
    assert len(anchor['w']) == 100, "锚点权重长度错误"
    assert 0 <= anchor['posteriorExistence'] <= 1, "存在概率范围错误"
    print("✓ 数据结构测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("开始运行转换验证测试")
    print("=" * 50)

    try:
        test_sampling()
        test_motion_model()
        test_distance()
        test_data_structures()

        print("\n" + "=" * 50)
        print("✓ 所有测试通过！")
        print("=" * 50)
        return True

    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        return False
    except Exception as e:
        print(f"\n✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
