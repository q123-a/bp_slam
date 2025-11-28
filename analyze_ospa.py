"""
分析OSPA误差，对比Python和MATLAB结果
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from bp_slam.visualization.plotting import ospa_dist

def analyze_ospa_error():
    """详细分析OSPA误差"""

    print("=" * 70)
    print("OSPA误差详细分析")
    print("=" * 70)
    print()

    # 加载真实锚点
    mat_data = sio.loadmat('scenarioCleanM2_new.mat')
    data_va_raw = mat_data['dataVA'][:, 0]

    print("真实锚点信息：")
    for sensor in range(len(data_va_raw)):
        true_pos = data_va_raw[sensor]['positions'][0, 0]
        print(f"\n传感器 {sensor + 1}:")
        print(f"  数量: {true_pos.shape[1]}")
        print(f"  位置 (x, y):")
        for i in range(true_pos.shape[1]):
            print(f"    锚点{i+1}: ({true_pos[0, i]:6.2f}, {true_pos[1, i]:6.2f})")

    print("\n" + "=" * 70)
    print("OSPA误差测试")
    print("=" * 70)
    print()

    # 测试1: 完美匹配（OSPA应该为0）
    print("测试1: 完美匹配（真实位置 vs 真实位置）")
    for sensor in range(len(data_va_raw)):
        true_pos = data_va_raw[sensor]['positions'][0, 0]
        ospa, loc_err, card_err = ospa_dist(true_pos, true_pos, 10, 1)
        print(f"  传感器 {sensor + 1}: OSPA={ospa:.4f}, 位置误差={loc_err:.4f}, 基数误差={card_err:.4f}")
    print()

    # 测试2: 缺少1个锚点
    print("测试2: 缺少1个锚点（模拟估计不准确）")
    for sensor in range(len(data_va_raw)):
        true_pos = data_va_raw[sensor]['positions'][0, 0]
        # 只使用前n-1个锚点
        estimated_pos = true_pos[:, :-1]
        ospa, loc_err, card_err = ospa_dist(true_pos, estimated_pos, 10, 1)
        print(f"  传感器 {sensor + 1}:")
        print(f"    真实锚点数: {true_pos.shape[1]}, 估计锚点数: {estimated_pos.shape[1]}")
        print(f"    OSPA={ospa:.4f}, 位置误差={loc_err:.4f}, 基数误差={card_err:.4f}")
        print(f"    说明: 缺少1个锚点，基数误差贡献 = 10/{true_pos.shape[1]} = {10/true_pos.shape[1]:.4f}")
    print()

    # 测试3: 位置偏移
    print("测试3: 所有锚点位置偏移1米")
    for sensor in range(len(data_va_raw)):
        true_pos = data_va_raw[sensor]['positions'][0, 0]
        # 所有位置偏移1米
        estimated_pos = true_pos + 1.0
        ospa, loc_err, card_err = ospa_dist(true_pos, estimated_pos, 10, 1)
        print(f"  传感器 {sensor + 1}: OSPA={ospa:.4f}, 位置误差={loc_err:.4f}, 基数误差={card_err:.4f}")
    print()

    # 测试4: 位置偏移5米
    print("测试4: 所有锚点位置偏移5米")
    for sensor in range(len(data_va_raw)):
        true_pos = data_va_raw[sensor]['positions'][0, 0]
        estimated_pos = true_pos + 5.0
        ospa, loc_err, card_err = ospa_dist(true_pos, estimated_pos, 10, 1)
        print(f"  传感器 {sensor + 1}: OSPA={ospa:.4f}, 位置误差={loc_err:.4f}, 基数误差={card_err:.4f}")
    print()

    print("=" * 70)
    print("当前Python版本的OSPA误差")
    print("=" * 70)
    print()
    print("传感器 1 平均OSPA误差: 6.25 m")
    print("传感器 2 平均OSPA误差: 5.76 m")
    print()

    print("=" * 70)
    print("分析结论")
    print("=" * 70)
    print()
    print("如果OSPA误差为6-7米，可能的原因：")
    print()
    print("1. 锚点数量不匹配")
    print("   - 如果估计的锚点数量与真实数量相差较大")
    print("   - 每个缺失/多余的锚点贡献约 10/n 米的基数误差")
    print("   - 例如：6个真实锚点，估计0个 → OSPA = 10米")
    print()
    print("2. 锚点位置偏差大")
    print("   - 如果估计位置与真实位置偏差5-7米")
    print("   - 位置误差直接贡献到OSPA")
    print()
    print("3. 存在概率阈值过高")
    print("   - 当前阈值: 0.5")
    print("   - 如果很多锚点的存在概率在0.3-0.5之间")
    print("   - 这些锚点会被过滤掉，导致数量不匹配")
    print()
    print("=" * 70)
    print("请提供MATLAB的对比数据")
    print("=" * 70)
    print()
    print("请告诉我MATLAB运行结果：")
    print("1. 传感器1的平均OSPA误差: _____ m")
    print("2. 传感器2的平均OSPA误差: _____ m")
    print("3. 最后一步的估计锚点数量:")
    print("   - 传感器1: _____ 个")
    print("   - 传感器2: _____ 个")
    print()
    print("这样我可以准确定位问题所在。")
    print()

if __name__ == '__main__':
    analyze_ospa_error()
