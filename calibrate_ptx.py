#!/usr/bin/env python3
"""
自动校准 P_tx 参数脚本
根据实际测量数据反推最佳的发射功率参数
"""

import scipy.io as sio
import numpy as np
import sys
from pathlib import Path


def calibrate_from_measurements(meas_file='measurementbadf.mat',
                                 traj_file='scenarioCleanM2_new_1500.mat',
                                 n_loss=2.0):
    """
    从测量数据中校准 P_tx 参数

    参数:
        meas_file: 测量数据文件 (包含距离和幅度)
        traj_file: 真实轨迹文件
        n_loss: 路径损耗指数 (默认 2.0)

    返回:
        estimated_ptx: 估计的发射功率 (dBm)
    """
    print("=" * 60)
    print("P_tx 自动校准工具")
    print("=" * 60)

    # 1. 加载测量数据
    print(f"\n[1/4] 加载测量数据: {meas_file}")
    try:
        meas_data = sio.loadmat(meas_file)
        print(f"  ✓ 成功加载")
    except FileNotFoundError:
        print(f"  ✗ 错误：找不到文件 {meas_file}")
        return None

    # 2. 加载轨迹数据
    print(f"\n[2/4] 加载轨迹数据: {traj_file}")
    try:
        traj_data = sio.loadmat(traj_file)
        true_traj = traj_data['trueTrajectory']
        print(f"  ✓ 成功加载，轨迹长度: {true_traj.shape[1]} 步")
    except FileNotFoundError:
        print(f"  ✗ 错误：找不到文件 {traj_file}")
        return None
    except KeyError:
        print(f"  ✗ 错误：文件中找不到 'trueTrajectory' 变量")
        return None

    # 3. 提取测量数据
    print(f"\n[3/4] 分析测量数据...")

    # 尝试不同的可能变量名
    measurements_raw = None
    for key in ['estimated_measurements_cell', 'measurements', 'Z', 'cluttered_measurements']:
        if key in meas_data:
            measurements_raw = meas_data[key]
            print(f"  ✓ 找到测量数据: '{key}'")
            break

    if measurements_raw is None:
        print(f"  ✗ 错误：找不到测量数据变量")
        print(f"  可用变量: {list(meas_data.keys())}")
        return None

    # 光速常数
    SPEED_OF_LIGHT = 3.0e8

    # 收集 P_tx 估计值
    p_tx_estimates = []
    sample_count = 0

    num_steps, num_sensors = measurements_raw.shape
    num_steps = min(num_steps, true_traj.shape[1])

    print(f"  - 数据维度: {num_steps} 步, {num_sensors} 个传感器")
    print(f"  - 路径损耗指数: n = {n_loss}")
    print(f"  - 正在采样分析...")

    # 4. 遍历数据进行采样
    for step in range(0, num_steps, 5):  # 每5帧采样一次
        for sensor in range(num_sensors):
            mvalse_data = measurements_raw[step, sensor]

            if mvalse_data.size == 0:
                continue

            # 检查数据维度
            if mvalse_data.shape[0] < 3:
                if step == 0 and sensor == 0:
                    print(f"  ⚠ 警告：数据只有 {mvalse_data.shape[0]} 维，缺少幅度信息")
                continue

            # 提取数据
            # 第0行: 时延 (s) -> 转换为距离 (m)
            # 第2行: 幅度 (dBm)
            delays = mvalse_data[0, :]
            rss_meas = mvalse_data[2, :]

            # 转换为距离
            dists = delays * SPEED_OF_LIGHT

            # 只使用近距离测量 (< 10m)，因为近处多径影响小
            valid_idx = (dists > 0.5) & (dists < 10.0)

            if valid_idx.sum() == 0:
                continue

            # 反推 P_tx = RSS_measured + 10 * n * log10(distance)
            valid_dists = dists[valid_idx]
            valid_rss = rss_meas[valid_idx]

            # 计算每个测量点的 P_tx 估计
            p_tx_est = valid_rss + 10 * n_loss * np.log10(valid_dists)
            p_tx_estimates.extend(p_tx_est.tolist())
            sample_count += len(p_tx_est)

    # 5. 计算结果
    print(f"\n[4/4] 计算校准结果...")

    if len(p_tx_estimates) == 0:
        print("  ✗ 错误：未找到有效的测量数据")
        print("  可能原因：")
        print("    1. 数据中没有幅度信息（第3维）")
        print("    2. 所有测量距离都超出范围 (0.5m ~ 10m)")
        return None

    p_tx_estimates = np.array(p_tx_estimates)

    # 使用中位数（比均值更鲁棒）
    estimated_ptx = np.median(p_tx_estimates)
    mean_ptx = np.mean(p_tx_estimates)
    std_ptx = np.std(p_tx_estimates)

    print(f"  ✓ 分析了 {sample_count} 个有效测量点")
    print(f"\n" + "=" * 60)
    print(f"📊 校准结果:")
    print(f"=" * 60)
    print(f"  推荐值 (中位数): P_tx = {estimated_ptx:.2f} dBm")
    print(f"  均值:           P_tx = {mean_ptx:.2f} dBm")
    print(f"  标准差:         σ = {std_ptx:.2f} dB")
    print(f"  数据范围:       [{np.min(p_tx_estimates):.2f}, {np.max(p_tx_estimates):.2f}] dBm")
    print(f"=" * 60)

    # 6. 生成修改建议
    print(f"\n💡 使用建议:")
    print(f"=" * 60)
    print(f"请在以下文件中修改 P_tx 参数：")
    print(f"\n1. bp_slam/core/slam.py (第 197 行):")
    print(f"   P_tx = {estimated_ptx:.2f}  # 发射功率 (dBm) - 自动校准")
    print(f"\n2. bp_slam/core/gnn_trainer.py (第 149 行):")
    print(f"   P_tx = {estimated_ptx:.2f}  # 根据数据校准")
    print(f"=" * 60)

    # 7. 数据质量评估
    print(f"\n📈 数据质量评估:")
    print(f"=" * 60)
    if std_ptx < 5.0:
        print(f"  ✓ 优秀：标准差 < 5 dB，数据一致性很好")
    elif std_ptx < 10.0:
        print(f"  ✓ 良好：标准差 < 10 dB，数据质量可接受")
    else:
        print(f"  ⚠ 警告：标准差 > 10 dB，数据波动较大")
        print(f"    可能原因：")
        print(f"      - 多径效应严重")
        print(f"      - 路径损耗指数 n={n_loss} 不准确")
        print(f"      - 测量噪声较大")
    print(f"=" * 60)

    return estimated_ptx


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='自动校准 P_tx 参数')
    parser.add_argument('--meas', type=str, default='measurementbadf.mat',
                        help='测量数据文件 (默认: measurementbadf.mat)')
    parser.add_argument('--traj', type=str, default='scenarioCleanM2_new_1500.mat',
                        help='轨迹数据文件 (默认: scenscenarioCleanM2_new_1500.mat
    parser.add_argument('--n', type=float, default=2.0,
                        help='路径损耗指数 (默认: 2.0)')

    args = parser.parse_args()

    # 运行校准
    result = calibrate_from_measurements(
        meas_file=args.meas,
        traj_file=args.traj,
        n_loss=args.n
    )

    if result is None:
        print("\n❌ 校准失败")
        sys.exit(1)
    else:
        print(f"\n✅ 校准成功！建议使用 P_tx = {result:.2f} dBm")
        sys.exit(0)


if __name__ == '__main__':
    main()
