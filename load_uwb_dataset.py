#!/usr/bin/env python3
"""
UWB数据集加载脚本
将data_set文件夹中的真实UWB数据转换为BP-SLAM可用的格式
"""
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io as sio

def load_uwb_dataset(location='location0', use_amplitude=True):
    """
    加载UWB数据集

    参数:
        location: 数据集位置 ('location0', 'location1', 'location2')
        use_amplitude: 是否使用RSS幅度信息 (默认True)

    返回:
        true_trajectory: (2, M) 真实轨迹 [x, y]
        anchors: (2, K) 锚点位置 [x, y]
        measurements: list[list[ndarray]] 测量数据 [num_steps][num_sensors]
                     每个测量: (3, num_detections) [距离, 方差, 幅度]
    """
    base_path = Path('data_set')
    location_path = base_path / location

    print("=" * 70)
    print(f"加载UWB数据集: {location}")
    print("=" * 70)

    # 1. 加载真实轨迹
    print("\n1. 加载真实轨迹...")
    trajectory_file = location_path / 'walking_path.csv'
    traj_data = pd.read_csv(trajectory_file, header=None).values  # (M, 3) [x, y, z]
    true_trajectory = traj_data[:, :2].T  # (2, M) 只取x, y
    num_steps = true_trajectory.shape[1]
    print(f"   ✓ 轨迹点数: {num_steps}")
    print(f"   ✓ 轨迹范围: X=[{true_trajectory[0].min():.2f}, {true_trajectory[0].max():.2f}]m, "
          f"Y=[{true_trajectory[1].min():.2f}, {true_trajectory[1].max():.2f}]m")

    # 2. 加载锚点位置
    print("\n2. 加载锚点位置...")
    anchors_file = location_path / 'anchors.csv'
    anchors_data = pd.read_csv(anchors_file, header=None).values  # (K, 4) [ID, x, y, z]
    anchors = anchors_data[:, 1:3].T  # (2, K) 只取x, y
    anchor_ids = anchors_data[:, 0]  # 锚点ID
    num_anchors = anchors.shape[1]
    print(f"   ✓ 锚点数量: {num_anchors}")
    print(f"   ✓ 锚点ID: {anchor_ids}")

    # 3. 加载原始测量数据
    print("\n3. 加载原始测量数据...")
    raw_data_path = base_path / 'raw_data' / location / 'data'

    # 初始化测量数据结构
    measurements = [[None for _ in range(num_anchors)] for _ in range(num_steps)]

    # 统计信息
    total_measurements = 0
    measurements_per_step = []

    # 遍历每个轨迹点
    for step_idx in range(num_steps):
        # 获取当前位置的文件夹名称
        x, y, z = traj_data[step_idx]
        folder_name = f"{x:.2f}_{y:.2f}_{z:.2f}"
        step_folder = raw_data_path / folder_name

        if not step_folder.exists():
            print(f"   ⚠ 警告: 步数 {step_idx} 的数据文件夹不存在: {folder_name}")
            # 创建空测量
            for anchor_idx in range(num_anchors):
                measurements[step_idx][anchor_idx] = np.zeros((3, 0))
            continue

        step_meas_count = 0

        # 遍历每个锚点
        for anchor_idx, anchor_id in enumerate(anchor_ids):
            # 查找对应的CSV文件 (格式: ch*_A*.csv)
            csv_files = list(step_folder.glob(f'*_{anchor_id}.csv'))

            if len(csv_files) == 0:
                # 没有测量数据
                measurements[step_idx][anchor_idx] = np.zeros((3, 0))
                continue

            # 读取第一个匹配的文件
            csv_file = csv_files[0]

            try:
                # 读取CSV文件
                df = pd.read_csv(csv_file, skiprows=1)  # 跳过版本号行

                if len(df) == 0:
                    measurements[step_idx][anchor_idx] = np.zeros((3, 0))
                    continue

                # 提取关键信息
                ranges = df['RANGE'].values  # 距离 (m)
                rss = df['RSS'].values  # 接收信号强度 (dBm)
                stdev_noise = df['STDEV_NOISE'].values  # 噪声标准差

                num_detections = len(ranges)
                step_meas_count += num_detections

                # 构造测量数据 (3, num_detections)
                meas = np.zeros((3, num_detections))
                meas[0, :] = ranges  # 距离

                # 方差：使用噪声标准差的平方，设置最小值0.0025 m²
                variances = (stdev_noise / 100.0) ** 2  # stdev_noise单位可能是cm
                meas[1, :] = np.maximum(variances, 0.0025)

                # 幅度：将RSS (dBm) 转换为线性幅度
                if use_amplitude:
                    # RSS (dBm) -> 线性功率: P = 10^(RSS/10) mW
                    # 幅度 = sqrt(P)
                    linear_power = 10 ** (rss / 10.0)  # mW
                    meas[2, :] = np.sqrt(linear_power / 1000.0)  # 转换为W的平方根
                else:
                    meas[2, :] = 0.0

                measurements[step_idx][anchor_idx] = meas

            except Exception as e:
                print(f"   ⚠ 警告: 读取文件失败 {csv_file.name}: {e}")
                measurements[step_idx][anchor_idx] = np.zeros((3, 0))

        measurements_per_step.append(step_meas_count)
        total_measurements += step_meas_count

        if (step_idx + 1) % 10 == 0:
            print(f"   进度: {step_idx + 1}/{num_steps} 步")

    print(f"\n   ✓ 总测量数: {total_measurements}")
    print(f"   ✓ 平均每步测量数: {np.mean(measurements_per_step):.1f}")
    print(f"   ✓ 测量数范围: [{np.min(measurements_per_step)}, {np.max(measurements_per_step)}]")

    # 4. 数据质量检查
    print("\n4. 数据质量检查...")

    # 检查距离范围
    all_ranges = []
    all_rss = []
    for step_meas in measurements:
        for anchor_meas in step_meas:
            if anchor_meas.shape[1] > 0:
                all_ranges.extend(anchor_meas[0, :])
                if use_amplitude:
                    all_rss.extend(anchor_meas[2, :])

    print(f"   ✓ 距离范围: [{np.min(all_ranges):.2f}, {np.max(all_ranges):.2f}]m")
    print(f"   ✓ 平均距离: {np.mean(all_ranges):.2f}m")

    if use_amplitude and len(all_rss) > 0:
        print(f"   ✓ 幅度范围: [{np.min(all_rss):.6f}, {np.max(all_rss):.6f}]")
        print(f"   ✓ 平均幅度: {np.mean(all_rss):.6f}")

    print("\n" + "=" * 70)
    print("数据集加载完成！")
    print("=" * 70)

    return true_trajectory, anchors, measurements, anchor_ids


def save_to_mat_format(true_trajectory, anchors, measurements, anchor_ids,
                       output_file='uwb_dataset.mat', location='location0'):
    """
    将UWB数据集保存为MAT格式，兼容现有的BP-SLAM代码

    参数:
        true_trajectory: (2, M) 真实轨迹
        anchors: (2, K) 锚点位置
        measurements: 测量数据列表
        anchor_ids: 锚点ID列表
        output_file: 输出文件名
        location: 数据集位置名称
    """
    print(f"\n保存为MAT格式: {output_file}")

    num_steps = true_trajectory.shape[1]
    num_sensors = len(measurements[0])

    # 构造dataVA结构（虚拟锚点数据）
    data_va = []
    for sensor_idx in range(num_sensors):
        sensor_data = {
            'positions': anchors[:, sensor_idx:sensor_idx+1],  # (2, 1)
            'visibility': np.ones((1, num_steps))  # 假设所有锚点全程可见
        }
        data_va.append(sensor_data)

    # 构造estimated_measurements_cell
    estimated_measurements_cell = np.empty((num_steps, num_sensors), dtype=object)
    for step_idx in range(num_steps):
        for sensor_idx in range(num_sensors):
            estimated_measurements_cell[step_idx, sensor_idx] = measurements[step_idx][sensor_idx]

    # 保存为MAT文件
    sio.savemat(output_file, {
        'trueTrajectory': true_trajectory,
        'dataVA': np.array(data_va, dtype=object).reshape(-1, 1),
        'estimated_measurements_cell': estimated_measurements_cell,
        'anchor_ids': anchor_ids,
        'location': location
    })

    print(f"✓ 已保存: {output_file}")


def main():
    """主函数：加载并转换UWB数据集"""
    import argparse

    parser = argparse.ArgumentParser(description='加载UWB数据集')
    parser.add_argument('--location', type=str, default='location0',
                       choices=['location0', 'location1', 'location2'],
                       help='数据集位置')
    parser.add_argument('--output', type=str, default=None,
                       help='输出MAT文件名 (默认: uwb_{location}.mat)')
    parser.add_argument('--no-amplitude', action='store_true',
                       help='不使用幅度信息')

    args = parser.parse_args()

    # 加载数据集
    true_trajectory, anchors, measurements, anchor_ids = load_uwb_dataset(
        location=args.location,
        use_amplitude=not args.no_amplitude
    )

    # 保存为MAT格式
    if args.output is None:
        args.output = f'uwb_{args.location}.mat'

    save_to_mat_format(
        true_trajectory, anchors, measurements, anchor_ids,
        output_file=args.output,
        location=args.location
    )

    print("\n" + "=" * 70)
    print("使用方法:")
    print(f"  python testbed.py --mode gnn --steps {true_trajectory.shape[1]} \\")
    print(f"                    --load-measurements {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()
