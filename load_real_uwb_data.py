#!/usr/bin/env python3
"""
加载 data_set 中的真实UWB数据并转换为BP-SLAM格式
"""
import numpy as np
import pandas as pd
import json
import scipy.io as sio
from pathlib import Path
import argparse

def load_uwb_dataset(location='location0', use_amplitude=True, max_steps=None):
    """
    加载真实UWB数据集

    参数:
        location: 数据集位置 ('location0', 'location1', 'location2', 'location3')
        use_amplitude: 是否使用RSS幅度信息
        max_steps: 最大加载步数（None表示全部加载）

    返回:
        true_trajectory: (2, M) 真实轨迹 [x, y]
        anchors: (2, K) 锚点位置 [x, y]
        measurements: list[list[ndarray]] 测量数据
        anchor_ids: 锚点ID列表
    """
    base_path = Path('data_set')
    location_path = base_path / location

    print("=" * 70)
    print(f"加载真实UWB数据集: {location}")
    print("=" * 70)

    # 1. 加载锚点位置
    print("\n1. 加载锚点位置...")
    anchors_file = location_path / 'anchors.csv'
    anchors_df = pd.read_csv(anchors_file, header=None, names=['id', 'x', 'y', 'z'])
    anchors = anchors_df[['x', 'y']].values.T  # (2, K)
    anchor_ids = anchors_df['id'].values
    num_anchors = len(anchor_ids)
    print(f"   ✓ 锚点数量: {num_anchors}")
    print(f"   ✓ 锚点ID: {anchor_ids}")
    print(f"   ✓ 锚点位置范围: X=[{anchors[0].min():.2f}, {anchors[0].max():.2f}]m, "
          f"Y=[{anchors[1].min():.2f}, {anchors[1].max():.2f}]m")

    # 2. 加载真实轨迹
    print("\n2. 加载真实轨迹...")
    trajectory_file = location_path / 'walking_path.csv'
    traj_df = pd.read_csv(trajectory_file, header=None, names=['x', 'y', 'z'])

    if max_steps is not None:
        traj_df = traj_df.iloc[:max_steps]

    true_trajectory = traj_df[['x', 'y']].values.T  # (2, M)
    num_steps = true_trajectory.shape[1]
    print(f"   ✓ 轨迹点数: {num_steps}")
    print(f"   ✓ 轨迹范围: X=[{true_trajectory[0].min():.2f}, {true_trajectory[0].max():.2f}]m, "
          f"Y=[{true_trajectory[1].min():.2f}, {true_trajectory[1].max():.2f}]m")

    # 3. 加载测量数据 (从JSON文件)
    print("\n3. 加载测量数据...")
    data_file = location_path / 'data.json'

    print(f"   正在读取 {data_file.name} (可能需要一些时间)...")
    with open(data_file, 'r') as f:
        data_json = json.load(f)

    # 初始化测量数据结构
    measurements = [[None for _ in range(num_anchors)] for _ in range(num_steps)]

    # 统计信息
    total_measurements = 0
    measurements_per_step = []

    # 解析JSON数据
    for step_idx in range(num_steps):
        step_key = str(step_idx)
        if step_key not in data_json:
            # 没有数据，创建空测量
            for anchor_idx in range(num_anchors):
                measurements[step_idx][anchor_idx] = np.zeros((3, 0))
            measurements_per_step.append(0)
            continue

        step_data = data_json[step_key]
        step_meas_count = 0

        for anchor_idx, anchor_id in enumerate(anchor_ids):
            if anchor_id not in step_data:
                measurements[step_idx][anchor_idx] = np.zeros((3, 0))
                continue

            anchor_data = step_data[anchor_id]

            if len(anchor_data) == 0:
                measurements[step_idx][anchor_idx] = np.zeros((3, 0))
                continue

            # 提取测量数据
            ranges = []
            variances = []
            rss_values = []

            for meas in anchor_data:
                if 'range' in meas and 'rss' in meas:
                    ranges.append(meas['range'])
                    # 使用固定方差或从数据中提取
                    variances.append(meas.get('variance', 0.01))  # 默认1cm²
                    rss_values.append(meas['rss'])

            num_detections = len(ranges)
            step_meas_count += num_detections

            if num_detections == 0:
                measurements[step_idx][anchor_idx] = np.zeros((3, 0))
                continue

            # 构造测量数据 (3, num_detections)
            meas = np.zeros((3, num_detections))
            meas[0, :] = ranges  # 距离
            meas[1, :] = variances  # 方差

            if use_amplitude:
                # RSS (dBm) -> 线性幅度
                linear_power = 10 ** (np.array(rss_values) / 10.0)  # mW
                meas[2, :] = np.sqrt(linear_power / 1000.0)  # 转换为W的平方根
            else:
                meas[2, :] = 0.0

            measurements[step_idx][anchor_idx] = meas

        measurements_per_step.append(step_meas_count)
        total_measurements += step_meas_count

        if (step_idx + 1) % 50 == 0:
            print(f"   进度: {step_idx + 1}/{num_steps} 步")

    print(f"\n   ✓ 总测量数: {total_measurements}")
    print(f"   ✓ 平均每步测量数: {np.mean(measurements_per_step):.1f}")
    print(f"   ✓ 测量数范围: [{np.min(measurements_per_step)}, {np.max(measurements_per_step)}]")

    print("\n" + "=" * 70)
    print("数据集加载完成！")
    print("=" * 70)

    return true_trajectory, anchors, measurements, anchor_ids


def save_to_mat_format(true_trajectory, anchors, measurements, anchor_ids,
                       output_file='uwb_dataset.mat', location='location0'):
    """
    将数据保存为MAT格式，兼容BP-SLAM
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
    parser = argparse.ArgumentParser(description='加载真实UWB数据集')
    parser.add_argument('--location', type=str, default='location0',
                       choices=['location0', 'location1', 'location2', 'location3'],
                       help='数据集位置')
    parser.add_argument('--output', type=str, default=None,
                       help='输出MAT文件名 (默认: uwb_{location}.mat)')
    parser.add_argument('--no-amplitude', action='store_true',
                       help='不使用幅度信息')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='最大加载步数 (用于快速测试)')

    args = parser.parse_args()

    # 加载数据集
    true_trajectory, anchors, measurements, anchor_ids = load_uwb_dataset(
        location=args.location,
        use_amplitude=not args.no_amplitude,
        max_steps=args.max_steps
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
