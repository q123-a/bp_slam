#!/usr/bin/env python3
"""
从raw_data文件夹提取UWB数据（避免加载大JSON文件）
"""
import numpy as np
import scipy.io as sio
from pathlib import Path
import argparse

def extract_from_raw_data(location='location0', num_steps=50):
    """
    从raw_data文件夹提取数据
    """
    base_path = Path('data_set')
    location_path = base_path / location
    raw_data_path = base_path / 'raw_data' / location / 'data'

    print(f"从raw_data提取 {location} 的前 {num_steps} 步...")

    # 1. 加载锚点
    print("\n1. 加载锚点...")
    anchors_data = np.loadtxt(location_path / 'anchors.csv', delimiter=',', dtype=str)
    anchor_ids = anchors_data[:, 0]
    anchors = anchors_data[:, 1:3].astype(float).T  # (2, K)
    num_anchors = len(anchor_ids)
    print(f"   ✓ {num_anchors} 个锚点: {list(anchor_ids)}")

    # 2. 加载轨迹
    print("\n2. 加载轨迹...")
    trajectory_data = np.loadtxt(location_path / 'walking_path.csv', delimiter=',')
    trajectory_data = trajectory_data[:num_steps]
    true_trajectory = trajectory_data[:, :2].T  # (2, M)
    print(f"   ✓ {num_steps} 个轨迹点")

    # 3. 从raw_data逐步加载测量
    print("\n3. 从raw_data加载测量...")
    measurements = []
    total_meas = 0

    for step_idx in range(num_steps):
        x, y, z = trajectory_data[step_idx]
        folder_name = f"{x:.2f}_{y:.2f}_{z:.2f}"
        step_folder = raw_data_path / folder_name

        step_measurements = []

        if not step_folder.exists():
            # 文件夹不存在
            for _ in range(num_anchors):
                step_measurements.append(np.zeros((3, 0)))
        else:
            # 遍历每个锚点
            for anchor_id in anchor_ids:
                csv_files = list(step_folder.glob(f'*_{anchor_id}.csv'))

                if len(csv_files) == 0:
                    step_measurements.append(np.zeros((3, 0)))
                    continue

                # 读取CSV
                try:
                    # 跳过第一行（版本号）
                    data = np.genfromtxt(csv_files[0], delimiter=',', skip_header=1,
                                        names=['RANGE', 'RSS', 'STDEV_NOISE'])

                    if data.size == 0:
                        step_measurements.append(np.zeros((3, 0)))
                        continue

                    # 处理单行数据
                    if data.ndim == 0:
                        data = np.array([data])

                    ranges = data['RANGE']
                    rss = data['RSS']
                    stdev = data['STDEV_NOISE']

                    # 确保是数组
                    if np.isscalar(ranges):
                        ranges = np.array([ranges])
                        rss = np.array([rss])
                        stdev = np.array([stdev])

                    # 构造测量矩阵 (3, N)
                    meas = np.zeros((3, len(ranges)))
                    meas[0, :] = ranges  # 距离
                    meas[1, :] = np.maximum((stdev / 100.0) ** 2, 0.0025)  # 方差
                    # RSS转幅度
                    linear_power = 10 ** (rss / 10.0)
                    meas[2, :] = np.sqrt(linear_power / 1000.0)

                    step_measurements.append(meas)
                    total_meas += len(ranges)

                except Exception as e:
                    print(f"   ⚠ 读取失败 {csv_files[0].name}: {e}")
                    step_measurements.append(np.zeros((3, 0)))

        measurements.append(step_measurements)

        if (step_idx + 1) % 10 == 0:
            print(f"   进度: {step_idx + 1}/{num_steps}")

    print(f"   ✓ 完成，总测量数: {total_meas}")

    return true_trajectory, anchors, measurements, anchor_ids


def save_to_mat(true_trajectory, anchors, measurements, anchor_ids, output_file):
    """保存为MAT格式"""
    print(f"\n保存到 {output_file}...")

    num_steps = true_trajectory.shape[1]
    num_sensors = len(anchor_ids)

    # 构造dataVA
    data_va = []
    for i in range(num_sensors):
        data_va.append({
            'positions': anchors[:, i:i+1],
            'visibility': np.ones((1, num_steps))
        })

    # 构造measurements cell
    meas_cell = np.empty((num_steps, num_sensors), dtype=object)
    for i in range(num_steps):
        for j in range(num_sensors):
            meas_cell[i, j] = measurements[i][j]

    # 保存
    sio.savemat(output_file, {
        'trueTrajectory': true_trajectory,
        'dataVA': np.array(data_va, dtype=object).reshape(-1, 1),
        'estimated_measurements_cell': meas_cell,
        'anchor_ids': anchor_ids
    })

    print(f"✓ 已保存")


def main():
    parser = argparse.ArgumentParser(description='从raw_data提取UWB数据')
    parser.add_argument('--location', default='location0',
                       choices=['location0', 'location1', 'location2', 'location3'])
    parser.add_argument('--num-steps', type=int, default=50,
                       help='提取的步数')
    parser.add_argument('--output', default=None)

    args = parser.parse_args()

    if args.output is None:
        args.output = f'uwb_{args.location}_{args.num_steps}steps.mat'

    # 提取数据
    traj, anchors, meas, ids = extract_from_raw_data(args.location, args.num_steps)

    # 保存
    save_to_mat(traj, anchors, meas, ids, args.output)

    print(f"\n" + "=" * 70)
    print("使用方法:")
    print(f"  python testbed.py --mode gnn --steps {traj.shape[1]} \\")
    print(f"                    --load-measurements {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()
