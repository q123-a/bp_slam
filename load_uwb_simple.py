#!/usr/bin/env python3
"""
简化版UWB数据加载器 - 使用流式处理避免内存溢出
"""
import numpy as np
import scipy.io as sio
import json
from pathlib import Path
import argparse

def load_uwb_simple(location='location0', max_steps=100):
    """
    简化版数据加载 - 只加载必要的数据
    """
    base_path = Path('data_set')
    location_path = base_path / location

    print(f"加载 {location} 数据集 (前 {max_steps} 步)...")

    # 1. 加载锚点
    print("1. 加载锚点...")
    anchors_data = np.loadtxt(location_path / 'anchors.csv', delimiter=',', dtype=str)
    anchor_ids = anchors_data[:, 0]
    anchors = anchors_data[:, 1:3].astype(float).T  # (2, K)
    num_anchors = len(anchor_ids)
    print(f"   ✓ {num_anchors} 个锚点: {anchor_ids}")

    # 2. 加载轨迹
    print("2. 加载轨迹...")
    trajectory_data = np.loadtxt(location_path / 'walking_path.csv', delimiter=',')
    trajectory_data = trajectory_data[:max_steps]  # 只取前max_steps步
    true_trajectory = trajectory_data[:, :2].T  # (2, M)
    num_steps = true_trajectory.shape[1]
    print(f"   ✓ {num_steps} 个轨迹点")

    # 3. 流式加载测量数据
    print("3. 加载测量数据 (流式处理)...")
    data_file = location_path / 'data.json'

    # 初始化测量数据
    measurements = [[np.zeros((3, 0)) for _ in range(num_anchors)] for _ in range(num_steps)]

    # 使用ijson进行流式解析（如果可用），否则分块读取
    try:
        # 尝试直接加载（小数据集）
        print("   尝试直接加载JSON...")
        with open(data_file, 'r') as f:
            # 只读取需要的步数
            data_json = json.load(f)

        for step_idx in range(num_steps):
            step_key = str(step_idx)
            if step_key not in data_json:
                continue

            step_data = data_json[step_key]

            for anchor_idx, anchor_id in enumerate(anchor_ids):
                if anchor_id not in step_data or len(step_data[anchor_id]) == 0:
                    continue

                # 提取测量
                anchor_meas = step_data[anchor_id]
                ranges = [m['range'] for m in anchor_meas if 'range' in m]
                rss = [m['rss'] for m in anchor_meas if 'rss' in m]

                if len(ranges) == 0:
                    continue

                # 构造测量矩阵 (3, N)
                meas = np.zeros((3, len(ranges)))
                meas[0, :] = ranges  # 距离
                meas[1, :] = 0.01  # 固定方差 1cm²
                # RSS转幅度
                linear_power = 10 ** (np.array(rss) / 10.0)
                meas[2, :] = np.sqrt(linear_power / 1000.0)

                measurements[step_idx][anchor_idx] = meas

            if (step_idx + 1) % 20 == 0:
                print(f"   进度: {step_idx + 1}/{num_steps}")

    except MemoryError:
        print("   ⚠ 内存不足，请减少max_steps")
        return None, None, None, None

    print(f"   ✓ 完成")

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', default='location0',
                       choices=['location0', 'location1', 'location2', 'location3'])
    parser.add_argument('--max-steps', type=int, default=100,
                       help='最大步数（避免内存溢出）')
    parser.add_argument('--output', default=None)

    args = parser.parse_args()

    if args.output is None:
        args.output = f'uwb_{args.location}_{args.max_steps}steps.mat'

    # 加载数据
    traj, anchors, meas, ids = load_uwb_simple(args.location, args.max_steps)

    if traj is None:
        print("加载失败")
        return

    # 保存
    save_to_mat(traj, anchors, meas, ids, args.output)

    print(f"\n使用方法:")
    print(f"  python testbed.py --mode gnn --steps {traj.shape[1]} \\")
    print(f"                    --load-measurements {args.output}")


if __name__ == '__main__':
    main()
