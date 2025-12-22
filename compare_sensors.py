"""
对比 GNN 和 BP 在两个传感器上的 OSPA 地图误差
创建两张图：一张对比 Sensor 1，一张对比 Sensor 2
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path
from bp_slam.visualization.plotting import ospa_dist


def load_results(file1='results/results_bp.npz', file2='results/results_gnn.npz'):
    """
    加载两个结果文件

    参数:
        file1: 第一个结果文件路径
        file2: 第二个结果文件路径

    返回:
        data1: 第一个结果数据
        data2: 第二个结果数据
    """
    data1 = np.load(file1, allow_pickle=True)
    data2 = np.load(file2, allow_pickle=True)

    return data1, data2


def load_true_anchors(data_file='scenarioCleanM2_new_1500.mat'):
    """
    加载真实锚点数据

    返回:
        data_va: 真实锚点数据列表
    """
    mat_data = sio.loadmat(data_file)
    data_va_raw = mat_data['dataVA'][:, 0]
    num_sensors = len(data_va_raw)

    data_va = []
    for sensor in range(num_sensors):
        sensor_data = {
            'positions': data_va_raw[sensor]['positions'][0, 0],
        }
        data_va.append(sensor_data)

    return data_va


def compute_ospa_errors(estimated_anchors, true_anchor_positions, detection_threshold, num_steps):
    """
    计算 OSPA 地图误差

    参数:
        estimated_anchors: 估计的锚点数据 (每个时间步的锚点列表)
        true_anchor_positions: (2, N) 真实锚点位置
        detection_threshold: 检测阈值
        num_steps: 时间步数

    返回:
        ospa_errors: (T,) OSPA 误差数组
    """
    ospa_errors = np.zeros(num_steps)

    for step in range(num_steps):
        # 提取估计的锚点位置
        if estimated_anchors[step] is not None:
            num_anchors_step = len(estimated_anchors[step])
            estimated_anchor_positions = []

            for anchor in range(num_anchors_step):
                if estimated_anchors[step][anchor] is not None:
                    anchor_pos = estimated_anchors[step][anchor]['x']
                    anchor_existence = estimated_anchors[step][anchor]['posteriorExistence']

                    if anchor_existence >= detection_threshold:
                        estimated_anchor_positions.append(anchor_pos)

            if len(estimated_anchor_positions) > 0:
                estimated_anchor_positions = np.array(estimated_anchor_positions).T
            else:
                estimated_anchor_positions = np.zeros((2, 0))
        else:
            estimated_anchor_positions = np.zeros((2, 0))

        # 计算 OSPA 距离
        ospa, _, _ = ospa_dist(true_anchor_positions, estimated_anchor_positions, 10, 1)
        ospa_errors[step] = ospa

    return ospa_errors


def plot_ospa_comparison(data1, data2, data_va, label1='Method 1', label2='Method 2', save_dir='results'):
    """
    绘制 OSPA 地图误差对比图（每个传感器一张独立的图）

    参数:
        data1: 第一个方法的结果数据
        data2: 第二个方法的结果数据
        data_va: 真实锚点数据
        label1: 第一个方法的标签
        label2: 第二个方法的标签
        save_dir: 保存目录
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # 提取参数
    params1 = data1['parameters'].item()
    detection_threshold = params1.get('detectionThreshold', 0.5)

    # 提取锚点数据
    estimated_anchors1 = data1['estimated_anchors']
    estimated_anchors2 = data2['estimated_anchors']

    num_sensors = len(estimated_anchors1)
    num_steps = data1['estimated_trajectory'].shape[1]

    # 为每个传感器创建独立的图
    for sensor_idx in range(num_sensors):
        print(f"\n处理 Sensor {sensor_idx + 1}...")

        # 获取真实锚点位置
        true_anchor_positions = data_va[sensor_idx]['positions']

        # 计算第一个方法的 OSPA 误差
        ospa_errors1 = compute_ospa_errors(
            estimated_anchors1[sensor_idx],
            true_anchor_positions,
            detection_threshold,
            num_steps
        )

        # 计算第二个方法的 OSPA 误差
        ospa_errors2 = compute_ospa_errors(
            estimated_anchors2[sensor_idx],
            true_anchor_positions,
            detection_threshold,
            num_steps
        )

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制 OSPA 误差曲线
        time_steps = np.arange(1, num_steps + 1)
        ax.plot(time_steps, ospa_errors1, '-',
                color='blue', linewidth=1.5, label=label1, alpha=0.8)
        ax.plot(time_steps, ospa_errors2, '-',
                color='red', linewidth=1.5, label=label2, alpha=0.8)

        # 设置标签和标题
        ax.set_xlabel('Trajectory steps', fontsize=14)
        ax.set_ylabel('OSPA map error [m]', fontsize=14)
        ax.set_title(f'Sensor {sensor_idx + 1}: OSPA Distance ({label2} vs {label1})',
                    fontsize=16, fontweight='bold')

        # 图例
        ax.legend(loc='best', fontsize=12, framealpha=0.9)

        # 网格
        ax.grid(True, alpha=0.3, linestyle='--')

        # 设置坐标轴范围
        ax.set_xlim([0, num_steps])
        # 设置 y 轴范围，留出更多空间
        y_max = max(np.max(ospa_errors1), np.max(ospa_errors2))
        ax.set_ylim([0, y_max * 1.15])

        plt.tight_layout()

        # 保存图形（使用标签生成文件名）
        safe_label1 = label1.replace(' ', '_').replace('/', '_')
        safe_label2 = label2.replace(' ', '_').replace('/', '_')
        save_path = save_dir / f'sensor{sensor_idx + 1}_ospa_{safe_label1}_vs_{safe_label2}.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Sensor {sensor_idx + 1} OSPA 对比图已保存: {save_path}")

        plt.close()

        # 打印统计信息
        print(f"  {label1} - 平均 OSPA 误差: {np.mean(ospa_errors1):.4f} m")
        print(f"  {label2} - 平均 OSPA 误差: {np.mean(ospa_errors2):.4f} m")
        improvement = (np.mean(ospa_errors1) - np.mean(ospa_errors2)) / np.mean(ospa_errors1) * 100
        print(f"  改进: {improvement:+.2f}%")


def plot_combined_ospa_comparison(data1, data2, data_va, label1='Method 1', label2='Method 2', save_dir='results'):
    """
    绘制两个传感器的 OSPA 误差对比图（合并在一张图中）

    参数:
        data1: 第一个方法的结果数据
        data2: 第二个方法的结果数据
        data_va: 真实锚点数据
        label1: 第一个方法的标签
        label2: 第二个方法的标签
        save_dir: 保存目录
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # 提取参数
    params1 = data1['parameters'].item()
    detection_threshold = params1.get('detectionThreshold', 0.5)

    # 提取锚点数据
    estimated_anchors1 = data1['estimated_anchors']
    estimated_anchors2 = data2['estimated_anchors']

    num_sensors = len(estimated_anchors1)
    num_steps = data1['estimated_trajectory'].shape[1]

    # 创建图形（两个子图）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 为每个传感器绘制子图
    for sensor_idx in range(num_sensors):
        ax = axes[sensor_idx]

        # 获取真实锚点位置
        true_anchor_positions = data_va[sensor_idx]['positions']

        # 计算第一个方法的 OSPA 误差
        ospa_errors1 = compute_ospa_errors(
            estimated_anchors1[sensor_idx],
            true_anchor_positions,
            detection_threshold,
            num_steps
        )

        # 计算第二个方法的 OSPA 误差
        ospa_errors2 = compute_ospa_errors(
            estimated_anchors2[sensor_idx],
            true_anchor_positions,
            detection_threshold,
            num_steps
        )

        # 绘制 OSPA 误差曲线
        time_steps = np.arange(1, num_steps + 1)
        ax.plot(time_steps, ospa_errors1, '-',
                color='blue', linewidth=1.5, label=label1, alpha=0.8)
        ax.plot(time_steps, ospa_errors2, '-',
                color='red', linewidth=1.5, label=label2, alpha=0.8)

        # 设置标签和标题
        ax.set_xlabel('Trajectory steps', fontsize=12)
        ax.set_ylabel('OSPA map error [m]', fontsize=12)
        ax.set_title(f'Sensor {sensor_idx + 1}', fontsize=14, fontweight='bold')

        # 图例
        ax.legend(loc='best', fontsize=11, framealpha=0.9)

        # 网格
        ax.grid(True, alpha=0.3, linestyle='--')

        # 设置坐标轴范围
        ax.set_xlim([0, num_steps])
        # 设置 y 轴范围，留出更多空间
        y_max = max(np.max(ospa_errors1), np.max(ospa_errors2))
        ax.set_ylim([0, y_max * 1.15])

    plt.suptitle(f'OSPA Map Error: {label2} vs {label1}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # 保存图形（使用标签生成文件名）
    safe_label1 = label1.replace(' ', '_').replace('/', '_')
    safe_label2 = label2.replace(' ', '_').replace('/', '_')
    save_path = save_dir / f'ospa_combined_{safe_label1}_vs_{safe_label2}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 合并 OSPA 对比图已保存: {save_path}")

    plt.close()


def print_statistics(data1, data2, data_va, label1='Method 1', label2='Method 2'):
    """
    打印详细的统计信息

    参数:
        data1: 第一个方法的结果数据
        data2: 第二个方法的结果数据
        data_va: 真实锚点数据
        label1: 第一个方法的标签
        label2: 第二个方法的标签
    """
    # 提取参数
    params1 = data1['parameters'].item()
    detection_threshold = params1.get('detectionThreshold', 0.5)

    # 提取锚点数据
    estimated_anchors1 = data1['estimated_anchors']
    estimated_anchors2 = data2['estimated_anchors']

    num_sensors = len(estimated_anchors1)
    num_steps = data1['estimated_trajectory'].shape[1]

    print("\n" + "=" * 70)
    print(f"OSPA 地图误差统计 ({label1} vs {label2})")
    print("=" * 70)

    for sensor_idx in range(num_sensors):
        # 获取真实锚点位置
        true_anchor_positions = data_va[sensor_idx]['positions']

        # 计算第一个方法的 OSPA 误差
        ospa_errors1 = compute_ospa_errors(
            estimated_anchors1[sensor_idx],
            true_anchor_positions,
            detection_threshold,
            num_steps
        )

        # 计算第二个方法的 OSPA 误差
        ospa_errors2 = compute_ospa_errors(
            estimated_anchors2[sensor_idx],
            true_anchor_positions,
            detection_threshold,
            num_steps
        )

        print(f"\nSensor {sensor_idx + 1}:")
        print(f"  {label1}:")
        print(f"    平均 OSPA 误差: {np.mean(ospa_errors1):.4f} m")
        print(f"    标准差:         {np.std(ospa_errors1):.4f} m")
        print(f"    最大 OSPA 误差: {np.max(ospa_errors1):.4f} m")
        print(f"    最小 OSPA 误差: {np.min(ospa_errors1):.4f} m")

        print(f"  {label2}:")
        print(f"    平均 OSPA 误差: {np.mean(ospa_errors2):.4f} m")
        print(f"    标准差:         {np.std(ospa_errors2):.4f} m")
        print(f"    最大 OSPA 误差: {np.max(ospa_errors2):.4f} m")
        print(f"    最小 OSPA 误差: {np.min(ospa_errors2):.4f} m")

        improvement = (np.mean(ospa_errors1) - np.mean(ospa_errors2)) / np.mean(ospa_errors1) * 100
        print(f"  改进: {improvement:+.2f}%")

    print("=" * 70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='对比两个方法在传感器上的 OSPA 地图误差',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 默认对比 (BP vs GNN):
   python compare_sensors.py

2. 对比自定义文件:
   python compare_sensors.py --file1 results/results_bp.npz --file2 results/results_gnn.npz

3. 对比不同 GNN 版本:
   python compare_sensors.py --file1 results/results_gnn_v1.npz --file2 results/results_gnn_v2.npz --label1 "V1" --label2 "V2"

4. 指定场景数据文件:
   python compare_sensors.py --file1 results/results_bp.npz --file2 results/results_gnn.npz --data scenarioCleanM2_new_1500.mat

5. 对比 V2 vs V3:
   python compare_sensors.py --file1 results/results_gnn_v2.npz --file2 results/results_gnn_v3.npz --label1 "V2 (Implicit)" --label2 "V3 (Explicit)"
        """
    )

    parser.add_argument('--file1', type=str, default='results/results_bp.npz',
                        help='第一个结果文件路径 (默认: results/results_bp.npz)')
    parser.add_argument('--file2', type=str, default='results/results_gnn.npz',
                        help='第二个结果文件路径 (默认: results/results_gnn.npz)')
    parser.add_argument('--label1', type=str, default=None,
                        help='第一个方法的标签 (默认: 从文件名推断)')
    parser.add_argument('--label2', type=str, default=None,
                        help='第二个方法的标签 (默认: 从文件名推断)')
    parser.add_argument('--data', type=str, default='scenarioCleanM2_new_1500.mat',
                        help='场景数据文件路径 (默认: scenarioCleanM2_new_1500.mat)')
    parser.add_argument('--save-dir', type=str, default='results',
                        help='保存目录 (默认: results)')

    args = parser.parse_args()

    # 自动推断标签
    def infer_label(filepath):
        """从文件路径推断标签"""
        filename = Path(filepath).stem
        if filename.startswith('results_'):
            label = filename[8:]
        else:
            label = filename
        label = label.upper().replace('_', ' ')
        return label

    # 设置标签
    label1 = args.label1 if args.label1 else infer_label(args.file1)
    label2 = args.label2 if args.label2 else infer_label(args.file2)

    # 检查文件是否存在
    if not Path(args.file1).exists():
        print(f"错误: 结果文件不存在: {args.file1}")
        import sys
        sys.exit(1)

    if not Path(args.file2).exists():
        print(f"错误: 结果文件不存在: {args.file2}")
        import sys
        sys.exit(1)

    if not Path(args.data).exists():
        print(f"错误: 数据文件不存在: {args.data}")
        import sys
        sys.exit(1)

    # 加载数据
    print("=" * 70)
    print(f"对比 {label1} 和 {label2} 的 OSPA 地图误差")
    print("=" * 70)
    print("\n加载数据...")
    data1, data2 = load_results(args.file1, args.file2)
    data_va = load_true_anchors(args.data)
    print(f"✓ 成功加载结果文件")
    print(f"  - {label1}: {args.file1}")
    print(f"  - {label2}: {args.file2}")
    print(f"✓ 成功加载真实锚点数据 ({len(data_va)} 个传感器)")

    # 绘制独立的 OSPA 对比图
    print("\n" + "=" * 70)
    print("生成 OSPA 误差对比图...")
    print("=" * 70)
    plot_ospa_comparison(data1, data2, data_va, label1, label2, args.save_dir)

    # 绘制合并的 OSPA 对比图
    plot_combined_ospa_comparison(data1, data2, data_va, label1, label2, args.save_dir)

    # 打印统计信息
    print_statistics(data1, data2, data_va, label1, label2)

    print("\n✓ 完成!")
