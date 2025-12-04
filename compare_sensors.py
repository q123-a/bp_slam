"""
对比 GNN 和 BP 在两个传感器上的 OSPA 地图误差
创建两张图：一张对比 Sensor 1，一张对比 Sensor 2
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path
from bp_slam.visualization.plotting import ospa_dist


def load_results(bp_file='results/results_bp.npz', gnn_file='results/results_gnn.npz'):
    """
    加载 BP 和 GNN 的结果文件

    返回:
        bp_data: BP 结果数据
        gnn_data: GNN 结果数据
    """
    bp_data = np.load(bp_file, allow_pickle=True)
    gnn_data = np.load(gnn_file, allow_pickle=True)

    return bp_data, gnn_data


def load_true_anchors(data_file='scenarioCleanM2_new901.mat'):
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


def plot_ospa_comparison(bp_data, gnn_data, data_va, save_dir='results'):
    """
    绘制 OSPA 地图误差对比图（每个传感器一张独立的图）

    参数:
        bp_data: BP 结果数据
        gnn_data: GNN 结果数据
        data_va: 真实锚点数据
        save_dir: 保存目录
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # 提取参数
    bp_params = bp_data['parameters'].item()
    gnn_params = gnn_data['parameters'].item()
    detection_threshold = bp_params.get('detectionThreshold', 0.5)

    # 提取锚点数据
    bp_estimated_anchors = bp_data['estimated_anchors']
    gnn_estimated_anchors = gnn_data['estimated_anchors']

    num_sensors = len(bp_estimated_anchors)
    num_steps = bp_data['estimated_trajectory'].shape[1]

    # MATLAB 风格颜色
    mycolors = np.array([
        [0.66, 0.00, 0.00],  # 深红色 - 传感器1
        [0.00, 0.30, 0.70],  # 深蓝色 - 传感器2
    ])

    # 为每个传感器创建独立的图
    for sensor_idx in range(num_sensors):
        print(f"\n处理 Sensor {sensor_idx + 1}...")

        # 获取真实锚点位置
        true_anchor_positions = data_va[sensor_idx]['positions']

        # 计算 BP 的 OSPA 误差
        bp_ospa_errors = compute_ospa_errors(
            bp_estimated_anchors[sensor_idx],
            true_anchor_positions,
            detection_threshold,
            num_steps
        )

        # 计算 GNN 的 OSPA 误差
        gnn_ospa_errors = compute_ospa_errors(
            gnn_estimated_anchors[sensor_idx],
            true_anchor_positions,
            detection_threshold,
            num_steps
        )

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制 OSPA 误差曲线
        time_steps = np.arange(1, num_steps + 1)
        ax.plot(time_steps, bp_ospa_errors, '-',
                color='blue', linewidth=1.5, label='BP', alpha=0.8)
        ax.plot(time_steps, gnn_ospa_errors, '-',
                color='red', linewidth=1.5, label='GNN', alpha=0.8)

        # 设置标签和标题
        ax.set_xlabel('Trajectory steps', fontsize=14)
        ax.set_ylabel('OSPA map error [m]', fontsize=14)
        ax.set_title(f'Sensor {sensor_idx + 1}: OSPA Distance (GNN vs BP)',
                    fontsize=16, fontweight='bold')

        # 图例
        ax.legend(loc='best', fontsize=12, framealpha=0.9)

        # 网格
        ax.grid(True, alpha=0.3, linestyle='--')

        # 设置坐标轴范围
        ax.set_xlim([0, num_steps])
        # 设置 y 轴范围，留出更多空间
        y_max = max(np.max(bp_ospa_errors), np.max(gnn_ospa_errors))
        ax.set_ylim([0, y_max * 1.15])

        plt.tight_layout()

        # 保存图形
        save_path = save_dir / f'figure_sensor{sensor_idx + 1}_ospa_comparison.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Sensor {sensor_idx + 1} OSPA 对比图已保存: {save_path}")

        plt.close()

        # 打印统计信息
        print(f"  BP  - 平均 OSPA 误差: {np.mean(bp_ospa_errors):.4f} m")
        print(f"  GNN - 平均 OSPA 误差: {np.mean(gnn_ospa_errors):.4f} m")
        improvement = (np.mean(bp_ospa_errors) - np.mean(gnn_ospa_errors)) / np.mean(bp_ospa_errors) * 100
        print(f"  改进: {improvement:+.2f}%")


def plot_combined_ospa_comparison(bp_data, gnn_data, data_va, save_dir='results'):
    """
    绘制两个传感器的 OSPA 误差对比图（合并在一张图中）

    参数:
        bp_data: BP 结果数据
        gnn_data: GNN 结果数据
        data_va: 真实锚点数据
        save_dir: 保存目录
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # 提取参数
    bp_params = bp_data['parameters'].item()
    detection_threshold = bp_params.get('detectionThreshold', 0.5)

    # 提取锚点数据
    bp_estimated_anchors = bp_data['estimated_anchors']
    gnn_estimated_anchors = gnn_data['estimated_anchors']

    num_sensors = len(bp_estimated_anchors)
    num_steps = bp_data['estimated_trajectory'].shape[1]

    # 创建图形（两个子图）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 为每个传感器绘制子图
    for sensor_idx in range(num_sensors):
        ax = axes[sensor_idx]

        # 获取真实锚点位置
        true_anchor_positions = data_va[sensor_idx]['positions']

        # 计算 BP 的 OSPA 误差
        bp_ospa_errors = compute_ospa_errors(
            bp_estimated_anchors[sensor_idx],
            true_anchor_positions,
            detection_threshold,
            num_steps
        )

        # 计算 GNN 的 OSPA 误差
        gnn_ospa_errors = compute_ospa_errors(
            gnn_estimated_anchors[sensor_idx],
            true_anchor_positions,
            detection_threshold,
            num_steps
        )

        # 绘制 OSPA 误差曲线
        time_steps = np.arange(1, num_steps + 1)
        ax.plot(time_steps, bp_ospa_errors, '-',
                color='blue', linewidth=1.5, label='BP', alpha=0.8)
        ax.plot(time_steps, gnn_ospa_errors, '-',
                color='red', linewidth=1.5, label='GNN', alpha=0.8)

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
        y_max = max(np.max(bp_ospa_errors), np.max(gnn_ospa_errors))
        ax.set_ylim([0, y_max * 1.15])

    plt.suptitle('OSPA Map Error: GNN vs BP', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # 保存图形
    save_path = save_dir / 'ospa_comparison_combined.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 合并 OSPA 对比图已保存: {save_path}")

    plt.close()


def print_statistics(bp_data, gnn_data, data_va):
    """
    打印详细的统计信息

    参数:
        bp_data: BP 结果数据
        gnn_data: GNN 结果数据
        data_va: 真实锚点数据
    """
    # 提取参数
    bp_params = bp_data['parameters'].item()
    detection_threshold = bp_params.get('detectionThreshold', 0.5)

    # 提取锚点数据
    bp_estimated_anchors = bp_data['estimated_anchors']
    gnn_estimated_anchors = gnn_data['estimated_anchors']

    num_sensors = len(bp_estimated_anchors)
    num_steps = bp_data['estimated_trajectory'].shape[1]

    print("\n" + "=" * 70)
    print("OSPA 地图误差统计")
    print("=" * 70)

    for sensor_idx in range(num_sensors):
        # 获取真实锚点位置
        true_anchor_positions = data_va[sensor_idx]['positions']

        # 计算 BP 的 OSPA 误差
        bp_ospa_errors = compute_ospa_errors(
            bp_estimated_anchors[sensor_idx],
            true_anchor_positions,
            detection_threshold,
            num_steps
        )

        # 计算 GNN 的 OSPA 误差
        gnn_ospa_errors = compute_ospa_errors(
            gnn_estimated_anchors[sensor_idx],
            true_anchor_positions,
            detection_threshold,
            num_steps
        )

        print(f"\nSensor {sensor_idx + 1}:")
        print(f"  BP:")
        print(f"    平均 OSPA 误差: {np.mean(bp_ospa_errors):.4f} m")
        print(f"    标准差:         {np.std(bp_ospa_errors):.4f} m")
        print(f"    最大 OSPA 误差: {np.max(bp_ospa_errors):.4f} m")
        print(f"    最小 OSPA 误差: {np.min(bp_ospa_errors):.4f} m")

        print(f"  GNN:")
        print(f"    平均 OSPA 误差: {np.mean(gnn_ospa_errors):.4f} m")
        print(f"    标准差:         {np.std(gnn_ospa_errors):.4f} m")
        print(f"    最大 OSPA 误差: {np.max(gnn_ospa_errors):.4f} m")
        print(f"    最小 OSPA 误差: {np.min(gnn_ospa_errors):.4f} m")

        improvement = (np.mean(bp_ospa_errors) - np.mean(gnn_ospa_errors)) / np.mean(bp_ospa_errors) * 100
        print(f"  改进: {improvement:+.2f}%")

    print("=" * 70)


if __name__ == '__main__':
    import sys

    # 默认文件路径
    bp_file = 'results/results_bp.npz'
    gnn_file = 'results/results_gnn.npz'
    data_file = 'scenarioCleanM2_new901.mat'

    # 检查命令行参数
    if len(sys.argv) > 2:
        bp_file = sys.argv[1]
        gnn_file = sys.argv[2]
    if len(sys.argv) > 3:
        data_file = sys.argv[3]

    # 检查文件是否存在
    if not Path(bp_file).exists():
        print(f"错误: BP 结果文件不存在: {bp_file}")
        sys.exit(1)

    if not Path(gnn_file).exists():
        print(f"错误: GNN 结果文件不存在: {gnn_file}")
        sys.exit(1)

    if not Path(data_file).exists():
        print(f"错误: 数据文件不存在: {data_file}")
        sys.exit(1)

    # 加载数据
    print("=" * 70)
    print("加载数据...")
    print("=" * 70)
    bp_data, gnn_data = load_results(bp_file, gnn_file)
    data_va = load_true_anchors(data_file)
    print(f"✓ 成功加载 BP 和 GNN 结果")
    print(f"✓ 成功加载真实锚点数据 ({len(data_va)} 个传感器)")

    # 绘制独立的 OSPA 对比图
    print("\n" + "=" * 70)
    print("生成 OSPA 误差对比图...")
    print("=" * 70)
    plot_ospa_comparison(bp_data, gnn_data, data_va)

    # 绘制合并的 OSPA 对比图
    plot_combined_ospa_comparison(bp_data, gnn_data, data_va)

    # 打印统计信息
    print_statistics(bp_data, gnn_data, data_va)

    print("\n✓ 完成!")
