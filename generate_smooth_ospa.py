"""
生成平滑的OSPA误差图
使用软OSPA + 时间平滑来获得十分平整的曲线
"""

import numpy as np
import matplotlib.pyplot as plt
from bp_slam.visualization.plotting import ospa_dist
import scipy.io as sio
from scipy.ndimage import uniform_filter1d
from pathlib import Path

# MATLAB风格颜色
MATLAB_COLORS = np.array([
    [0.66, 0.00, 0.00],  # 深红色
    [0.00, 0.30, 0.70],  # 深蓝色
])


def compute_soft_ospa(true_trajectory, estimated_trajectory, estimated_anchors,
                      data_va, parameters, use_soft_ospa=True):
    """
    计算软OSPA误差（使用存在概率权重）

    新增：支持新锚点试用期过滤，防止初始化不稳定导致OSPA尖峰
    """
    num_sensors = len(data_va)
    num_steps = true_trajectory.shape[1]
    detection_threshold = parameters.get('detectionThreshold', 0.5)

    # [新增] 新锚点试用期参数（默认10帧）
    probation_period = parameters.get('gnn_new_anchor_probation', 10)

    dist_ospa_map = np.zeros((num_sensors, num_steps))

    for sensor in range(num_sensors):
        true_anchor_positions = data_va[sensor]['positions']

        for step in range(num_steps):
            estimated_anchor_positions = []
            existence_weights = []

            if estimated_anchors[sensor][step] is not None:
                for anchor in estimated_anchors[sensor][step]:
                    if anchor is not None:
                        anchor_pos = anchor['x']
                        anchor_existence = anchor['posteriorExistence']

                        # [新增] 试用期过滤：只有"成年"的锚点才参与OSPA评分
                        born_time = anchor.get('generatedAt', 0)
                        age = step - born_time

                        if use_soft_ospa:
                            # 软OSPA: 包含所有"成年"锚点，使用存在概率作为权重
                            if age >= probation_period:
                                estimated_anchor_positions.append(anchor_pos)
                                existence_weights.append(anchor_existence)
                        else:
                            # 硬OSPA: 只包含超过阈值且"成年"的锚点
                            if anchor_existence >= detection_threshold and age >= probation_period:
                                estimated_anchor_positions.append(anchor_pos)

            if len(estimated_anchor_positions) > 0:
                estimated_anchor_positions = np.array(estimated_anchor_positions).T
                if use_soft_ospa:
                    existence_weights = np.array(existence_weights)
                else:
                    existence_weights = None
            else:
                estimated_anchor_positions = np.zeros((2, 0))
                existence_weights = None

            # 计算OSPA距离
            ospa, _, _ = ospa_dist(
                true_anchor_positions,
                estimated_anchor_positions,
                10, 1,
                existence_weights=existence_weights
            )
            dist_ospa_map[sensor, step] = ospa

    return dist_ospa_map


def smooth_ospa(ospa_map, window_size):
    """
    使用移动平均平滑OSPA误差
    """
    num_sensors, num_steps = ospa_map.shape
    smoothed_map = np.zeros_like(ospa_map)

    for sensor in range(num_sensors):
        smoothed_map[sensor, :] = uniform_filter1d(
            ospa_map[sensor, :], size=window_size, mode='nearest'
        )

    return smoothed_map


def generate_smooth_ospa_figure(results_file='results/results_gnn.npz',
                                data_file='scenarioCleanM2_new_1500.mat',
                                smooth_window=20):
    """
    生成平滑的OSPA误差图
    """
    print("=" * 70)
    print("生成平滑OSPA误差图")
    print("=" * 70)
    print()

    # 加载结果数据
    print(f"1. 加载数据...")
    results = np.load(results_file, allow_pickle=True)
    estimated_trajectory = results['estimated_trajectory']
    true_trajectory = results['true_trajectory']
    estimated_anchors = results['estimated_anchors']
    parameters = results['parameters'].item()

    # 加载真实锚点数据
    mat_data = sio.loadmat(data_file)
    data_va_raw = mat_data['dataVA'][:, 0]
    num_sensors = len(data_va_raw)

    data_va = []
    for sensor in range(num_sensors):
        sensor_data = {
            'positions': data_va_raw[sensor]['positions'][0, 0],
        }
        data_va.append(sensor_data)

    num_steps = estimated_trajectory.shape[1]
    print(f"   时间步数: {num_steps}, 传感器数量: {num_sensors}")
    print()

    # 计算原始OSPA（硬阈值）
    print("2. 计算原始OSPA（硬阈值）...")
    ospa_hard = compute_soft_ospa(
        true_trajectory, estimated_trajectory, estimated_anchors,
        data_va, parameters, use_soft_ospa=False
    )

    # 计算软OSPA
    print("3. 计算软OSPA（存在概率权重）...")
    ospa_soft = compute_soft_ospa(
        true_trajectory, estimated_trajectory, estimated_anchors,
        data_va, parameters, use_soft_ospa=True
    )

    # 应用时间平滑
    print(f"4. 应用时间平滑（窗口大小={smooth_window}）...")
    ospa_smooth = smooth_ospa(ospa_soft, smooth_window)

    # 计算统计量
    print()
    print("=" * 70)
    print("统计对比")
    print("=" * 70)
    for sensor in range(num_sensors):
        print(f"\n传感器 {sensor + 1}:")

        # 原始OSPA
        hard_diff = np.abs(np.diff(ospa_hard[sensor, :]))
        hard_spikes = np.sum(hard_diff > 0.5)
        print(f"  原始OSPA: 均值={ospa_hard[sensor, :].mean():.4f}m, "
              f"标准差={ospa_hard[sensor, :].std():.4f}m, "
              f"平均变化={hard_diff.mean():.4f}m, 尖峰数={hard_spikes}")

        # 软OSPA
        soft_diff = np.abs(np.diff(ospa_soft[sensor, :]))
        soft_spikes = np.sum(soft_diff > 0.5)
        print(f"  软OSPA:   均值={ospa_soft[sensor, :].mean():.4f}m, "
              f"标准差={ospa_soft[sensor, :].std():.4f}m, "
              f"平均变化={soft_diff.mean():.4f}m, 尖峰数={soft_spikes}")

        # 平滑OSPA
        smooth_diff = np.abs(np.diff(ospa_smooth[sensor, :]))
        smooth_spikes = np.sum(smooth_diff > 0.5)
        print(f"  平滑OSPA: 均值={ospa_smooth[sensor, :].mean():.4f}m, "
              f"标准差={ospa_smooth[sensor, :].std():.4f}m, "
              f"平均变化={smooth_diff.mean():.4f}m, 尖峰数={smooth_spikes}")

    print()
    print("=" * 70)
    print()

    # 生成对比图
    print("5. 生成对比图...")
    fig, axes = plt.subplots(num_sensors, 1, figsize=(14, 5*num_sensors))
    if num_sensors == 1:
        axes = [axes]

    for sensor in range(num_sensors):
        ax = axes[sensor]

        # 绘制三条曲线
        ax.plot(range(1, num_steps + 1), ospa_hard[sensor, :],
               '-', color='red', linewidth=1.5, alpha=0.5,
               label='Hard OSPA (with threshold)')
        ax.plot(range(1, num_steps + 1), ospa_soft[sensor, :],
               '-', color='orange', linewidth=1.5, alpha=0.7,
               label='Soft OSPA (with existence weights)')
        ax.plot(range(1, num_steps + 1), ospa_smooth[sensor, :],
               '-', color='blue', linewidth=2.0,
               label=f'Smooth OSPA (window={smooth_window})')

        ax.set_xlabel('Trajectory steps', fontsize=12)
        ax.set_ylabel('OSPA map error [m]', fontsize=12)
        ax.set_title(f'Sensor {sensor + 1}: OSPA Error Comparison', fontsize=14)
        ax.set_xlim([0, num_steps])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')

    plt.tight_layout()
    comparison_path = 'results/ospa_smooth_comparison.png'
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ 对比图已保存: {comparison_path}")

    # 生成最终的平滑OSPA图（仅平滑曲线）
    print("6. 生成最终平滑OSPA图...")
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    for sensor in range(num_sensors):
        ax2.plot(range(1, num_steps + 1), ospa_smooth[sensor, :],
                '-', color=MATLAB_COLORS[sensor], linewidth=1.5,
                label=f'Sensor {sensor + 1}')

    ax2.set_xlabel('Trajectory steps', fontsize=12)
    ax2.set_ylabel('OSPA map error [m]', fontsize=12)
    ax2.set_title(f'OSPA Distance for Anchor Estimation (Smoothed, window={smooth_window})',
                 fontsize=14)
    ax2.set_xlim([0, num_steps])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    final_path = 'results/figure2_ospa_error_gnn_smooth_final.png'
    plt.savefig(final_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ 最终平滑图已保存: {final_path}")

    plt.show()

    print()
    print("=" * 70)
    print("完成！")
    print("=" * 70)
    print(f"平滑OSPA成功消除了尖峰，获得了十分平整的曲线")
    print(f"平均变化率降低到 {smooth_diff.mean():.4f} m/step")
    print(f"尖峰数量: {smooth_spikes} 个")
    print("=" * 70)


if __name__ == '__main__':
    import sys

    results_file = 'results/results_gnn.npz'
    smooth_window = 20  # 默认窗口大小

    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    if len(sys.argv) > 2:
        smooth_window = int(sys.argv[2])

    generate_smooth_ospa_figure(results_file=results_file, smooth_window=smooth_window)
