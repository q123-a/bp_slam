"""
对比硬OSPA和软OSPA的效果
比较使用硬阈值和软权重计算OSPA误差的差异
"""

import numpy as np
import matplotlib.pyplot as plt
from bp_slam.visualization.visualizer import BPSLAMVisualizer
import scipy.io as sio

def compare_ospa_methods(results_file='results/results_gnn.npz',
                        data_file='scenarioCleanM2_new_1500.mat'):
    """
    对比硬OSPA和软OSPA的效果
    """
    print("=" * 70)
    print("对比硬OSPA vs 软OSPA")
    print("=" * 70)
    print()

    # 加载结果数据
    print(f"1. 加载结果数据: {results_file}...")
    results = np.load(results_file, allow_pickle=True)
    estimated_trajectory = results['estimated_trajectory']
    true_trajectory = results['true_trajectory']
    estimated_anchors = results['estimated_anchors']
    parameters = results['parameters'].item()

    num_steps = estimated_trajectory.shape[1]
    print(f"   时间步数: {num_steps}")

    # 加载真实锚点数据
    print(f"2. 加载场景数据: {data_file}...")
    mat_data = sio.loadmat(data_file)
    data_va_raw = mat_data['dataVA'][:, 0]
    num_sensors = len(data_va_raw)

    data_va = []
    for sensor in range(num_sensors):
        sensor_data = {
            'positions': data_va_raw[sensor]['positions'][0, 0],
        }
        data_va.append(sensor_data)
    print(f"   传感器数量: {num_sensors}")
    print()

    # 创建可视化器
    visualizer = BPSLAMVisualizer(output_dir='results', mode='gnn')

    # 计算硬OSPA
    print("3. 计算硬OSPA（使用硬阈值）...")
    fig_hard, ospa_hard = visualizer.plot_ospa_error(
        true_trajectory, estimated_trajectory, estimated_anchors,
        data_va, parameters, use_soft_ospa=False
    )
    plt.close(fig_hard)

    # 计算软OSPA
    print("4. 计算软OSPA（使用存在概率权重）...")
    fig_soft, ospa_soft = visualizer.plot_ospa_error(
        true_trajectory, estimated_trajectory, estimated_anchors,
        data_va, parameters, use_soft_ospa=True
    )
    plt.close(fig_soft)

    # 对比分析
    print()
    print("=" * 70)
    print("对比分析")
    print("=" * 70)

    for sensor in range(num_sensors):
        print(f"\n传感器 {sensor + 1}:")

        # 计算统计量
        hard_mean = np.mean(ospa_hard[sensor, :])
        soft_mean = np.mean(ospa_soft[sensor, :])
        hard_std = np.std(ospa_hard[sensor, :])
        soft_std = np.std(ospa_soft[sensor, :])
        hard_max = np.max(ospa_hard[sensor, :])
        soft_max = np.max(ospa_soft[sensor, :])

        # 计算尖峰数量（误差突然增加超过1m）
        hard_diff = np.abs(np.diff(ospa_hard[sensor, :]))
        soft_diff = np.abs(np.diff(ospa_soft[sensor, :]))
        hard_spikes = np.sum(hard_diff > 1.0)
        soft_spikes = np.sum(soft_diff > 1.0)

        print(f"  硬OSPA: 均值={hard_mean:.4f}m, 标准差={hard_std:.4f}m, 最大值={hard_max:.4f}m, 尖峰数={hard_spikes}")
        print(f"  软OSPA: 均值={soft_mean:.4f}m, 标准差={soft_std:.4f}m, 最大值={soft_max:.4f}m, 尖峰数={soft_spikes}")
        print(f"  改进: 标准差减少 {(hard_std-soft_std)/hard_std*100:.1f}%, 尖峰减少 {hard_spikes-soft_spikes} 个")

    # 绘制对比图
    print()
    print("5. 生成对比图...")
    fig, axes = plt.subplots(num_sensors, 1, figsize=(14, 5*num_sensors))
    if num_sensors == 1:
        axes = [axes]

    for sensor in range(num_sensors):
        ax = axes[sensor]

        # 绘制硬OSPA和软OSPA
        ax.plot(range(1, num_steps + 1), ospa_hard[sensor, :],
               '-', color='red', linewidth=1.5, alpha=0.7,
               label=f'硬OSPA (阈值={parameters.get("detectionThreshold", 0.5)})')
        ax.plot(range(1, num_steps + 1), ospa_soft[sensor, :],
               '-', color='blue', linewidth=1.5, alpha=0.7,
               label='软OSPA (存在概率权重)')

        # 标记尖峰位置
        hard_diff = np.abs(np.diff(ospa_hard[sensor, :]))
        spike_indices = np.where(hard_diff > 1.0)[0]
        if len(spike_indices) > 0:
            ax.scatter(spike_indices + 1, ospa_hard[sensor, spike_indices],
                      color='red', marker='x', s=100, zorder=5,
                      label=f'硬OSPA尖峰 ({len(spike_indices)}个)')

        ax.set_xlabel('时间步', fontsize=12)
        ax.set_ylabel('OSPA误差 [m]', fontsize=12)
        ax.set_title(f'传感器 {sensor + 1}: 硬OSPA vs 软OSPA', fontsize=14)
        ax.set_xlim([0, num_steps])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    output_path = 'results/ospa_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ 对比图已保存: {output_path}")

    # 保存软OSPA图
    fig_soft.savefig('results/figure2_ospa_error_gnn_soft.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ 软OSPA图已保存: results/figure2_ospa_error_gnn_soft.png")

    plt.show()

    print()
    print("=" * 70)
    print("结论")
    print("=" * 70)
    print("软OSPA通过使用存在概率权重而非硬阈值，可以：")
    print("1. 消除锚点数量突变导致的尖峰")
    print("2. 提供更平滑、更稳定的误差曲线")
    print("3. 更准确地反映估计质量的连续变化")
    print("=" * 70)


if __name__ == '__main__':
    import sys

    results_file = 'results/results_gnn.npz'
    if len(sys.argv) > 1:
        results_file = sys.argv[1]

    compare_ospa_methods(results_file=results_file)
