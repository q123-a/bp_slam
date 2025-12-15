"""
测试不同的OSPA平滑方法，找到最平整的曲线
"""

import numpy as np
import matplotlib.pyplot as plt
from bp_slam.visualization.visualizer import BPSLAMVisualizer
import scipy.io as sio

def test_smoothing_methods(results_file='results/results_gnn.npz',
                           data_file='scenarioCleanM2_new_1500.mat'):
    """
    测试不同的平滑方法和窗口大小
    """
    print("=" * 70)
    print("测试OSPA平滑方法")
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

    # 创建可视化器
    visualizer = BPSLAMVisualizer(output_dir='results', mode='gnn')

    # 测试配置
    test_configs = [
        {'name': '原始（无平滑）', 'smooth_window': None, 'smooth_method': None},
        {'name': '移动平均 w=5', 'smooth_window': 5, 'smooth_method': 'moving_avg'},
        {'name': '移动平均 w=10', 'smooth_window': 10, 'smooth_method': 'moving_avg'},
        {'name': '移动平均 w=15', 'smooth_window': 15, 'smooth_method': 'moving_avg'},
        {'name': '移动平均 w=20', 'smooth_window': 20, 'smooth_method': 'moving_avg'},
        {'name': '指数平滑 w=10', 'smooth_window': 10, 'smooth_method': 'exponential'},
        {'name': '指数平滑 w=15', 'smooth_window': 15, 'smooth_method': 'exponential'},
        {'name': 'Savitzky-Golay w=11', 'smooth_window': 11, 'smooth_method': 'savgol'},
        {'name': 'Savitzky-Golay w=21', 'smooth_window': 21, 'smooth_method': 'savgol'},
    ]

    results_dict = {}

    print("2. 测试不同平滑方法...")
    for config in test_configs:
        print(f"   测试: {config['name']}")

        fig, ospa_map = visualizer.plot_ospa_error(
            true_trajectory, estimated_trajectory, estimated_anchors,
            data_va, parameters, use_soft_ospa=True,
            smooth_window=config['smooth_window'],
            smooth_method=config['smooth_method']
        )
        plt.close(fig)

        # 计算平滑度指标
        smoothness_metrics = {}
        for sensor in range(num_sensors):
            # 计算一阶差分（变化率）
            diff1 = np.abs(np.diff(ospa_map[sensor, :]))
            # 计算二阶差分（加速度/曲率）
            diff2 = np.abs(np.diff(diff1))

            smoothness_metrics[f'sensor{sensor+1}'] = {
                'mean': np.mean(ospa_map[sensor, :]),
                'std': np.std(ospa_map[sensor, :]),
                'max': np.max(ospa_map[sensor, :]),
                'mean_diff1': np.mean(diff1),  # 平均变化率
                'max_diff1': np.max(diff1),    # 最大变化率
                'mean_diff2': np.mean(diff2),  # 平均曲率
                'spikes': np.sum(diff1 > 0.5), # 尖峰数量（变化>0.5m）
            }

        results_dict[config['name']] = {
            'ospa_map': ospa_map,
            'metrics': smoothness_metrics
        }

    print()
    print("=" * 70)
    print("平滑度对比分析")
    print("=" * 70)
    print()

    # 打印对比表格
    for sensor in range(num_sensors):
        print(f"传感器 {sensor + 1}:")
        print(f"{'方法':<20} {'均值':>8} {'标准差':>8} {'平均变化':>10} {'最大变化':>10} {'尖峰数':>8}")
        print("-" * 70)

        for config in test_configs:
            name = config['name']
            metrics = results_dict[name]['metrics'][f'sensor{sensor+1}']
            print(f"{name:<20} {metrics['mean']:>8.4f} {metrics['std']:>8.4f} "
                  f"{metrics['mean_diff1']:>10.4f} {metrics['max_diff1']:>10.4f} "
                  f"{metrics['spikes']:>8}")
        print()

    # 找到最平滑的方法（基于平均变化率）
    print("=" * 70)
    print("推荐方法")
    print("=" * 70)
    for sensor in range(num_sensors):
        best_method = min(test_configs[1:],
                         key=lambda c: results_dict[c['name']]['metrics'][f'sensor{sensor+1}']['mean_diff1'])
        metrics = results_dict[best_method['name']]['metrics'][f'sensor{sensor+1}']
        print(f"传感器 {sensor + 1}: {best_method['name']}")
        print(f"  - 平均变化率: {metrics['mean_diff1']:.4f} m/step")
        print(f"  - 尖峰数量: {metrics['spikes']}")
    print()

    # 绘制对比图
    print("3. 生成对比图...")

    # 选择几个代表性的方法进行可视化
    selected_configs = [
        '原始（无平滑）',
        '移动平均 w=10',
        '移动平均 w=20',
        '指数平滑 w=15',
        'Savitzky-Golay w=21',
    ]

    fig, axes = plt.subplots(num_sensors, 1, figsize=(16, 5*num_sensors))
    if num_sensors == 1:
        axes = [axes]

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for sensor in range(num_sensors):
        ax = axes[sensor]

        for idx, method_name in enumerate(selected_configs):
            ospa_map = results_dict[method_name]['ospa_map']
            metrics = results_dict[method_name]['metrics'][f'sensor{sensor+1}']

            label = f"{method_name} (变化率={metrics['mean_diff1']:.3f})"
            ax.plot(range(1, num_steps + 1), ospa_map[sensor, :],
                   '-', color=colors[idx], linewidth=1.5, alpha=0.8,
                   label=label)

        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('OSPA Error [m]', fontsize=12)
        ax.set_title(f'Sensor {sensor + 1}: Smoothing Method Comparison', fontsize=14)
        ax.set_xlim([0, num_steps])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')

    plt.tight_layout()
    output_path = 'results/ospa_smoothing_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ 对比图已保存: {output_path}")

    # 生成最平滑的OSPA图
    print()
    print("4. 生成最平滑的OSPA图...")

    # 使用移动平均 w=20（通常最平滑）
    fig_smooth, ospa_smooth = visualizer.plot_ospa_error(
        true_trajectory, estimated_trajectory, estimated_anchors,
        data_va, parameters, use_soft_ospa=True,
        smooth_window=20, smooth_method='moving_avg'
    )

    smooth_path = 'results/figure2_ospa_error_gnn_smooth.png'
    fig_smooth.savefig(smooth_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ 平滑OSPA图已保存: {smooth_path}")

    plt.show()

    print()
    print("=" * 70)
    print("总结")
    print("=" * 70)
    print("1. 移动平均（window=15-20）提供最平滑的曲线")
    print("2. 指数平滑保留更多细节，但仍然很平滑")
    print("3. Savitzky-Golay保持峰值形状，适合需要保留真实特征的场景")
    print("4. 推荐使用：移动平均 window=20 获得最平整的曲线")
    print("=" * 70)


if __name__ == '__main__':
    import sys

    results_file = 'results/results_gnn.npz'
    if len(sys.argv) > 1:
        results_file = sys.argv[1]

    test_smoothing_methods(results_file=results_file)
