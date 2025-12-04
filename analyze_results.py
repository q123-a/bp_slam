"""
结果分析脚本
从 .npz 文件中计算详细的统计信息，包括方差、标准差等
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_results(results_file):
    """
    分析结果文件，计算详细统计信息

    参数:
        results_file: 结果文件路径 (如 'results/results_gnn.npz')

    返回:
        stats: 统计信息字典
    """
    print("=" * 70)
    print(f"分析结果文件: {results_file}")
    print("=" * 70)

    # 加载数据
    data = np.load(results_file, allow_pickle=True)

    true_trajectory = data['true_trajectory']
    estimated_trajectory = data['estimated_trajectory']
    num_steps = true_trajectory.shape[1]

    # 1. 计算位置误差 (每个时间步)
    position_errors = np.linalg.norm(
        true_trajectory[0:2, :] - estimated_trajectory[0:2, :],
        axis=0
    )

    # 2. 计算速度误差 (如果有速度信息)
    if true_trajectory.shape[0] >= 4 and estimated_trajectory.shape[0] >= 4:
        velocity_errors = np.linalg.norm(
            true_trajectory[2:4, :] - estimated_trajectory[2:4, :],
            axis=0
        )
    else:
        velocity_errors = None

    # 3. 计算 X 和 Y 方向的误差
    x_errors = true_trajectory[0, :] - estimated_trajectory[0, :]
    y_errors = true_trajectory[1, :] - estimated_trajectory[1, :]

    # 4. 统计信息
    stats = {
        # 位置误差统计
        'position_mean': np.mean(position_errors),
        'position_std': np.std(position_errors),
        'position_var': np.var(position_errors),
        'position_median': np.median(position_errors),
        'position_min': np.min(position_errors),
        'position_max': np.max(position_errors),
        'position_final': position_errors[-1],
        'position_rmse': np.sqrt(np.mean(position_errors**2)),

        # X 方向误差统计
        'x_mean': np.mean(x_errors),
        'x_std': np.std(x_errors),
        'x_var': np.var(x_errors),
        'x_bias': np.mean(x_errors),  # 偏差

        # Y 方向误差统计
        'y_mean': np.mean(y_errors),
        'y_std': np.std(y_errors),
        'y_var': np.var(y_errors),
        'y_bias': np.mean(y_errors),  # 偏差

        # 原始数据
        'position_errors': position_errors,
        'x_errors': x_errors,
        'y_errors': y_errors,
        'num_steps': num_steps,
    }

    # 速度误差统计 (如果有)
    if velocity_errors is not None:
        stats.update({
            'velocity_mean': np.mean(velocity_errors),
            'velocity_std': np.std(velocity_errors),
            'velocity_var': np.var(velocity_errors),
            'velocity_rmse': np.sqrt(np.mean(velocity_errors**2)),
            'velocity_errors': velocity_errors,
        })

    # 5. 打印统计信息
    print("\n" + "=" * 70)
    print("位置误差统计 (Position Error Statistics)")
    print("=" * 70)
    print(f"平均误差 (Mean):        {stats['position_mean']:.6f} m")
    print(f"标准差 (Std Dev):       {stats['position_std']:.6f} m")
    print(f"方差 (Variance):        {stats['position_var']:.6f} m²")
    print(f"中位数 (Median):        {stats['position_median']:.6f} m")
    print(f"最小误差 (Min):         {stats['position_min']:.6f} m")
    print(f"最大误差 (Max):         {stats['position_max']:.6f} m")
    print(f"最终误差 (Final):       {stats['position_final']:.6f} m")
    print(f"均方根误差 (RMSE):      {stats['position_rmse']:.6f} m")

    print("\n" + "=" * 70)
    print("X 方向误差统计 (X-axis Error Statistics)")
    print("=" * 70)
    print(f"平均误差 (Mean):        {stats['x_mean']:.6f} m")
    print(f"标准差 (Std Dev):       {stats['x_std']:.6f} m")
    print(f"方差 (Variance):        {stats['x_var']:.6f} m²")
    print(f"偏差 (Bias):            {stats['x_bias']:.6f} m")

    print("\n" + "=" * 70)
    print("Y 方向误差统计 (Y-axis Error Statistics)")
    print("=" * 70)
    print(f"平均误差 (Mean):        {stats['y_mean']:.6f} m")
    print(f"标准差 (Std Dev):       {stats['y_std']:.6f} m")
    print(f"方差 (Variance):        {stats['y_var']:.6f} m²")
    print(f"偏差 (Bias):            {stats['y_bias']:.6f} m")

    if velocity_errors is not None:
        print("\n" + "=" * 70)
        print("速度误差统计 (Velocity Error Statistics)")
        print("=" * 70)
        print(f"平均误差 (Mean):        {stats['velocity_mean']:.6f} m/s")
        print(f"标准差 (Std Dev):       {stats['velocity_std']:.6f} m/s")
        print(f"方差 (Variance):        {stats['velocity_var']:.6f} (m/s)²")
        print(f"均方根误差 (RMSE):      {stats['velocity_rmse']:.6f} m/s")

    print("\n" + "=" * 70)
    print(f"总时间步数: {num_steps}")
    print("=" * 70)

    return stats


def compare_results(bp_file='results/results_bp.npz',
                   gnn_file='results/results_gnn.npz'):
    """
    对比 BP 和 GNN 的结果

    参数:
        bp_file: BP 结果文件路径
        gnn_file: GNN 结果文件路径
    """
    print("\n" + "=" * 70)
    print("对比 BP 和 GNN 结果")
    print("=" * 70)

    # 检查文件是否存在
    if not Path(bp_file).exists():
        print(f"警告: BP 结果文件不存在: {bp_file}")
        bp_stats = None
    else:
        print(f"\n分析 BP 结果...")
        bp_stats = analyze_results(bp_file)

    if not Path(gnn_file).exists():
        print(f"警告: GNN 结果文件不存在: {gnn_file}")
        gnn_stats = None
    else:
        print(f"\n分析 GNN 结果...")
        gnn_stats = analyze_results(gnn_file)

    # 对比
    if bp_stats is not None and gnn_stats is not None:
        print("\n" + "=" * 70)
        print("对比结果 (Comparison)")
        print("=" * 70)

        print("\n位置误差对比:")
        print(f"{'指标':<20} {'BP':<15} {'GNN':<15} {'改进':<15}")
        print("-" * 70)

        metrics = [
            ('平均误差 (Mean)', 'position_mean', 'm'),
            ('标准差 (Std Dev)', 'position_std', 'm'),
            ('方差 (Variance)', 'position_var', 'm²'),
            ('RMSE', 'position_rmse', 'm'),
            ('最大误差 (Max)', 'position_max', 'm'),
            ('最终误差 (Final)', 'position_final', 'm'),
        ]

        for name, key, unit in metrics:
            bp_val = bp_stats[key]
            gnn_val = gnn_stats[key]
            improvement = (bp_val - gnn_val) / bp_val * 100
            print(f"{name:<20} {bp_val:<15.6f} {gnn_val:<15.6f} {improvement:>+6.2f}%")

        print("\n" + "=" * 70)

        # 绘制对比图
        plot_comparison(bp_stats, gnn_stats)

    return bp_stats, gnn_stats


def plot_comparison(bp_stats, gnn_stats, save_dir='results'):
    """
    绘制 BP 和 GNN 的对比图

    参数:
        bp_stats: BP 统计信息
        gnn_stats: GNN 统计信息
        save_dir: 保存目录
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 位置误差随时间变化
    ax = axes[0, 0]
    steps = np.arange(bp_stats['num_steps'])
    ax.plot(steps, bp_stats['position_errors'], 'b-', linewidth=1.5, label='BP', alpha=0.7)
    ax.plot(steps, gnn_stats['position_errors'], 'r-', linewidth=1.5, label='GNN', alpha=0.7)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Position Error (m)', fontsize=12)
    ax.set_title('Position Error over Time', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. X 方向误差对比
    ax = axes[0, 1]
    ax.plot(steps, bp_stats['x_errors'], 'b-', linewidth=1.5, label='BP', alpha=0.7)
    ax.plot(steps, gnn_stats['x_errors'], 'r-', linewidth=1.5, label='GNN', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('X Error (m)', fontsize=12)
    ax.set_title('X-axis Error over Time', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. Y 方向误差对比
    ax = axes[1, 0]
    ax.plot(steps, bp_stats['y_errors'], 'b-', linewidth=1.5, label='BP', alpha=0.7)
    ax.plot(steps, gnn_stats['y_errors'], 'r-', linewidth=1.5, label='GNN', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Y Error (m)', fontsize=12)
    ax.set_title('Y-axis Error over Time', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. 统计指标对比柱状图
    ax = axes[1, 1]
    metrics = ['Mean', 'Std Dev', 'RMSE', 'Max']
    bp_values = [
        bp_stats['position_mean'],
        bp_stats['position_std'],
        bp_stats['position_rmse'],
        bp_stats['position_max']
    ]
    gnn_values = [
        gnn_stats['position_mean'],
        gnn_stats['position_std'],
        gnn_stats['position_rmse'],
        gnn_stats['position_max']
    ]

    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, bp_values, width, label='BP', color='blue', alpha=0.7)
    ax.bar(x + width/2, gnn_values, width, label='GNN', color='red', alpha=0.7)
    ax.set_ylabel('Error (m)', fontsize=12)
    ax.set_title('Statistical Metrics Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存图形
    save_path = save_dir / 'comparison_bp_vs_gnn.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存: {save_path}")

    plt.show()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        # 单个文件分析
        results_file = sys.argv[1]
        stats = analyze_results(results_file)
    else:
        # 对比分析
        bp_stats, gnn_stats = compare_results(
            bp_file='results/results_bp.npz',
            gnn_file='results/results_gnn.npz'
        )
