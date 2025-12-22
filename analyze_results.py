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


def compare_results(file1='results/results_bp.npz',
                   file2='results/results_gnn.npz',
                   label1='Method 1',
                   label2='Method 2'):
    """
    对比两个结果文件

    参数:
        file1: 第一个结果文件路径
        file2: 第二个结果文件路径
        label1: 第一个方法的标签 (用于显示)
        label2: 第二个方法的标签 (用于显示)
    """
    print("\n" + "=" * 70)
    print(f"对比 {label1} 和 {label2} 结果")
    print("=" * 70)

    # 检查文件是否存在
    if not Path(file1).exists():
        print(f"警告: 结果文件不存在: {file1}")
        stats1 = None
    else:
        print(f"\n分析 {label1} 结果...")
        stats1 = analyze_results(file1)

    if not Path(file2).exists():
        print(f"警告: 结果文件不存在: {file2}")
        stats2 = None
    else:
        print(f"\n分析 {label2} 结果...")
        stats2 = analyze_results(file2)

    # 对比
    if stats1 is not None and stats2 is not None:
        print("\n" + "=" * 70)
        print("对比结果 (Comparison)")
        print("=" * 70)

        print("\n位置误差对比:")
        print(f"{'指标':<20} {label1:<15} {label2:<15} {'改进':<15}")
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
            val1 = stats1[key]
            val2 = stats2[key]
            improvement = (val1 - val2) / val1 * 100
            print(f"{name:<20} {val1:<15.6f} {val2:<15.6f} {improvement:>+6.2f}%")

        print("\n" + "=" * 70)

        # 绘制对比图
        plot_comparison(stats1, stats2, label1=label1, label2=label2)

    return stats1, stats2


def plot_comparison(stats1, stats2, label1='Method 1', label2='Method 2', save_dir='results'):
    """
    绘制两个方法的对比图

    参数:
        stats1: 第一个方法的统计信息
        stats2: 第二个方法的统计信息
        label1: 第一个方法的标签
        label2: 第二个方法的标签
        save_dir: 保存目录
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 确定最小的时间步数，以对齐两个轨迹
    min_steps = min(stats1['num_steps'], stats2['num_steps'])

    # 1. 位置误差随时间变化
    ax = axes[0, 0]
    steps = np.arange(min_steps)
    ax.plot(steps, stats1['position_errors'][:min_steps], 'b-', linewidth=1.5, label=label1, alpha=0.7)
    ax.plot(steps, stats2['position_errors'][:min_steps], 'r-', linewidth=1.5, label=label2, alpha=0.7)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Position Error (m)', fontsize=12)
    ax.set_title('Position Error over Time', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. X 方向误差对比
    ax = axes[0, 1]
    ax.plot(steps, stats1['x_errors'][:min_steps], 'b-', linewidth=1.5, label=label1, alpha=0.7)
    ax.plot(steps, stats2['x_errors'][:min_steps], 'r-', linewidth=1.5, label=label2, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('X Error (m)', fontsize=12)
    ax.set_title('X-axis Error over Time', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. Y 方向误差对比
    ax = axes[1, 0]
    ax.plot(steps, stats1['y_errors'][:min_steps], 'b-', linewidth=1.5, label=label1, alpha=0.7)
    ax.plot(steps, stats2['y_errors'][:min_steps], 'r-', linewidth=1.5, label=label2, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Y Error (m)', fontsize=12)
    ax.set_title('Y-axis Error over Time', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. 统计指标对比柱状图
    ax = axes[1, 1]
    metrics = ['Mean', 'Std Dev', 'RMSE', 'Max']
    values1 = [
        stats1['position_mean'],
        stats1['position_std'],
        stats1['position_rmse'],
        stats1['position_max']
    ]
    values2 = [
        stats2['position_mean'],
        stats2['position_std'],
        stats2['position_rmse'],
        stats2['position_max']
    ]

    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, values1, width, label=label1, color='blue', alpha=0.7)
    ax.bar(x + width/2, values2, width, label=label2, color='red', alpha=0.7)
    ax.set_ylabel('Error (m)', fontsize=12)
    ax.set_title('Statistical Metrics Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存图形 (使用标签生成文件名)
    safe_label1 = label1.replace(' ', '_').replace('/', '_')
    safe_label2 = label2.replace(' ', '_').replace('/', '_')
    save_path = save_dir / f'comparison_{safe_label1}_vs_{safe_label2}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存: {save_path}")

    # 关闭图形以释放内存
    plt.close(fig)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='分析和对比 BP-SLAM 结果文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 分析单个结果文件:
   python analyze_results.py results/results_gnn.npz

2. 对比两个结果文件 (默认 BP vs GNN):
   python analyze_results.py --compare

3. 对比自定义文件:
   python analyze_results.py --compare --file1 results/results_bp.npz --file2 results/results_gnn.npz

4. 对比不同 GNN 版本:
   python analyze_results.py --compare --file1 results/results_gnn_v1.npz --file2 results/results_gnn_v2.npz --label1 "GNN V1" --label2 "GNN V2"

5. 对比三个版本 (V1 vs V2, V2 vs V3):
   python analyze_results.py --compare --file1 results/results_gnn_v1.npz --file2 results/results_gnn_v2.npz --label1 "V1" --label2 "V2"
   python analyze_results.py --compare --file1 results/results_gnn_v2.npz --file2 results/results_gnn_v3.npz --label1 "V2" --label2 "V3"
        """
    )

    parser.add_argument('file', nargs='?', default=None,
                        help='单个结果文件路径 (用于单文件分析)')
    parser.add_argument('--compare', action='store_true',
                        help='对比两个结果文件')
    parser.add_argument('--file1', type=str, default='results/results_bp.npz',
                        help='第一个结果文件路径 (默认: results/results_bp.npz)')
    parser.add_argument('--file2', type=str, default='results/results_gnn.npz',
                        help='第二个结果文件路径 (默认: results/results_gnn.npz)')
    parser.add_argument('--label1', type=str, default=None,
                        help='第一个方法的标签 (默认: 从文件名推断)')
    parser.add_argument('--label2', type=str, default=None,
                        help='第二个方法的标签 (默认: 从文件名推断)')

    args = parser.parse_args()

    # 自动推断标签
    def infer_label(filepath):
        """从文件路径推断标签"""
        filename = Path(filepath).stem  # 获取不带扩展名的文件名
        # 移除 'results_' 前缀
        if filename.startswith('results_'):
            label = filename[8:]  # 去掉 'results_'
        else:
            label = filename
        # 转换为大写并美化
        label = label.upper().replace('_', ' ')
        return label

    if args.compare:
        # 对比模式
        label1 = args.label1 if args.label1 else infer_label(args.file1)
        label2 = args.label2 if args.label2 else infer_label(args.file2)

        stats1, stats2 = compare_results(
            file1=args.file1,
            file2=args.file2,
            label1=label1,
            label2=label2
        )
    elif args.file:
        # 单文件分析模式
        stats = analyze_results(args.file)
    else:
        # 默认: 对比 BP vs GNN
        print("提示: 使用 --compare 进行对比分析，或提供文件路径进行单文件分析")
        print("运行 'python analyze_results.py --help' 查看使用说明\n")

        stats1, stats2 = compare_results(
            file1='results/results_bp.npz',
            file2='results/results_gnn.npz',
            label1='BP',
            label2='GNN'
        )
