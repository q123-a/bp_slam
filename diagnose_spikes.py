#!/usr/bin/env python3
"""
诊断GNN位置误差中的凸起问题
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def diagnose_spikes(results_file='results/results_gnn.npz', threshold_multiplier=2.0):
    """
    诊断GNN结果中的凸起

    参数:
        results_file: 结果文件路径
        threshold_multiplier: 凸起阈值倍数（相对于mean + std）
    """
    print("=" * 70)
    print("GNN 位置误差凸起诊断")
    print("=" * 70)

    # 加载数据
    data = np.load(results_file, allow_pickle=True)
    true_traj = data['true_trajectory']
    est_traj = data['estimated_trajectory']

    # 确保长度一致
    num_steps = min(true_traj.shape[1], est_traj.shape[1])

    # 计算位置误差
    errors = np.linalg.norm(
        true_traj[0:2, :num_steps] - est_traj[0:2, :num_steps],
        axis=0
    )

    # 基本统计
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)

    print(f"\n1. 基本统计:")
    print(f"   平均误差: {mean_error:.4f} m")
    print(f"   标准差:   {std_error:.4f} m")
    print(f"   最大误差: {max_error:.4f} m")
    print(f"   最小误差: {min_error:.4f} m")

    # 检测凸起
    spike_threshold = mean_error + threshold_multiplier * std_error
    spike_indices = np.where(errors > spike_threshold)[0]
    num_spikes = len(spike_indices)
    spike_percentage = 100 * num_spikes / num_steps

    print(f"\n2. 凸起检测 (阈值: {spike_threshold:.4f} m = mean + {threshold_multiplier}*std):")
    print(f"   凸起数量: {num_spikes} / {num_steps} ({spike_percentage:.1f}%)")

    if num_spikes > 0:
        print(f"   凸起位置 (前20个): {spike_indices[:20].tolist()}")
        print(f"   凸起误差值:")
        for i in spike_indices[:10]:
            print(f"      步骤 {i}: {errors[i]:.4f} m")

        # 分析连续凸起
        consecutive_groups = []
        if num_spikes > 0:
            current_group = [spike_indices[0]]
            for i in range(1, len(spike_indices)):
                if spike_indices[i] == spike_indices[i-1] + 1:
                    current_group.append(spike_indices[i])
                else:
                    if len(current_group) > 1:
                        consecutive_groups.append(current_group)
                    current_group = [spike_indices[i]]
            if len(current_group) > 1:
                consecutive_groups.append(current_group)

        print(f"\n3. 连续凸起分析:")
        print(f"   连续凸起组数: {len(consecutive_groups)}")
        if consecutive_groups:
            for idx, group in enumerate(consecutive_groups[:5]):
                print(f"   组{idx+1}: 步骤 {group[0]}-{group[-1]} (长度: {len(group)})")

        # 时间分布
        early_spikes = np.sum(spike_indices < num_steps // 3)
        middle_spikes = np.sum((spike_indices >= num_steps // 3) & (spike_indices < 2 * num_steps // 3))
        late_spikes = np.sum(spike_indices >= 2 * num_steps // 3)

        print(f"\n4. 凸起的时间分布:")
        print(f"   前1/3: {early_spikes} ({100*early_spikes/num_spikes:.1f}%)")
        print(f"   中1/3: {middle_spikes} ({100*middle_spikes/num_spikes:.1f}%)")
        print(f"   后1/3: {late_spikes} ({100*late_spikes/num_spikes:.1f}%)")

    # 可视化
    plt.figure(figsize=(14, 8))

    # 子图1: 误差曲线
    plt.subplot(2, 1, 1)
    plt.plot(errors, 'b-', linewidth=1, label='Position Error')
    plt.axhline(y=mean_error, color='g', linestyle='--', label=f'Mean: {mean_error:.4f}m')
    plt.axhline(y=spike_threshold, color='r', linestyle='--', label=f'Spike Threshold: {spike_threshold:.4f}m')
    if num_spikes > 0:
        plt.scatter(spike_indices, errors[spike_indices], color='r', s=50, zorder=5, label=f'Spikes ({num_spikes})')
    plt.xlabel('Time Step')
    plt.ylabel('Position Error (m)')
    plt.title(f'GNN Position Error - {num_spikes} Spikes Detected ({spike_percentage:.1f}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2: 误差直方图
    plt.subplot(2, 1, 2)
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=mean_error, color='g', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.4f}m')
    plt.axvline(x=spike_threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold: {spike_threshold:.4f}m')
    plt.xlabel('Position Error (m)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    output_file = 'results/spike_diagnosis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ 诊断图已保存: {output_file}")

    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)

    return {
        'num_spikes': num_spikes,
        'spike_percentage': spike_percentage,
        'spike_indices': spike_indices,
        'mean_error': mean_error,
        'std_error': std_error,
        'consecutive_groups': consecutive_groups
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='诊断GNN凸起问题')
    parser.add_argument('--file', type=str, default='results/results_gnn.npz',
                       help='结果文件路径')
    parser.add_argument('--threshold', type=float, default=2.0,
                       help='凸起阈值倍数 (mean + threshold*std)')

    args = parser.parse_args()

    diagnose_spikes(args.file, args.threshold)
