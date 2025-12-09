#!/usr/bin/env python3
"""
深入分析GNN凸起的具体原因
检查凸起时刻的测量数据、锚点数量、训练损失等
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_spike_causes():
    """深入分析GNN凸起的根本原因"""

    # 加载结果
    results_gnn = np.load('results/results_gnn.npz', allow_pickle=True)
    results_bp = np.load('results/results_bp.npz', allow_pickle=True)

    # 计算位置误差
    est_pos_gnn = results_gnn['estimated_trajectory'][:2, :]
    true_traj_gnn = results_gnn['true_trajectory']
    est_pos_bp = results_bp['estimated_trajectory'][:2, :]
    true_traj_bp = results_bp['true_trajectory']

    pos_error_gnn = np.sqrt(np.sum((est_pos_gnn - true_traj_gnn)**2, axis=0))
    pos_error_bp = np.sqrt(np.sum((est_pos_bp - true_traj_bp)**2, axis=0))

    # 获取锚点数量
    num_anchors_gnn = results_gnn['num_estimated_anchors']  # (num_sensors, num_steps)
    num_anchors_bp = results_bp['num_estimated_anchors']

    print("=" * 70)
    print("GNN 凸起深度分析")
    print("=" * 70)

    # 1. 识别凸起
    threshold = np.mean(pos_error_gnn) + 2 * np.std(pos_error_gnn)
    spike_indices = np.where(pos_error_gnn > threshold)[0]

    print(f"\n1. 凸起识别 (阈值: {threshold:.4f}m)")
    print(f"   凸起数量: {len(spike_indices)} / {len(pos_error_gnn)} ({len(spike_indices)/len(pos_error_gnn)*100:.1f}%)")
    print(f"   凸起位置: {spike_indices[:20].tolist()}..." if len(spike_indices) > 20 else f"   凸起位置: {spike_indices.tolist()}")

    # 2. 分析凸起的特征
    if len(spike_indices) > 0:
        print(f"\n2. 凸起特征分析:")

        # 凸起时的误差统计
        spike_errors = pos_error_gnn[spike_indices]
        print(f"   凸起误差范围: {np.min(spike_errors):.4f}m ~ {np.max(spike_errors):.4f}m")
        print(f"   凸起平均误差: {np.mean(spike_errors):.4f}m")

        # 凸起时的锚点数量
        spike_anchors_gnn = num_anchors_gnn[:, spike_indices]  # (num_sensors, num_spikes)
        spike_anchors_bp = num_anchors_bp[:, spike_indices]

        print(f"\n   凸起时的锚点数量 (传感器1):")
        print(f"     GNN: {np.mean(spike_anchors_gnn[0]):.1f} ± {np.std(spike_anchors_gnn[0]):.1f}")
        print(f"     BP:  {np.mean(spike_anchors_bp[0]):.1f} ± {np.std(spike_anchors_bp[0]):.1f}")

        if num_anchors_gnn.shape[0] > 1:
            print(f"   凸起时的锚点数量 (传感器2):")
            print(f"     GNN: {np.mean(spike_anchors_gnn[1]):.1f} ± {np.std(spike_anchors_gnn[1]):.1f}")
            print(f"     BP:  {np.mean(spike_anchors_bp[1]):.1f} ± {np.std(spike_anchors_bp[1]):.1f}")

        # 3. 凸起的时间模式
        print(f"\n3. 凸起的时间模式:")

        # 检查凸起是否连续
        consecutive_spikes = []
        current_group = [spike_indices[0]]
        for i in range(1, len(spike_indices)):
            if spike_indices[i] - spike_indices[i-1] == 1:
                current_group.append(spike_indices[i])
            else:
                if len(current_group) > 1:
                    consecutive_spikes.append(current_group)
                current_group = [spike_indices[i]]
        if len(current_group) > 1:
            consecutive_spikes.append(current_group)

        print(f"   连续凸起组数: {len(consecutive_spikes)}")
        if len(consecutive_spikes) > 0:
            print(f"   最长连续凸起: {max(len(g) for g in consecutive_spikes)} 步")
            print(f"   连续凸起示例:")
            for i, group in enumerate(consecutive_spikes[:5]):
                print(f"     组{i+1}: 步数 {group[0]}-{group[-1]} (长度{len(group)})")

        # 4. 凸起前后的误差变化
        print(f"\n4. 凸起前后的误差变化:")

        # 选择几个典型凸起进行分析
        sample_spikes = spike_indices[::max(1, len(spike_indices)//5)][:5]  # 选5个代表性凸起

        for spike_idx in sample_spikes:
            # 获取前后5步的误差
            start = max(0, spike_idx - 5)
            end = min(len(pos_error_gnn), spike_idx + 6)

            window_errors_gnn = pos_error_gnn[start:end]
            window_errors_bp = pos_error_bp[start:end]

            spike_pos_in_window = spike_idx - start

            print(f"\n   凸起@步数{spike_idx}:")
            print(f"     前5步GNN误差: {window_errors_gnn[:spike_pos_in_window]}")
            print(f"     凸起时GNN误差: {window_errors_gnn[spike_pos_in_window]:.4f}m")
            print(f"     后5步GNN误差: {window_errors_gnn[spike_pos_in_window+1:]}")
            print(f"     对应BP误差:   {window_errors_bp[spike_pos_in_window]:.4f}m")

            # 检查是否是突然跳变
            if spike_pos_in_window > 0:
                jump_before = window_errors_gnn[spike_pos_in_window] - window_errors_gnn[spike_pos_in_window-1]
                print(f"     误差跳变: {jump_before:+.4f}m")

    # 5. 可视化分析
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))

    # 子图1: 位置误差 + 凸起标记
    ax1 = axes[0]
    ax1.plot(pos_error_gnn, 'r-', alpha=0.6, linewidth=1, label='GNN')
    ax1.plot(pos_error_bp, 'b-', alpha=0.4, linewidth=1, label='BP')
    ax1.axhline(threshold, color='k', linestyle='--', alpha=0.3, label=f'Threshold')
    if len(spike_indices) > 0:
        ax1.scatter(spike_indices, pos_error_gnn[spike_indices],
                   color='red', s=30, marker='o', label='Spikes', zorder=5)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Position Error (m)')
    ax1.set_title('Position Error with Spike Markers')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: 锚点数量对比
    ax2 = axes[1]
    ax2.plot(num_anchors_gnn[0, :], 'r-', alpha=0.6, linewidth=1, label='GNN Sensor1')
    ax2.plot(num_anchors_bp[0, :], 'b-', alpha=0.4, linewidth=1, label='BP Sensor1')
    if len(spike_indices) > 0:
        ax2.scatter(spike_indices, num_anchors_gnn[0, spike_indices],
                   color='red', s=30, marker='o', zorder=5)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Number of Anchors')
    ax2.set_title('Number of Anchors (Sensor 1)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图3: 误差差异
    ax3 = axes[2]
    error_diff = pos_error_gnn - pos_error_bp
    ax3.plot(error_diff, 'g-', alpha=0.6, linewidth=1)
    ax3.axhline(0, color='k', linestyle='-', alpha=0.3)
    if len(spike_indices) > 0:
        ax3.scatter(spike_indices, error_diff[spike_indices],
                   color='red', s=30, marker='o', zorder=5)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Error Difference (GNN - BP) (m)')
    ax3.set_title('GNN vs BP Error Difference')
    ax3.grid(True, alpha=0.3)

    # 子图4: 误差的一阶差分（变化率）
    ax4 = axes[3]
    error_change_gnn = np.diff(pos_error_gnn, prepend=pos_error_gnn[0])
    error_change_bp = np.diff(pos_error_bp, prepend=pos_error_bp[0])
    ax4.plot(error_change_gnn, 'r-', alpha=0.6, linewidth=1, label='GNN change')
    ax4.plot(error_change_bp, 'b-', alpha=0.4, linewidth=1, label='BP change')
    ax4.axhline(0, color='k', linestyle='-', alpha=0.3)
    if len(spike_indices) > 0:
        ax4.scatter(spike_indices, error_change_gnn[spike_indices],
                   color='red', s=30, marker='o', zorder=5)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Error Change (m)')
    ax4.set_title('Position Error Change Rate (1st Derivative)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/spike_cause_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ 分析图已保存: results/spike_cause_analysis.png")
    plt.close()

    # 6. 结论和建议
    print("\n" + "=" * 70)
    print("诊断结论:")
    print("=" * 70)

    if len(spike_indices) > 0:
        # 检查凸起是否与锚点数量相关
        all_anchors_gnn = num_anchors_gnn[0, :]
        spike_anchor_mean = np.mean(all_anchors_gnn[spike_indices])
        normal_anchor_mean = np.mean(np.delete(all_anchors_gnn, spike_indices))

        print(f"\n锚点数量分析:")
        print(f"  凸起时平均锚点数: {spike_anchor_mean:.1f}")
        print(f"  正常时平均锚点数: {normal_anchor_mean:.1f}")
        print(f"  差异: {spike_anchor_mean - normal_anchor_mean:+.1f}")

        if abs(spike_anchor_mean - normal_anchor_mean) > 1:
            print(f"\n⚠ 凸起与锚点数量变化相关！")
            if spike_anchor_mean < normal_anchor_mean:
                print("  → 凸起时锚点数量较少，可能是锚点丢失导致")
            else:
                print("  → 凸起时锚点数量较多，可能是新锚点初始化不准确")

        # 检查是否有连续凸起
        if len(consecutive_spikes) > 0:
            print(f"\n⚠ 存在{len(consecutive_spikes)}组连续凸起！")
            print("  → 这表明GNN在某些时段持续产生错误的关联")
            print("  → 可能原因:")
            print("    1. 伪标签质量在这些时段较差")
            print("    2. GRU记忆状态传播了错误信息")
            print("    3. 学习率过高导致参数震荡")

        # 检查误差跳变
        large_jumps = np.where(np.abs(error_change_gnn) > 0.01)[0]
        print(f"\n误差跳变分析:")
        print(f"  大跳变次数 (>0.01m): {len(large_jumps)}")
        print(f"  最大跳变: {np.max(np.abs(error_change_gnn)):.4f}m")

        if len(large_jumps) > len(pos_error_gnn) * 0.05:
            print(f"\n⚠ 误差跳变频繁 ({len(large_jumps)/len(pos_error_gnn)*100:.1f}%)！")
            print("  → 这表明GNN输出不够平滑")
            print("  → 建议:")
            print("    1. 增强EMA效果 (提高decay到0.9995)")
            print("    2. 降低学习率 (从1e-4降到5e-5)")
            print("    3. 增加迭代次数 (从5次增加到10次)")

    print("\n推荐的改进方案:")
    print("  1. 降低学习率: 1e-4 → 5e-5")
    print("  2. 增强EMA: decay 0.999 → 0.9995")
    print("  3. 更强梯度裁剪: 0.1 → 0.05")
    print("  4. 增加训练迭代: 5次 → 10次")
    print("  5. 使用更保守的伪标签: 'and' + 更严格阈值")
    print("=" * 70)

if __name__ == "__main__":
    analyze_spike_causes()
