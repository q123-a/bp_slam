#!/usr/bin/env python3
"""
诊断 GNN 位置误差突起的原因
分析伪标签质量、训练稳定性等因素
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_gnn_spikes():
    """分析 GNN 结果中的位置误差突起"""

    # 加载结果
    results_gnn = np.load('results/results_gnn.npz', allow_pickle=True)
    results_bp = np.load('results/results_bp.npz', allow_pickle=True)

    # 从轨迹计算位置误差
    est_traj_gnn = results_gnn['estimated_trajectory']  # (4, M) - [x, y, vx, vy]
    true_traj_gnn = results_gnn['true_trajectory']      # (2, M) - [x, y]
    est_traj_bp = results_bp['estimated_trajectory']    # (4, M)
    true_traj_bp = results_bp['true_trajectory']        # (2, M)

    # 只取位置部分 (前两行)
    est_pos_gnn = est_traj_gnn[:2, :]  # (2, M)
    est_pos_bp = est_traj_bp[:2, :]    # (2, M)

    # 计算欧氏距离误差
    pos_error_gnn = np.sqrt(np.sum((est_pos_gnn - true_traj_gnn)**2, axis=0))  # (M,)
    pos_error_bp = np.sqrt(np.sum((est_pos_bp - true_traj_bp)**2, axis=0))     # (M,)

    print("=" * 60)
    print("GNN 位置误差突起诊断报告")
    print("=" * 60)

    # 1. 基本统计
    print("\n1. 基本统计:")
    print(f"   GNN 平均误差: {np.mean(pos_error_gnn):.4f} m")
    print(f"   BP  平均误差: {np.mean(pos_error_bp):.4f} m")
    print(f"   GNN 最大误差: {np.max(pos_error_gnn):.4f} m")
    print(f"   BP  最大误差: {np.max(pos_error_bp):.4f} m")
    print(f"   GNN 标准差:   {np.std(pos_error_gnn):.4f} m")
    print(f"   BP  标准差:   {np.std(pos_error_bp):.4f} m")

    # 2. 突起检测
    threshold = np.mean(pos_error_gnn) + 2 * np.std(pos_error_gnn)
    spike_indices_gnn = np.where(pos_error_gnn > threshold)[0]
    spike_indices_bp = np.where(pos_error_bp > threshold)[0]

    print(f"\n2. 突起检测 (阈值: {threshold:.4f} m):")
    print(f"   GNN 突起数量: {len(spike_indices_gnn)} / {len(pos_error_gnn)} ({len(spike_indices_gnn)/len(pos_error_gnn)*100:.1f}%)")
    print(f"   BP  突起数量: {len(spike_indices_bp)} / {len(pos_error_bp)} ({len(spike_indices_bp)/len(pos_error_bp)*100:.1f}%)")

    # 3. GNN 特有突起
    gnn_only_spikes = set(spike_indices_gnn) - set(spike_indices_bp)
    print(f"\n3. GNN 特有突起:")
    print(f"   数量: {len(gnn_only_spikes)} ({len(gnn_only_spikes)/len(pos_error_gnn)*100:.1f}%)")

    if len(gnn_only_spikes) > 0:
        gnn_only_list = sorted(list(gnn_only_spikes))[:10]  # 显示前10个
        print(f"   位置 (前10个): {gnn_only_list}")
        print(f"   误差值:")
        for idx in gnn_only_list:
            print(f"      Step {idx}: GNN={pos_error_gnn[idx]:.4f}m, BP={pos_error_bp[idx]:.4f}m, 差值={pos_error_gnn[idx]-pos_error_bp[idx]:.4f}m")

    # 4. 误差差异分析
    error_diff = pos_error_gnn - pos_error_bp
    print(f"\n4. GNN vs BP 误差差异:")
    print(f"   平均差异: {np.mean(error_diff):.4f} m")
    print(f"   最大正差异: {np.max(error_diff):.4f} m (GNN 更差)")
    print(f"   最大负差异: {np.min(error_diff):.4f} m (GNN 更好)")
    print(f"   差异标准差: {np.std(error_diff):.4f} m")

    worse_count = np.sum(error_diff > 0)
    print(f"   GNN 更差的步数: {worse_count} / {len(error_diff)} ({worse_count/len(error_diff)*100:.1f}%)")

    # 5. 时间分布分析
    print(f"\n5. 突起的时间分布:")
    if len(spike_indices_gnn) > 0:
        early_spikes = np.sum(spike_indices_gnn < len(pos_error_gnn) // 3)
        mid_spikes = np.sum((spike_indices_gnn >= len(pos_error_gnn) // 3) &
                           (spike_indices_gnn < 2 * len(pos_error_gnn) // 3))
        late_spikes = np.sum(spike_indices_gnn >= 2 * len(pos_error_gnn) // 3)

        print(f"   前1/3: {early_spikes} ({early_spikes/len(spike_indices_gnn)*100:.1f}%)")
        print(f"   中1/3: {mid_spikes} ({mid_spikes/len(spike_indices_gnn)*100:.1f}%)")
        print(f"   后1/3: {late_spikes} ({late_spikes/len(spike_indices_gnn)*100:.1f}%)")

    # 6. 可视化分析
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 子图1: 位置误差对比
    ax1 = axes[0]
    ax1.plot(pos_error_bp, 'b-', alpha=0.6, linewidth=1, label='BP')
    ax1.plot(pos_error_gnn, 'r-', alpha=0.6, linewidth=1, label='GNN')
    ax1.axhline(threshold, color='k', linestyle='--', alpha=0.3, label=f'Threshold ({threshold:.2f}m)')
    if len(gnn_only_spikes) > 0:
        gnn_only_array = np.array(sorted(list(gnn_only_spikes)))
        ax1.scatter(gnn_only_array, pos_error_gnn[gnn_only_array],
                   color='orange', s=50, marker='x', label='GNN-only spikes', zorder=5)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Position Error (m)')
    ax1.set_title('Position Error Comparison: BP vs GNN')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: 误差差异
    ax2 = axes[1]
    ax2.plot(error_diff, 'g-', alpha=0.6, linewidth=1)
    ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax2.fill_between(range(len(error_diff)), 0, error_diff,
                     where=(error_diff > 0), color='red', alpha=0.3, label='GNN worse')
    ax2.fill_between(range(len(error_diff)), 0, error_diff,
                     where=(error_diff <= 0), color='blue', alpha=0.3, label='GNN better')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Error Difference (m)')
    ax2.set_title('GNN - BP Position Error Difference')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图3: 误差分布直方图
    ax3 = axes[2]
    bins = np.linspace(0, max(np.max(pos_error_bp), np.max(pos_error_gnn)), 50)
    ax3.hist(pos_error_bp, bins=bins, alpha=0.5, color='blue', label='BP', density=True)
    ax3.hist(pos_error_gnn, bins=bins, alpha=0.5, color='red', label='GNN', density=True)
    ax3.axvline(np.mean(pos_error_bp), color='blue', linestyle='--', label=f'BP mean ({np.mean(pos_error_bp):.2f}m)')
    ax3.axvline(np.mean(pos_error_gnn), color='red', linestyle='--', label=f'GNN mean ({np.mean(pos_error_gnn):.2f}m)')
    ax3.set_xlabel('Position Error (m)')
    ax3.set_ylabel('Density')
    ax3.set_title('Position Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/diagnostic_gnn_spikes.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ 诊断图已保存: results/diagnostic_gnn_spikes.png")
    plt.close()

    # 7. 结论和建议
    print("\n" + "=" * 60)
    print("诊断结论:")
    print("=" * 60)

    if len(gnn_only_spikes) > len(spike_indices_gnn) * 0.5:
        print("⚠ GNN 特有突起占比较高，可能原因:")
        print("  1. 伪标签质量不稳定 - 几何/幅度一致性判断过于宽松")
        print("  2. 模型过拟合到错误的伪标签")
        print("  3. GRU 记忆状态传播错误")
        print("  4. 学习率仍然偏高，导致参数震荡")

    if np.mean(error_diff) > 0.1:
        print("⚠ GNN 平均误差明显高于 BP，建议:")
        print("  1. 收紧伪标签生成条件")
        print("  2. 增加负样本权重")
        print("  3. 使用更保守的学习率调度")

    if early_spikes > late_spikes * 2:
        print("⚠ 突起主要集中在前期，可能是:")
        print("  1. 模型初始化不佳")
        print("  2. 早期伪标签质量差")
        print("  3. 建议使用预训练或 warm-up")

    print("\n建议的改进方案:")
    print("  1. 收紧伪标签条件: 将 OR 改回 AND，或提高阈值")
    print("  2. 添加置信度加权: 根据几何/幅度一致性程度加权损失")
    print("  3. 使用学习率调度: 从高到低逐渐衰减")
    print("  4. 增加梯度裁剪强度: 从 0.5 降到 0.1")
    print("  5. 添加 EMA (指数移动平均): 平滑模型参数更新")
    print("=" * 60)

if __name__ == "__main__":
    analyze_gnn_spikes()
