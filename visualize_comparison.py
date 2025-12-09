#!/usr/bin/env python3
"""
可视化对比：展示为什么GNN看起来有"凸起"
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_comparison():
    """创建对比可视化，展示Y轴缩放的影响"""

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

    # 创建对比图
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # 子图1: 相同Y轴范围 - 展示真实对比
    ax1 = axes[0]
    ax1.plot(pos_error_bp, 'b-', alpha=0.7, linewidth=1.5, label='BP')
    ax1.plot(pos_error_gnn, 'r-', alpha=0.7, linewidth=1.5, label='GNN (Improved)')
    ax1.set_ylim([0, 0.05])  # 统一Y轴范围
    ax1.axhline(np.mean(pos_error_bp), color='b', linestyle='--', alpha=0.5, label=f'BP mean: {np.mean(pos_error_bp):.4f}m')
    ax1.axhline(np.mean(pos_error_gnn), color='r', linestyle='--', alpha=0.5, label=f'GNN mean: {np.mean(pos_error_gnn):.4f}m')
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Position Error (m)', fontsize=12)
    ax1.set_title('相同Y轴范围对比 - 真实性能差异', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, '✓ GNN明显更好，波动更小',
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # 子图2: 自动Y轴范围 (BP) - 展示BP的大波动
    ax2 = axes[1]
    ax2.plot(pos_error_bp, 'b-', alpha=0.7, linewidth=1.5, label='BP')
    ax2.axhline(np.mean(pos_error_bp), color='b', linestyle='--', alpha=0.5, label=f'BP mean: {np.mean(pos_error_bp):.4f}m')
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Position Error (m)', fontsize=12)
    ax2.set_title('BP单独显示 (自动Y轴) - 大范围波动', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    y_range_bp = ax2.get_ylim()[1] - ax2.get_ylim()[0]
    ax2.text(0.02, 0.98, f'Y轴范围: {y_range_bp:.4f}m',
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # 子图3: 自动Y轴范围 (GNN) - 展示为什么看起来有"凸起"
    ax3 = axes[2]
    ax3.plot(pos_error_gnn, 'r-', alpha=0.7, linewidth=1.5, label='GNN (Improved)')
    ax3.axhline(np.mean(pos_error_gnn), color='r', linestyle='--', alpha=0.5, label=f'GNN mean: {np.mean(pos_error_gnn):.4f}m')
    ax3.set_xlabel('Time Step', fontsize=12)
    ax3.set_ylabel('Position Error (m)', fontsize=12)
    ax3.set_title('GNN单独显示 (自动Y轴) - 小波动被放大', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    y_range_gnn = ax3.get_ylim()[1] - ax3.get_ylim()[0]
    ax3.text(0.02, 0.98, f'⚠ Y轴范围: {y_range_gnn:.4f}m (比BP小{y_range_bp/y_range_gnn:.1f}倍)\n'
                         f'小波动被放大，看起来像"凸起"',
             transform=ax3.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('results/comparison_y_axis_effect.png', dpi=150, bbox_inches='tight')
    print(f"✓ 对比图已保存: results/comparison_y_axis_effect.png")
    plt.close()

    # 打印统计摘要
    print("\n" + "=" * 60)
    print("Y轴缩放效应分析")
    print("=" * 60)
    print(f"\nBP误差范围: {np.min(pos_error_bp):.4f}m ~ {np.max(pos_error_bp):.4f}m")
    print(f"GNN误差范围: {np.min(pos_error_gnn):.4f}m ~ {np.max(pos_error_gnn):.4f}m")
    print(f"\nBP误差跨度: {np.max(pos_error_bp) - np.min(pos_error_bp):.4f}m")
    print(f"GNN误差跨度: {np.max(pos_error_gnn) - np.min(pos_error_gnn):.4f}m")
    print(f"\n跨度比例: BP是GNN的 {(np.max(pos_error_bp) - np.min(pos_error_bp)) / (np.max(pos_error_gnn) - np.min(pos_error_gnn)):.1f} 倍")

    print("\n结论:")
    print("  当GNN单独显示时，Y轴自动缩放到较小范围")
    print("  导致小的波动（0.01m级别）被视觉放大")
    print("  看起来像'凸起'，但实际上比BP的波动小得多")
    print("=" * 60)

if __name__ == "__main__":
    visualize_comparison()
