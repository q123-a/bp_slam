"""
GNN Loss 可视化脚本
从权重文件或结果文件中读取 Loss 历史并绘制曲线
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import sys


def plot_loss_from_checkpoint(checkpoint_path='checkpoints/gnn_model.pth', save_path=None):
    """
    从权重文件中读取并绘制 Loss 曲线

    参数:
        checkpoint_path: 权重文件路径
        save_path: 保存图片路径 (None 表示不保存)
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"错误: 权重文件不存在: {checkpoint_path}")
        return None

    # 加载权重文件
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 获取 Loss 历史和预热步数
    loss_history = checkpoint.get('loss_history', [])
    warmup_steps = checkpoint.get('warmup_steps', None)
    num_steps = checkpoint.get('num_steps', None)

    if len(loss_history) == 0:
        print("=" * 60)
        print("警告: 权重文件中没有 Loss 历史记录")
        print("=" * 60)
        print("\n这个权重文件是在添加 Loss 记录功能之前保存的。")
        print("请重新训练以生成包含 Loss 历史的权重文件：")
        print("\n  python testbed.py --mode gnn --steps 900 --warmup 45")
        print("\n或者删除旧权重重新训练：")
        print("\n  rm checkpoints/gnn_model.pth")
        print("  python testbed.py --mode gnn --steps 900 --warmup 45")
        print("=" * 60)
        return None, []

    print(f"加载 Loss 历史: {len(loss_history)} 个数据点")
    if warmup_steps is not None:
        print(f"预热步数: {warmup_steps}")
    if num_steps is not None:
        print(f"总步数: {num_steps}")

    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 1. 完整 Loss 曲线
    ax = axes[0]

    # 使用 SLAM 步数作为 x 轴（更直观）
    if warmup_steps is not None and num_steps is not None:
        # x 轴从 warmup_steps 开始，到 num_steps-1 结束
        slam_steps = np.arange(warmup_steps, warmup_steps + len(loss_history))
        ax.plot(slam_steps, loss_history, 'b-', linewidth=1, alpha=0.6, label='Loss')

        # 添加移动平均线
        if len(loss_history) > 50:
            window_size = min(50, len(loss_history) // 10)
            moving_avg = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
            ax.plot(slam_steps[window_size-1:], moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window_size})')

        # 标注预热分界线（在 warmup_steps 位置）
        ax.axvline(x=warmup_steps, color='green', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'GNN Start (step {warmup_steps})')

        # 添加预热区域的阴影
        ax.axvspan(0, warmup_steps, alpha=0.1, color='gray', label='Warmup Period (BP)')

        ax.set_xlabel('SLAM Step', fontsize=12)
        ax.set_xlim([0, num_steps])
    else:
        # 如果没有预热信息，使用 GNN 训练步数
        steps = np.arange(len(loss_history))
        ax.plot(steps, loss_history, 'b-', linewidth=1, alpha=0.6, label='Loss')

        if len(loss_history) > 50:
            window_size = min(50, len(loss_history) // 10)
            moving_avg = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
            ax.plot(steps[window_size-1:], moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window_size})')

        ax.set_xlabel('GNN Training Step', fontsize=12)

    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('GNN Training Loss', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    # 2. 对数尺度 Loss 曲线
    ax = axes[1]

    if warmup_steps is not None and num_steps is not None:
        # 使用 SLAM 步数作为 x 轴
        slam_steps = np.arange(warmup_steps, warmup_steps + len(loss_history))
        ax.semilogy(slam_steps, loss_history, 'b-', linewidth=1, alpha=0.6, label='Loss (log scale)')

        if len(loss_history) > 50:
            ax.semilogy(slam_steps[window_size-1:], moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window_size})')

        # 标注预热分界线
        ax.axvline(x=warmup_steps, color='green', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'GNN Start (step {warmup_steps})')

        # 添加预热区域的阴影
        ax.axvspan(0, warmup_steps, alpha=0.1, color='gray', label='Warmup Period (BP)')

        ax.set_xlabel('SLAM Step', fontsize=12)
        ax.set_xlim([0, num_steps])
    else:
        # 如果没有预热信息，使用 GNN 训练步数
        steps = np.arange(len(loss_history))
        ax.semilogy(steps, loss_history, 'b-', linewidth=1, alpha=0.6, label='Loss (log scale)')

        if len(loss_history) > 50:
            ax.semilogy(steps[window_size-1:], moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window_size})')

        ax.set_xlabel('GNN Training Step', fontsize=12)

    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('GNN Training Loss (Log Scale)', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 打印统计信息
    print("\n" + "=" * 60)
    print("Loss 统计信息")
    print("=" * 60)
    print(f"总训练步数: {len(loss_history)}")
    print(f"初始 Loss: {loss_history[0]:.6f}")
    print(f"最终 Loss: {loss_history[-1]:.6f}")
    print(f"最小 Loss: {np.min(loss_history):.6f} (步数 {np.argmin(loss_history)})")
    print(f"最大 Loss: {np.max(loss_history):.6f} (步数 {np.argmax(loss_history)})")
    print(f"平均 Loss: {np.mean(loss_history):.6f}")
    print(f"标准差: {np.std(loss_history):.6f}")

    # 最后 100 步的统计
    if len(loss_history) > 100:
        recent_loss = loss_history[-100:]
        print(f"\n最近 100 步:")
        print(f"  平均 Loss: {np.mean(recent_loss):.6f}")
        print(f"  标准差: {np.std(recent_loss):.6f}")

    print("=" * 60)

    # 保存图片
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Loss 曲线已保存: {save_path}")

    return fig, loss_history


def compare_loss_curves(checkpoint_paths, labels=None, save_path=None):
    """
    对比多个权重文件的 Loss 曲线

    参数:
        checkpoint_paths: 权重文件路径列表
        labels: 标签列表 (None 表示使用文件名)
        save_path: 保存图片路径
    """
    if labels is None:
        labels = [Path(p).stem for p in checkpoint_paths]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown']

    for idx, (checkpoint_path, label) in enumerate(zip(checkpoint_paths, labels)):
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            print(f"警告: 跳过不存在的文件: {checkpoint_path}")
            continue

        # 加载 Loss 历史
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        loss_history = checkpoint.get('loss_history', [])

        if len(loss_history) == 0:
            print(f"警告: {checkpoint_path} 没有 Loss 历史")
            continue

        color = colors[idx % len(colors)]
        steps = np.arange(len(loss_history))

        # 线性尺度
        axes[0].plot(steps, loss_history, color=color, linewidth=1.5, alpha=0.7, label=label)

        # 对数尺度
        axes[1].semilogy(steps, loss_history, color=color, linewidth=1.5, alpha=0.7, label=label)

    # 设置图表
    axes[0].set_xlabel('Training Step', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('GNN Training Loss Comparison', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Training Step', fontsize=12)
    axes[1].set_ylabel('Loss (log scale)', fontsize=12)
    axes[1].set_title('GNN Training Loss Comparison (Log Scale)', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 对比图已保存: {save_path}")

    return fig


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # 命令行指定权重文件
        checkpoint_path = sys.argv[1]
        save_path = 'results/gnn_loss_curve.png' if len(sys.argv) <= 2 else sys.argv[2]
    else:
        # 默认路径
        checkpoint_path = 'checkpoints/gnn_model.pth'
        save_path = 'results/gnn_loss_curve.png'

    # 绘制 Loss 曲线
    result = plot_loss_from_checkpoint(checkpoint_path, save_path)

    if result is not None:
        fig, loss_history = result
        if fig is not None:
            plt.show()
    else:
        sys.exit(1)
