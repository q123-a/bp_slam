#!/usr/bin/env python3
"""
绘制 GNN 训练过程中的 Loss 变化曲线

用法:
    python plot_loss.py                           # 绘制 results/results_gnn.npz 的 loss
    python plot_loss.py results/V3/results_gnn_v3.npz  # 绘制指定文件的 loss
    python plot_loss.py file1.npz file2.npz       # 对比多个文件的 loss
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path


def plot_loss_from_npz(npz_files, save_path=None):
    """
    从 .npz 文件中读取并绘制 loss 曲线

    参数:
        npz_files: str 或 list，.npz 文件路径
        save_path: str，保存图片的路径（可选）
    """
    if isinstance(npz_files, str):
        npz_files = [npz_files]

    plt.figure(figsize=(12, 6))

    for npz_file in npz_files:
        npz_path = Path(npz_file)

        if not npz_path.exists():
            print(f"错误: 文件不存在 - {npz_file}")
            continue

        # 加载数据
        data = np.load(npz_file, allow_pickle=True)

        # 检查是否包含 loss_history
        if 'loss_history' not in data:
            print(f"警告: {npz_file} 中没有 loss_history 数据")
            print(f"可用的键: {list(data.keys())}")
            continue

        loss_history = data['loss_history']

        # 获取文件名作为标签
        file_label = npz_path.stem  # 例如 "results_gnn_v3"

        # 检查 loss_history 的维度
        if loss_history.ndim == 1:
            # 新格式：一维数组 (num_steps,) - 所有传感器的平均 loss
            num_steps = loss_history.shape[0]
            loss = loss_history

            # 过滤掉 loss=0 的步骤（可能是纯BP模式或未使用GNN的步骤）
            valid_mask = loss > 0
            valid_steps = np.where(valid_mask)[0]
            valid_loss = loss[valid_mask]

            if len(valid_loss) == 0:
                print(f"警告: {npz_file} 没有有效的 loss 数据")
                continue

            # 绘制曲线
            plt.plot(valid_steps, valid_loss, label=file_label, alpha=0.8, linewidth=1.5)

            # 打印统计信息
            print(f"\n{file_label}:")
            print(f"  有效步数: {len(valid_loss)}/{num_steps}")
            print(f"  Loss 范围: [{valid_loss.min():.4f}, {valid_loss.max():.4f}]")
            print(f"  平均 Loss: {valid_loss.mean():.4f}")
            print(f"  最终 Loss: {valid_loss[-1]:.4f}")

        elif loss_history.ndim == 2:
            # 旧格式：二维数组 (num_sensors, num_steps) - 每个传感器独立的 loss
            num_sensors, num_steps = loss_history.shape

            # 绘制每个传感器的 loss
            for sensor in range(num_sensors):
                loss = loss_history[sensor, :]

                # 过滤掉 loss=0 的步骤
                valid_mask = loss > 0
                valid_steps = np.where(valid_mask)[0]
                valid_loss = loss[valid_mask]

                if len(valid_loss) == 0:
                    print(f"警告: {npz_file} 传感器 {sensor+1} 没有有效的 loss 数据")
                    continue

                # 绘制曲线
                label = f"{file_label} - Sensor {sensor+1}"
                if num_sensors == 1:
                    label = file_label

                plt.plot(valid_steps, valid_loss, label=label, alpha=0.8, linewidth=1.5)

                # 打印统计信息
                print(f"\n{file_label} - 传感器 {sensor+1}:")
                print(f"  有效步数: {len(valid_loss)}/{num_steps}")
                print(f"  Loss 范围: [{valid_loss.min():.4f}, {valid_loss.max():.4f}]")
                print(f"  平均 Loss: {valid_loss.mean():.4f}")
                print(f"  最终 Loss: {valid_loss[-1]:.4f}")
        else:
            print(f"错误: {npz_file} 的 loss_history 维度不正确: {loss_history.shape}")
            continue

    # 添加45步预热分界线
    plt.axvline(x=45, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Warmup End (Step 45)')

    # 添加预热区域的背景色
    plt.axvspan(0, 45, alpha=0.1, color='red', label='Warmup Period')

    # 设置图表样式
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('GNN Training Loss over Time', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n图表已保存到: {save_path}")
    else:
        # 自动保存到本地
        default_save_path = 'results/loss_curve.png'
        plt.savefig(default_save_path, dpi=300, bbox_inches='tight')
        print(f"\n图表已保存到: {default_save_path}")
        plt.show()


def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 使用命令行参数指定的文件
        npz_files = sys.argv[1:]
    else:
        # 默认使用 results/results_gnn.npz
        npz_files = ['results/results_gnn.npz']

    print("=" * 60)
    print("GNN Loss 曲线绘制工具")
    print("=" * 60)
    print(f"\n读取文件: {npz_files}")

    # 绘制 loss 曲线
    plot_loss_from_npz(npz_files)


if __name__ == '__main__':
    main()
