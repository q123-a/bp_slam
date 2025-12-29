"""
损失可视化脚本
Visualize GNN training loss from CSV file
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
from pathlib import Path


def load_loss_from_csv(csv_path):
    """从 CSV 文件加载损失历史"""
    steps = []
    total_loss = []
    quality_loss = []
    assoc_loss = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['Step']))
            total_loss.append(float(row['Total_Loss']))
            quality_loss.append(float(row['Quality_Loss']))
            assoc_loss.append(float(row['Assoc_Loss']))

    return np.array(steps), np.array(total_loss), np.array(quality_loss), np.array(assoc_loss)


def plot_loss_curves(steps, total_loss, quality_loss, assoc_loss,
                     save_path=None, window_size=10, show=True):
    """绘制损失曲线"""

    # 计算移动平均
    def moving_average(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GNN Training Loss History', fontsize=16, fontweight='bold')

    # 1. 总损失
    ax = axes[0, 0]
    ax.plot(steps, total_loss, alpha=0.3, label='Raw', color='blue')
    if len(total_loss) >= window_size:
        smoothed = moving_average(total_loss, window_size)
        ax.plot(steps[:len(smoothed)], smoothed, label=f'MA({window_size})',
                color='blue', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 质量损失
    ax = axes[0, 1]
    ax.plot(steps, quality_loss, alpha=0.3, label='Raw', color='green')
    if len(quality_loss) >= window_size:
        smoothed = moving_average(quality_loss, window_size)
        ax.plot(steps[:len(smoothed)], smoothed, label=f'MA({window_size})',
                color='green', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Quality Loss (Clutter Detection)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 关联损失
    ax = axes[1, 0]
    ax.plot(steps, assoc_loss, alpha=0.3, label='Raw', color='red')
    if len(assoc_loss) >= window_size:
        smoothed = moving_average(assoc_loss, window_size)
        ax.plot(steps[:len(smoothed)], smoothed, label=f'MA({window_size})',
                color='red', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Association Loss (Data Association)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 所有损失对比
    ax = axes[1, 1]
    if len(total_loss) >= window_size:
        total_smooth = moving_average(total_loss, window_size)
        quality_smooth = moving_average(quality_loss, window_size)
        assoc_smooth = moving_average(assoc_loss, window_size)

        ax.plot(steps[:len(total_smooth)], total_smooth,
                label='Total', color='blue', linewidth=2)
        ax.plot(steps[:len(quality_smooth)], quality_smooth,
                label='Quality', color='green', linewidth=2)
        ax.plot(steps[:len(assoc_smooth)], assoc_smooth,
                label='Association', color='red', linewidth=2)
    else:
        ax.plot(steps, total_loss, label='Total', color='blue', linewidth=2)
        ax.plot(steps, quality_loss, label='Quality', color='green', linewidth=2)
        ax.plot(steps, assoc_loss, label='Association', color='red', linewidth=2)

    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss Comparison (MA {window_size})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图形
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 损失曲线已保存: {save_path}")

    # 显示图形
    if show:
        plt.show()

    plt.close()


def print_loss_summary(steps, total_loss, quality_loss, assoc_loss):
    """打印损失统计摘要"""
    print("\n" + "=" * 80)
    print("损失统计摘要")
    print("=" * 80)

    print(f"\n训练步数: {len(steps)}")

    print(f"\n总损失 (Total Loss):")
    print(f"  初始值: {total_loss[0]:.4f}")
    print(f"  最终值: {total_loss[-1]:.4f}")
    print(f"  最小值: {np.min(total_loss):.4f} (步骤 {steps[np.argmin(total_loss)]})")
    print(f"  最大值: {np.max(total_loss):.4f} (步骤 {steps[np.argmax(total_loss)]})")
    print(f"  平均值: {np.mean(total_loss):.4f}")
    print(f"  标准差: {np.std(total_loss):.4f}")

    print(f"\n质量损失 (Quality Loss):")
    print(f"  初始值: {quality_loss[0]:.4f}")
    print(f"  最终值: {quality_loss[-1]:.4f}")
    print(f"  最小值: {np.min(quality_loss):.4f} (步骤 {steps[np.argmin(quality_loss)]})")
    print(f"  最大值: {np.max(quality_loss):.4f} (步骤 {steps[np.argmax(quality_loss)]})")
    print(f"  平均值: {np.mean(quality_loss):.4f}")

    print(f"\n关联损失 (Association Loss):")
    print(f"  初始值: {assoc_loss[0]:.4f}")
    print(f"  最终值: {assoc_loss[-1]:.4f}")
    print(f"  最小值: {np.min(assoc_loss):.4f} (步骤 {steps[np.argmin(assoc_loss)]})")
    print(f"  最大值: {np.max(assoc_loss):.4f} (步骤 {steps[np.argmax(assoc_loss)]})")
    print(f"  平均值: {np.mean(assoc_loss):.4f}")

    # 计算最后 50 步的平均值
    if len(steps) >= 50:
        print(f"\n最后 50 步平均:")
        print(f"  总损失: {np.mean(total_loss[-50:]):.4f}")
        print(f"  质量损失: {np.mean(quality_loss[-50:]):.4f}")
        print(f"  关联损失: {np.mean(assoc_loss[-50:]):.4f}")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='可视化 GNN 训练损失')
    parser.add_argument('--csv', type=str, default='results/loss_history.csv',
                       help='损失历史 CSV 文件路径')
    parser.add_argument('--output', type=str, default='results/loss_curves.png',
                       help='输出图像路径')
    parser.add_argument('--window', type=int, default=10,
                       help='移动平均窗口大小')
    parser.add_argument('--no-show', action='store_true',
                       help='不显示图形窗口')

    args = parser.parse_args()

    # 检查文件是否存在
    if not Path(args.csv).exists():
        print(f"❌ 错误: 找不到文件 {args.csv}")
        print(f"   请先运行训练生成损失历史文件")
        return

    # 加载损失数据
    print(f"加载损失历史: {args.csv}")
    steps, total_loss, quality_loss, assoc_loss = load_loss_from_csv(args.csv)
    print(f"✓ 成功加载 {len(steps)} 步的损失数据")

    # 打印统计摘要
    print_loss_summary(steps, total_loss, quality_loss, assoc_loss)

    # 绘制损失曲线
    print(f"绘制损失曲线...")
    plot_loss_curves(steps, total_loss, quality_loss, assoc_loss,
                    save_path=args.output, window_size=args.window,
                    show=not args.no_show)

    print("✓ 完成！")


if __name__ == '__main__':
    main()
