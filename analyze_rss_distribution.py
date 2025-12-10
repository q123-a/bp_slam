#!/usr/bin/env python3
"""
分析measurement1500.mat中的RSS差异分布
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# 物理常数
c = 3.0e8
f_carrier = 28e9
P_ref = 10 * np.log10(c / (4 * np.pi * f_carrier))

print("=" * 70)
print("分析measurement1500.mat中的RSS差异")
print("=" * 70)

# 1. 加载真实轨迹和锚点
mat_data = sio.loadmat('scenarioCleanM2_new_1500.mat')
true_trajectory = mat_data['trueTrajectory']
data_va = mat_data['dataVA'][:, 0]

# 2. 加载测量数据
meas_data = sio.loadmat('measurement1500.mat')
meas_cell = meas_data['estimated_measurements_cell']

num_steps = min(900, meas_cell.shape[0])
num_sensors = meas_cell.shape[1]

print(f"\n数据集信息:")
print(f"  步数: {num_steps}")
print(f"  传感器数: {num_sensors}")

# 3. 分析每个步骤的RSS差异
all_rss_diffs = []
all_geo_diffs = []

for step in range(num_steps):
    target_pos = true_trajectory[:2, step]

    for sensor in range(num_sensors):
        # 获取锚点位置
        anchor_positions = data_va[sensor]['positions'][0, 0]
        num_anchors = anchor_positions.shape[1]

        # 获取测量数据
        meas = meas_cell[step, sensor]
        if meas is None or meas.size == 0 or meas.shape[0] < 3:
            continue

        # 测量的距离和幅度
        z_meas = meas[0, :]  # 距离
        z_rss_linear = meas[2, :]  # 线性幅度

        # 转换为dB
        z_rss_power = z_rss_linear ** 2
        z_rss_power = np.maximum(z_rss_power, 1e-10)
        z_rss_db = 10 * np.log10(z_rss_power)

        # 计算每个锚点的真实距离和预测RSS
        for anchor_idx in range(num_anchors):
            anchor_pos = anchor_positions[:, anchor_idx]

            # 真实距离
            true_dist = np.linalg.norm(target_pos - anchor_pos)

            # 预测RSS
            rss_pred = P_ref - 10 * np.log10(max(true_dist, 0.1))

            # 与所有测量的RSS差异
            for meas_rss in z_rss_db:
                rss_diff = abs(rss_pred - meas_rss)
                all_rss_diffs.append(rss_diff)

            # 与所有测量的几何距离差异
            for meas_dist in z_meas:
                geo_diff = abs(true_dist - meas_dist)
                all_geo_diffs.append(geo_diff)

all_rss_diffs = np.array(all_rss_diffs)
all_geo_diffs = np.array(all_geo_diffs)

print(f"\n总样本数: {len(all_rss_diffs)}")

# 4. 统计分析
print(f"\nRSS差异统计 (dB):")
print(f"  最小值: {np.min(all_rss_diffs):.2f} dB")
print(f"  最大值: {np.max(all_rss_diffs):.2f} dB")
print(f"  平均值: {np.mean(all_rss_diffs):.2f} dB")
print(f"  中位数: {np.median(all_rss_diffs):.2f} dB")
print(f"  标准差: {np.std(all_rss_diffs):.2f} dB")

print(f"\n几何距离差异统计 (m):")
print(f"  最小值: {np.min(all_geo_diffs):.4f} m")
print(f"  最大值: {np.max(all_geo_diffs):.4f} m")
print(f"  平均值: {np.mean(all_geo_diffs):.4f} m")
print(f"  中位数: {np.median(all_geo_diffs):.4f} m")
print(f"  标准差: {np.std(all_geo_diffs):.4f} m")

# 5. 百分位数分析
percentiles = [50, 75, 90, 95, 99]
print(f"\nRSS差异百分位数:")
for p in percentiles:
    val = np.percentile(all_rss_diffs, p)
    count = np.sum(all_rss_diffs <= val)
    print(f"  {p}%: {val:.2f} dB ({count}/{len(all_rss_diffs)} = {100*count/len(all_rss_diffs):.1f}%)")

# 6. 不同阈值的覆盖率
thresholds = [2, 3, 4, 5, 6, 8, 10]
print(f"\n不同RSS阈值的覆盖率:")
for thresh in thresholds:
    count = np.sum(all_rss_diffs <= thresh)
    percentage = 100 * count / len(all_rss_diffs)
    print(f"  {thresh}dB: {count}/{len(all_rss_diffs)} ({percentage:.1f}%)")

# 7. 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1: RSS差异直方图
ax = axes[0, 0]
ax.hist(all_rss_diffs, bins=100, alpha=0.7, edgecolor='black')
ax.axvline(x=4, color='r', linestyle='--', linewidth=2, label='4dB阈值')
ax.axvline(x=np.median(all_rss_diffs), color='g', linestyle='--', linewidth=2, label=f'中位数: {np.median(all_rss_diffs):.2f}dB')
ax.set_xlabel('RSS差异 (dB)')
ax.set_ylabel('频数')
ax.set_title('RSS差异分布')
ax.legend()
ax.grid(True, alpha=0.3)

# 子图2: RSS差异累积分布
ax = axes[0, 1]
sorted_rss = np.sort(all_rss_diffs)
cumulative = np.arange(1, len(sorted_rss) + 1) / len(sorted_rss) * 100
ax.plot(sorted_rss, cumulative, linewidth=2)
ax.axvline(x=4, color='r', linestyle='--', linewidth=2, label='4dB阈值')
ax.axhline(y=50, color='g', linestyle='--', alpha=0.5)
ax.axhline(y=75, color='orange', linestyle='--', alpha=0.5)
ax.axhline(y=90, color='purple', linestyle='--', alpha=0.5)
ax.set_xlabel('RSS差异 (dB)')
ax.set_ylabel('累积百分比 (%)')
ax.set_title('RSS差异累积分布')
ax.legend()
ax.grid(True, alpha=0.3)

# 子图3: 几何距离差异直方图
ax = axes[1, 0]
ax.hist(all_geo_diffs, bins=100, alpha=0.7, edgecolor='black')
ax.axvline(x=1.0, color='r', linestyle='--', linewidth=2, label='1m阈值')
ax.axvline(x=np.median(all_geo_diffs), color='g', linestyle='--', linewidth=2, label=f'中位数: {np.median(all_geo_diffs):.4f}m')
ax.set_xlabel('几何距离差异 (m)')
ax.set_ylabel('频数')
ax.set_title('几何距离差异分布')
ax.legend()
ax.grid(True, alpha=0.3)

# 子图4: RSS vs 几何距离散点图
ax = axes[1, 1]
# 采样以避免过多点
sample_size = min(10000, len(all_rss_diffs))
sample_indices = np.random.choice(len(all_rss_diffs), sample_size, replace=False)
ax.scatter(all_geo_diffs[sample_indices], all_rss_diffs[sample_indices],
          alpha=0.3, s=1)
ax.axhline(y=4, color='r', linestyle='--', linewidth=2, label='RSS阈值: 4dB')
ax.axvline(x=1.0, color='b', linestyle='--', linewidth=2, label='几何阈值: 1m')
ax.set_xlabel('几何距离差异 (m)')
ax.set_ylabel('RSS差异 (dB)')
ax.set_title('RSS差异 vs 几何距离差异')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/rss_distribution_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ 可视化已保存: results/rss_distribution_analysis.png")

print("\n" + "=" * 70)
print("分析完成")
print("=" * 70)
