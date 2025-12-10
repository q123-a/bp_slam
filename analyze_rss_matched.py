#!/usr/bin/env python3
"""
分析measurement1500.mat中匹配锚点的RSS差异
只看几何上最近的锚点（最可能的匹配）
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# 物理常数
c = 3.0e8
f_carrier = 28e9
P_ref = 10 * np.log10(c / (4 * np.pi * f_carrier))

print("=" * 70)
print("分析匹配锚点的RSS差异（只看几何最近的锚点）")
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

# 3. 分析每个测量与其几何最近锚点的RSS差异
matched_rss_diffs = []
matched_geo_diffs = []

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

        # 对每个测量，找到几何上最近的锚点
        for meas_idx in range(len(z_meas)):
            meas_dist = z_meas[meas_idx]
            meas_rss = z_rss_db[meas_idx]

            # 计算每个锚点的真实距离
            min_geo_diff = float('inf')
            best_anchor_idx = -1

            for anchor_idx in range(num_anchors):
                anchor_pos = anchor_positions[:, anchor_idx]
                true_dist = np.linalg.norm(target_pos - anchor_pos)
                geo_diff = abs(true_dist - meas_dist)

                if geo_diff < min_geo_diff:
                    min_geo_diff = geo_diff
                    best_anchor_idx = anchor_idx

            # 计算最佳匹配锚点的预测RSS
            if best_anchor_idx >= 0:
                anchor_pos = anchor_positions[:, best_anchor_idx]
                true_dist = np.linalg.norm(target_pos - anchor_pos)

                # 预测RSS（基于真实距离）
                rss_pred = P_ref - 10 * np.log10(max(true_dist, 0.1))

                # RSS差异
                rss_diff = abs(rss_pred - meas_rss)

                matched_rss_diffs.append(rss_diff)
                matched_geo_diffs.append(min_geo_diff)

matched_rss_diffs = np.array(matched_rss_diffs)
matched_geo_diffs = np.array(matched_geo_diffs)

print(f"\n总匹配样本数: {len(matched_rss_diffs)}")

# 4. 统计分析
print(f"\n匹配锚点的RSS差异统计 (dB):")
print(f"  最小值: {np.min(matched_rss_diffs):.2f} dB")
print(f"  最大值: {np.max(matched_rss_diffs):.2f} dB")
print(f"  平均值: {np.mean(matched_rss_diffs):.2f} dB")
print(f"  中位数: {np.median(matched_rss_diffs):.2f} dB")
print(f"  标准差: {np.std(matched_rss_diffs):.2f} dB")

print(f"\n匹配锚点的几何距离差异统计 (m):")
print(f"  最小值: {np.min(matched_geo_diffs):.4f} m")
print(f"  最大值: {np.max(matched_geo_diffs):.4f} m")
print(f"  平均值: {np.mean(matched_geo_diffs):.4f} m")
print(f"  中位数: {np.median(matched_geo_diffs):.4f} m")
print(f"  标准差: {np.std(matched_geo_diffs):.4f} m")

# 5. 百分位数分析
percentiles = [50, 75, 90, 95, 99]
print(f"\nRSS差异百分位数:")
for p in percentiles:
    val = np.percentile(matched_rss_diffs, p)
    count = np.sum(matched_rss_diffs <= val)
    print(f"  {p}%: {val:.2f} dB ({count}/{len(matched_rss_diffs)} = {100*count/len(matched_rss_diffs):.1f}%)")

# 6. 不同阈值的覆盖率
thresholds = [2, 3, 4, 5, 6, 6.5, 7, 8, 10]
print(f"\n不同RSS阈值的覆盖率:")
for thresh in thresholds:
    count = np.sum(matched_rss_diffs <= thresh)
    percentage = 100 * count / len(matched_rss_diffs)
    print(f"  {thresh}dB: {count}/{len(matched_rss_diffs)} ({percentage:.1f}%)")

# 7. 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1: RSS差异直方图
ax = axes[0, 0]
ax.hist(matched_rss_diffs, bins=100, alpha=0.7, edgecolor='black')
ax.axvline(x=6.5, color='r', linestyle='--', linewidth=2, label='6.5dB threshold')
ax.axvline(x=np.median(matched_rss_diffs), color='g', linestyle='--', linewidth=2,
          label=f'Median: {np.median(matched_rss_diffs):.2f}dB')
ax.set_xlabel('RSS Difference (dB)')
ax.set_ylabel('Frequency')
ax.set_title('RSS Difference Distribution (Matched Anchors Only)')
ax.legend()
ax.grid(True, alpha=0.3)

# 子图2: RSS差异累积分布
ax = axes[0, 1]
sorted_rss = np.sort(matched_rss_diffs)
cumulative = np.arange(1, len(sorted_rss) + 1) / len(sorted_rss) * 100
ax.plot(sorted_rss, cumulative, linewidth=2)
ax.axvline(x=6.5, color='r', linestyle='--', linewidth=2, label='6.5dB threshold')
ax.axhline(y=50, color='g', linestyle='--', alpha=0.5, label='50%')
ax.axhline(y=75, color='orange', linestyle='--', alpha=0.5, label='75%')
ax.set_xlabel('RSS Difference (dB)')
ax.set_ylabel('Cumulative Percentage (%)')
ax.set_title('RSS Difference Cumulative Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 子图3: 几何距离差异直方图
ax = axes[1, 0]
ax.hist(matched_geo_diffs, bins=100, alpha=0.7, edgecolor='black')
ax.axvline(x=1.0, color='r', linestyle='--', linewidth=2, label='1m threshold')
ax.axvline(x=np.median(matched_geo_diffs), color='g', linestyle='--', linewidth=2,
          label=f'Median: {np.median(matched_geo_diffs):.3f}m')
ax.set_xlabel('Geometric Distance Difference (m)')
ax.set_ylabel('Frequency')
ax.set_title('Geometric Distance Difference Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 子图4: RSS vs 几何距离散点图
ax = axes[1, 1]
sample_size = min(10000, len(matched_rss_diffs))
sample_indices = np.random.choice(len(matched_rss_diffs), sample_size, replace=False)
ax.scatter(matched_geo_diffs[sample_indices], matched_rss_diffs[sample_indices],
          alpha=0.3, s=1)
ax.axhline(y=6.5, color='r', linestyle='--', linewidth=2, label='RSS threshold: 6.5dB')
ax.axvline(x=1.0, color='b', linestyle='--', linewidth=2, label='Geo threshold: 1m')
ax.set_xlabel('Geometric Distance Difference (m)')
ax.set_ylabel('RSS Difference (dB)')
ax.set_title('RSS Difference vs Geometric Distance Difference')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/rss_matched_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved: results/rss_matched_analysis.png")

print("\n" + "=" * 70)
print("Analysis Complete")
print("=" * 70)
