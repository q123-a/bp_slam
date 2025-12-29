"""
训练诊断脚本
检查 GNN 训练中的常见问题
"""
import numpy as np
import scipy.io as sio
from scipy.optimize import linear_sum_assignment

print("=" * 80)
print("GNN 训练诊断")
print("=" * 80)

# 1. 加载数据
print("\n1. 加载数据...")
mat_data = sio.loadmat('scenarioCleanM2_new_1500.mat')
data_va_raw = mat_data['dataVA'][:, 0]
true_trajectory = mat_data['trueTrajectory']

mat_data2 = sio.loadmat('measurement1500.mat')
measurements_raw = mat_data2['estimated_measurements_cell']

print("✓ 数据加载完成")

# 2. 统计杂波比例（使用匈牙利算法推断）
print("\n2. 统计数据集杂波比例...")

num_steps = 100  # 只检查前100步
num_sensors = 2
SPEED_OF_LIGHT = 3.0e8
min_std = 0.05
variance_floor = min_std ** 2

total_measurements = 0
total_matched = 0
total_unmatched = 0

for step in range(num_steps):
    for sensor in range(num_sensors):
        mvalse_data = measurements_raw[step, sensor]
        if mvalse_data.size == 0:
            continue
