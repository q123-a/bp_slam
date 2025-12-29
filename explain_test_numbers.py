"""
解释测试数字的含义
"""
import numpy as np
import scipy.io as sio

print("=" * 80)
print("解释测试数字")
print("=" * 80)

# 加载数据
mat_data = sio.loadmat('scenarioCleanM2_new_1500.mat')
data_va_raw = mat_data['dataVA'][:, 0]

mat_data2 = sio.loadmat('measurement1500.mat')
measurements_raw = mat_data2['estimated_measurements_cell']

print("\n1. 原始测量数量分析 (前10步)")
print("-" * 80)

total_measurements = 0
for step in range(10):
    step_total = 0
    for sensor in range(2):
        meas = measurements_raw[step, sensor]
        if meas.size > 0:
            count = meas.shape[1]
            step_total += count
            print(f"  步骤 {step}, 传感器 {sensor}: {count} 个测量")
    total_measurements += step_total
    print(f"  步骤 {step} 总计: {step_total} 个测量")
    print()

print(f"前10步总测量数: {total_measurements}")
print(f"\n说明:")
print(f"  - 传感器0有 {data_va_raw[0]['positions'][0, 0].shape[1]} 个锚点")
print(f"  - 传感器1有 {data_va_raw[1]['positions'][0, 0].shape[1]} 个锚点")
print(f"  - 理论上每步应该有 6+5=11 个测量")
print(f"  - 10步理论上应该有 11×10=110 个测量")
print(f"  - 实际有 {total_measurements} 个测量")
print(f"  - 差异可能是由于物理仿真中的漏检")
