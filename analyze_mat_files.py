"""
分析 MAT 文件的数据结构
"""
import scipy.io as sio
import numpy as np

print("=" * 80)
print("分析 scenarioCleanM2_new_1500.mat")
print("=" * 80)

# 加载场景文件
mat_data = sio.loadmat('scenarioCleanM2_new_1500.mat')

print("\n文件中的变量:")
for key in mat_data.keys():
    if not key.startswith('__'):
        print(f"  - {key}: {type(mat_data[key])}, shape={mat_data[key].shape if hasattr(mat_data[key], 'shape') else 'N/A'}")

# 分析 dataVA
print("\n\n--- dataVA (虚拟锚点数据) ---")
data_va_raw = mat_data['dataVA'][:, 0]
print(f"传感器数量: {len(data_va_raw)}")

for sensor_idx in range(min(2, len(data_va_raw))):
    sensor_data = data_va_raw[sensor_idx]
    positions = sensor_data['positions'][0, 0]
    print(f"\n传感器 {sensor_idx}:")
    print(f"  - positions shape: {positions.shape}")
    print(f"  - 锚点数量: {positions.shape[1]}")
    print(f"  - 前3个锚点位置:\n{positions[:, :3]}")

# 分析 trueTrajectory
print("\n\n--- trueTrajectory (真实轨迹) ---")
true_trajectory = mat_data['trueTrajectory']
print(f"shape: {true_trajectory.shape}")
print(f"时间步数: {true_trajectory.shape[1]}")
print(f"维度: {true_trajectory.shape[0]} (通常是 [x, y, vx, vy])")
print(f"前5个时间步:\n{true_trajectory[:, :5]}")

print("\n\n" + "=" * 80)
print("分析 measurement1500.mat")
print("=" * 80)

# 加载测量文件
mat_data2 = sio.loadmat('measurement1500.mat')

print("\n文件中的变量:")
for key in mat_data2.keys():
    if not key.startswith('__'):
        print(f"  - {key}: {type(mat_data2[key])}, shape={mat_data2[key].shape if hasattr(mat_data2[key], 'shape') else 'N/A'}")

# 分析 estimated_measurements_cell
print("\n\n--- estimated_measurements_cell (测量数据) ---")
measurements_raw = mat_data2['estimated_measurements_cell']
num_steps, num_sensors = measurements_raw.shape
print(f"shape: {measurements_raw.shape}")
print(f"时间步数: {num_steps}")
print(f"传感器数量: {num_sensors}")

# 统计每个时间步的测量数量
print("\n前10个时间步的测量数量:")
for step in range(min(10, num_steps)):
    for sensor in range(num_sensors):
        meas = measurements_raw[step, sensor]
        if meas.size > 0:
            print(f"  步骤 {step}, 传感器 {sensor}: {meas.shape[1]} 个测量")
            if step == 0 and sensor == 0:
                print(f"    数据格式: shape={meas.shape}")
                print(f"    第0行 (时延): {meas[0, :3]} ...")
                print(f"    第1行 (噪声功率): {meas[1, :3]} ...")
                print(f"    第2行 (信号幅度): {meas[2, :3]} ...")

# 统计整体信息
total_measurements = 0
for step in range(num_steps):
    for sensor in range(num_sensors):
        meas = measurements_raw[step, sensor]
        if meas.size > 0:
            total_measurements += meas.shape[1]

print(f"\n总测量数量: {total_measurements}")
print(f"平均每步测量数: {total_measurements / num_steps:.2f}")

print("\n\n" + "=" * 80)
print("数据结构总结")
print("=" * 80)

print("""
1. scenarioCleanM2_new_1500.mat:
   - dataVA: 虚拟锚点数据
     * 包含每个传感器的锚点位置 (2, K)
     * K 是锚点数量
   - trueTrajectory: 真实轨迹 (4, T)
     * T 是时间步数 (1500)
     * 4维: [x, y, vx, vy]

2. measurement1500.mat:
   - estimated_measurements_cell: 测量数据 (T, num_sensors)
     * 每个元素是 (3, M) 的数组
     * M 是该时刻该传感器的测量数量
     * 3维:
       - 第0行: 时延 (s) → 需要转换为距离 (m)
       - 第1行: 噪声功率 (power) → 不使用，用固定方差代替
       - 第2行: 信号幅度 (amplitude) → 用于 GNN 特征 (RSS)

3. 数据流:
   - scenarioCleanM2_new_1500.mat 提供场景信息（锚点位置、真实轨迹）
   - measurement1500.mat 提供预先生成的测量数据
   - 测量数据是从物理仿真生成的，包含时延和信号幅度
   - testbed.py 加载这些数据并转换为 SLAM 算法需要的格式
""")
