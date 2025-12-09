"""
检查MAT文件的数据格式
"""
import scipy.io as sio
import numpy as np

# 加载MAT文件
mat_file = 'measurementbadf.mat'
print(f"加载文件: {mat_file}")
mat_data = sio.loadmat(mat_file)

print("\n文件中的变量:")
for key in mat_data.keys():
    if not key.startswith('__'):
        print(f"  - {key}: {type(mat_data[key])}, shape: {mat_data[key].shape if hasattr(mat_data[key], 'shape') else 'N/A'}")

# 检查 estimated_measurements_cell
if 'estimated_measurements_cell' in mat_data:
    measurements_raw = mat_data['estimated_measurements_cell']
    num_steps, num_sensors = measurements_raw.shape
    print(f"\n✓ 找到 'estimated_measurements_cell'")
    print(f"  - 形状: ({num_steps} 步, {num_sensors} 个传感器)")

    # 检查第一个非空数据
    print(f"\n检查前几个时间步的数据格式:")
    for step in range(min(5, num_steps)):
        for sensor in range(num_sensors):
            data = measurements_raw[step, sensor]
            if data.size > 0:
                print(f"\n  Step {step}, Sensor {sensor}:")
                print(f"    - 数据形状: {data.shape}")
                print(f"    - 数据类型: {data.dtype}")
                if data.shape[0] <= 5:  # 如果行数不多，显示所有行
                    for i in range(data.shape[0]):
                        print(f"    - 第{i}行 (前5个值): {data[i, :min(5, data.shape[1])]}")
                else:
                    print(f"    - 第0行 (前5个值): {data[0, :min(5, data.shape[1])]}")
                    print(f"    - 第1行 (前5个值): {data[1, :min(5, data.shape[1])]}")
                    print(f"    - 第2行 (前5个值): {data[2, :min(5, data.shape[1])]}")
                break
        else:
            continue
        break

    # 统计幅度数据（假设在第2行）
    print(f"\n统计幅度数据（假设在第2行）:")
    amp_samples = []
    for step in range(min(10, num_steps)):
        for sensor in range(num_sensors):
            data = measurements_raw[step, sensor]
            if data.size > 0 and data.shape[0] >= 3:
                amp_samples.extend(data[2, :].tolist())

    if len(amp_samples) > 0:
        amp_samples = np.array(amp_samples)
        print(f"  - 样本数: {len(amp_samples)}")
        print(f"  - 均值: {np.mean(amp_samples):.4f}")
        print(f"  - 标准差: {np.std(amp_samples):.4f}")
        print(f"  - 范围: [{np.min(amp_samples):.4f}, {np.max(amp_samples):.4f}]")
        print(f"  - 前10个值: {amp_samples[:10]}")
    else:
        print(f"  ⚠ 未找到幅度数据")

else:
    print(f"\n✗ 未找到 'estimated_measurements_cell' 变量")
    print(f"可用的变量: {[k for k in mat_data.keys() if not k.startswith('__')]}")
