"""
检查 lwx.mat 文件的数据格式
"""
import scipy.io as sio
import numpy as np

# 加载MAT文件
mat_file = 'lwx.mat'
print(f"加载文件: {mat_file}")

try:
    mat_data = sio.loadmat(mat_file)

    print("\n文件中的变量:")
    for key in mat_data.keys():
        if not key.startswith('__'):
            data = mat_data[key]
            print(f"  - {key}: {type(data)}, shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")

    # 尝试找到测量数据
    possible_keys = ['estimated_measurements_cell', 'measurements', 'data', 'lwx']

    for key in possible_keys:
        if key in mat_data:
            print(f"\n✓ 找到变量 '{key}'")
            measurements_raw = mat_data[key]
            print(f"  - 形状: {measurements_raw.shape}")
            print(f"  - 数据类型: {measurements_raw.dtype}")

            # 如果是cell数组
            if measurements_raw.dtype == 'object':
                print(f"  - 这是一个cell数组")

                # 检查第一个非空元素
                print(f"\n检查前几个元素的数据格式:")
                found = False
                for i in range(min(5, measurements_raw.size)):
                    if measurements_raw.flat[i] is not None and hasattr(measurements_raw.flat[i], 'shape'):
                        if measurements_raw.flat[i].size > 0:
                            data = measurements_raw.flat[i]
                            print(f"\n  元素 {i}:")
                            print(f"    - 形状: {data.shape}")
                            print(f"    - 数据类型: {data.dtype}")

                            if len(data.shape) == 2:
                                print(f"    - 行数: {data.shape[0]}, 列数: {data.shape[1]}")
                                for row in range(min(3, data.shape[0])):
                                    print(f"    - 第{row}行 (前5个值): {data[row, :min(5, data.shape[1])]}")

                            found = True
                            break

                if not found:
                    print("  ⚠ 未找到非空数据")

            # 如果是普通数组
            else:
                print(f"  - 这是一个普通数组")
                if len(measurements_raw.shape) >= 2:
                    print(f"\n数据预览:")
                    for i in range(min(3, measurements_raw.shape[0])):
                        print(f"  第{i}行 (前5个值): {measurements_raw[i, :min(5, measurements_raw.shape[1])]}")

            # 统计幅度数据（假设在某一行）
            print(f"\n尝试统计幅度数据:")
            if measurements_raw.dtype == 'object':
                # Cell数组：收集所有数据
                amp_samples = []
                for i in range(measurements_raw.size):
                    if measurements_raw.flat[i] is not None and hasattr(measurements_raw.flat[i], 'shape'):
                        data = measurements_raw.flat[i]
                        if data.size > 0 and len(data.shape) == 2 and data.shape[0] >= 3:
                            amp_samples.extend(data[2, :].tolist())

                if len(amp_samples) > 0:
                    amp_samples = np.array(amp_samples)
                    print(f"  假设幅度在第2行:")
                    print(f"    - 样本数: {len(amp_samples)}")
                    print(f"    - 均值: {np.mean(amp_samples):.4f}")
                    print(f"    - 标准差: {np.std(amp_samples):.4f}")
                    print(f"    - 范围: [{np.min(amp_samples):.4f}, {np.max(amp_samples):.4f}]")
                    print(f"    - 前10个值: {amp_samples[:10]}")

            break
    else:
        print(f"\n✗ 未找到常见的测量数据变量")
        print(f"可用的变量: {[k for k in mat_data.keys() if not k.startswith('__')]}")

except FileNotFoundError:
    print(f"✗ 文件未找到: {mat_file}")
    print(f"请确认文件路径是否正确")
except Exception as e:
    print(f"✗ 加载文件时出错: {e}")
