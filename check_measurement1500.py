#!/usr/bin/env python3
"""
检查 measurement1500.mat 数据集格式
验证幅度信息是否存在
"""

import numpy as np
import scipy.io as sio

def check_measurement1500():
    """检查 measurement1500.mat 数据集"""
    print("=" * 60)
    print("检查 measurement1500.mat 数据集")
    print("=" * 60)

    try:
        # 加载数据
        mat_data = sio.loadmat('measurement1500.mat')
        print("✓ 数据文件加载成功\n")

        # 显示所有键
        print("数据集包含的键:")
        for key in mat_data.keys():
            if not key.startswith('__'):
                print(f"  - {key}: {type(mat_data[key])}, shape: {mat_data[key].shape if hasattr(mat_data[key], 'shape') else 'N/A'}")

        # 检查测量数据
        measurement_key = None
        if 'measurements' in mat_data:
            measurement_key = 'measurements'
        elif 'estimated_measurements_cell' in mat_data:
            measurement_key = 'estimated_measurements_cell'

        if measurement_key:
            measurements = mat_data[measurement_key]
            print(f"\n✓ 找到 '{measurement_key}' 字段")
            print(f"  - 形状: {measurements.shape}")
            print(f"  - 数据类型: {measurements.dtype}")

            # 检查数据结构
            num_steps, num_sensors = measurements.shape
            print(f"  - 时间步数: {num_steps}")
            print(f"  - 传感器数: {num_sensors}")

            # 检查第一个非空测量
            print(f"\n检查测量数据格式:")
            found_data = False
            for step in range(min(10, num_steps)):
                for sensor in range(num_sensors):
                    data = measurements[step, sensor]
                    if data.size > 0:
                        print(f"\n  时间步 {step}, 传感器 {sensor}:")
                        print(f"    - 形状: {data.shape}")
                        print(f"    - 行数: {data.shape[0]} (期望: 3 行 = [时延, 方差, 幅度])")
                        print(f"    - 检测数: {data.shape[1]}")

                        if data.shape[0] >= 3:
                            print(f"    ✓ 包含幅度信息 (第3行)")
                            print(f"    - 时延范围: [{data[0, :].min():.6f}, {data[0, :].max():.6f}] s")
                            print(f"    - 方差范围: [{data[1, :].min():.6f}, {data[1, :].max():.6f}]")
                            print(f"    - 幅度范围: [{data[2, :].min():.6f}, {data[2, :].max():.6f}]")
                            print(f"    - 幅度前5个值: {data[2, :5]}")

                            # 统计幅度分布
                            amp_data = []
                            for s in range(min(100, num_steps)):
                                for sen in range(num_sensors):
                                    d = measurements[s, sen]
                                    if d.size > 0 and d.shape[0] >= 3:
                                        amp_data.extend(d[2, :].tolist())

                            if len(amp_data) > 0:
                                amp_data = np.array(amp_data)
                                print(f"\n  幅度统计 (前100步):")
                                print(f"    - 样本数: {len(amp_data)}")
                                print(f"    - 均值: {np.mean(amp_data):.4f}")
                                print(f"    - 标准差: {np.std(amp_data):.4f}")
                                print(f"    - 范围: [{np.min(amp_data):.4f}, {np.max(amp_data):.4f}]")
                                print(f"    - 中位数: {np.median(amp_data):.4f}")
                        else:
                            print(f"    ✗ 缺少幅度信息 (只有 {data.shape[0]} 行)")

                        found_data = True
                        break
                if found_data:
                    break

            if not found_data:
                print("  ⚠️  前10步没有找到有效测量数据")

        else:
            print("\n✗ 未找到 'measurements' 字段")
            print("可用的字段:", [k for k in mat_data.keys() if not k.startswith('__')])

        return True

    except FileNotFoundError:
        print("✗ 文件不存在: measurement1500.mat")
        print("请确保文件在当前目录下")
        return False
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    check_measurement1500()
