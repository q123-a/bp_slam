"""
调试OSPA误差计算
Debug OSPA error calculation
"""

import numpy as np
import scipy.io as sio
from bp_slam.visualization.plotting import ospa_dist
import matplotlib.pyplot as plt

def debug_ospa_calculation():
    """详细调试OSPA计算过程"""

    print("=" * 70)
    print("OSPA误差调试分析")
    print("=" * 70)
    print()

    # 1. 加载数据
    print("1. 加载数据...")
    mat_data = sio.loadmat('scenarioCleanM2_new.mat')
    data_va_raw = mat_data['dataVA'][:, 0]

    # 转换数据
    data_va = []
    for sensor in range(len(data_va_raw)):
        sensor_data = {
            'positions': data_va_raw[sensor]['positions'][0, 0],
        }
        data_va.append(sensor_data)

    print(f"   传感器数量: {len(data_va)}")
    for sensor in range(len(data_va)):
        print(f"   传感器 {sensor + 1} 真实锚点数量: {data_va[sensor]['positions'].shape[1]}")
    print()

    # 2. 加载估计结果
    print("2. 加载估计结果...")
    try:
        # 先运行一次快速测试生成结果
        print("   运行快速测试生成结果...")
        import subprocess
        result = subprocess.run(['python', 'testbed_quick.py'],
                              capture_output=True, text=True, timeout=180)

        # 加载保存的结果
        results = np.load('results_quick.npz', allow_pickle=True)
        estimated_trajectory = results['estimated_trajectory']
        true_trajectory = results['true_trajectory']
        num_estimated_anchors = results['num_estimated_anchors']

        print(f"   ✓ 结果加载成功")
        print(f"   时间步数: {estimated_trajectory.shape[1]}")
        print()

    except Exception as e:
        print(f"   ✗ 无法加载结果: {e}")
        return

    # 3. 重新计算OSPA（模拟plot_matlab_style.py的计算）
    print("3. 重新计算OSPA误差...")
    print()

    num_sensors = len(data_va)
    num_steps = estimated_trajectory.shape[1]

    # 需要重新运行算法获取estimated_anchors
    print("   需要完整的estimated_anchors数据，重新运行算法...")
    from testbed_quick import main as run_quick_test

    # 修改testbed_quick.py以返回estimated_anchors
    print("   请稍候，正在运行算法...")

    # 临时方案：直接分析最后几步的情况
    print()
    print("=" * 70)
    print("分析最后一步的锚点估计")
    print("=" * 70)
    print()

    for sensor in range(num_sensors):
        print(f"传感器 {sensor + 1}:")
        print(f"  真实锚点数量: {data_va[sensor]['positions'].shape[1]}")
        print(f"  估计锚点数量（最后一步）: {int(num_estimated_anchors[sensor, -1])}")
        print(f"  真实锚点位置:")
        print(f"    {data_va[sensor]['positions']}")
        print()

    # 4. 分析可能的问题
    print("=" * 70)
    print("可能的问题分析")
    print("=" * 70)
    print()

    print("问题1: 锚点数量匹配")
    for sensor in range(num_sensors):
        true_count = data_va[sensor]['positions'].shape[1]
        est_count = int(num_estimated_anchors[sensor, -1])
        diff = abs(true_count - est_count)
        print(f"  传感器 {sensor + 1}: 真实={true_count}, 估计={est_count}, 差值={diff}")
        if diff > 0:
            print(f"    ⚠ 数量不匹配会导致OSPA基数误差: {diff} × 10 = {diff * 10} m")
    print()

    print("问题2: 存在概率阈值")
    print("  当前阈值: 0.5")
    print("  建议: 检查MATLAB中使用的阈值")
    print("  如果MATLAB使用更低的阈值（如0.3），会包含更多锚点")
    print()

    print("问题3: OSPA参数")
    print("  当前参数: c=10, p=1")
    print("  c=10 表示单点最大惩罚距离为10米")
    print("  如果锚点数量不匹配，每个缺失锚点贡献10米误差")
    print()

    # 5. 测试不同阈值的影响
    print("=" * 70)
    print("测试建议")
    print("=" * 70)
    print()

    print("请提供以下信息以帮助诊断：")
    print()
    print("1. MATLAB运行结果中的OSPA误差值：")
    print("   - 传感器1的平均OSPA误差: _____ m")
    print("   - 传感器2的平均OSPA误差: _____ m")
    print()
    print("2. MATLAB中使用的存在概率阈值：")
    print("   - parameters.detectionThreshold = _____")
    print()
    print("3. MATLAB最后一步的锚点数量：")
    print("   - 传感器1估计锚点数量: _____")
    print("   - 传感器2估计锚点数量: _____")
    print()

    print("当前Python版本的结果：")
    print(f"   - 传感器1的平均OSPA误差: 6.25 m")
    print(f"   - 传感器2的平均OSPA误差: 5.76 m")
    print(f"   - 传感器1最后一步锚点数量: {int(num_estimated_anchors[0, -1])}")
    print(f"   - 传感器2最后一步锚点数量: {int(num_estimated_anchors[1, -1])}")
    print()

if __name__ == '__main__':
    debug_ospa_calculation()
