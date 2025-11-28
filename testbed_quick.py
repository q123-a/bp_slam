"""
BP-SLAM 快速测试脚本（减少步数和粒子数以加快测试）
Quick test script for BP-SLAM (reduced steps and particles for faster testing)
"""

import numpy as np
import scipy.io as sio
from bp_slam.utils.measurements import generate_measurements, generate_cluttered_measurements
from bp_slam.core.slam import bp_based_mint_slam


def main():
    """快速测试函数"""

    print("=" * 60)
    print("BP-SLAM 快速测试模式")
    print("Quick Test Mode: 100 steps, 10000 particles")
    print("=" * 60)

    # ---------------------------
    # 1. 通用参数及数据加载
    # ---------------------------
    parameters = {}
    parameters['known_track'] = 0  # 是否已知轨迹（0表示未知轨迹）

    # 加载场景数据
    print("\n加载数据文件...")
    mat_data = sio.loadmat('scenarioCleanM2_new.mat')
    data_va_raw = mat_data['dataVA'][:, 0]  # 修复：获取所有传感器数据
    true_trajectory = mat_data['trueTrajectory']

    # 将所有锚点的可见性设置为全可见（1）
    num_sensors = len(data_va_raw)
    data_va = []
    for sensor in range(num_sensors):
        sensor_data = {
            'positions': data_va_raw[sensor]['positions'][0, 0],
            'visibility': np.ones((data_va_raw[sensor]['positions'][0, 0].shape[1],
                                  true_trajectory.shape[1]))
        }
        data_va.append(sensor_data)

    # ---------------------------
    # 2. 算法参数配置（快速测试版本）
    # ---------------------------
    parameters['maxSteps'] = 100  # 减少到100步（原900步）
    true_trajectory = true_trajectory[:, :parameters['maxSteps']]
    parameters['lengthStep'] = 0.03
    parameters['scanTime'] = 1

    v_max = parameters['lengthStep'] / parameters['scanTime']
    parameters['drivingNoiseVariance'] = (v_max / 3 / parameters['scanTime'])**2

    parameters['measurementVariance'] = 0.1**2
    parameters['measurementVarianceLHF'] = 0.15**2
    parameters['detectionProbability'] = 0.95

    parameters['regionOfInterestSize'] = 30
    parameters['meanNumberOfClutter'] = 1
    parameters['clutterIntensity'] = (parameters['meanNumberOfClutter'] /
                                     parameters['regionOfInterestSize'])

    parameters['meanNumberOfBirth'] = 1e-4
    parameters['birthIntensity'] = (parameters['meanNumberOfBirth'] /
                                   (2 * parameters['regionOfInterestSize'])**2)

    parameters['meanNumberOfUndetectedAnchors'] = 6
    parameters['undetectedAnchorsIntensity'] = (parameters['meanNumberOfUndetectedAnchors'] /
                                               (2 * parameters['regionOfInterestSize'])**2)

    # 快速测试：减少粒子数
    parameters['numParticles'] = 10000  # 减少到10000（原100000）
    parameters['upSamplingFactor'] = 1

    parameters['detectionThreshold'] = 0.5
    parameters['survivalProbability'] = 0.999
    parameters['unreliabilityThreshold'] = 1e-4
    parameters['priorKnownAnchors'] = [[0], [0]]
    parameters['priorCovarianceAnchor'] = 0.001**2 * np.eye(2)
    parameters['anchorRegularNoiseVariance'] = 1e-4**2

    parameters['UniformRadius_pos'] = 0.5
    parameters['UniformRadius_vel'] = 0.05

    # ---------------------------
    # 3. 随机种子设置
    # ---------------------------
    np.random.seed(1)

    # ---------------------------
    # 4. 移动体初始位置均值设定
    # ---------------------------
    parameters['priorMean'] = np.vstack([true_trajectory[0:2, 0:1], np.zeros((2, 1))])

    # ---------------------------
    # 5. 生成理想测量数据
    # ---------------------------
    print("生成理想测量数据...")
    measurements = generate_measurements(true_trajectory, data_va, parameters)

    # ---------------------------
    # 6. 加入误报和漏检，生成带杂波测量
    # ---------------------------
    print("生成带杂波测量数据...")
    cluttered_measurements = generate_cluttered_measurements(measurements, parameters)

    # ---------------------------
    # 7. 调用核心BP-SLAM算法进行估计
    # ---------------------------
    print("\n开始运行BP-SLAM算法（快速测试模式）...\n")
    print("=" * 60)
    (estimated_trajectory, estimated_anchors,
     posterior_particles_anchors, num_estimated_anchors) = bp_based_mint_slam(
        data_va, cluttered_measurements, parameters, true_trajectory
    )

    print("\n" + "=" * 60)
    print("算法运行完成！")
    print(f"最终估计的锚点数量 - 传感器1: {num_estimated_anchors[0, -1]}")
    if num_sensors > 1:
        print(f"最终估计的锚点数量 - 传感器2: {num_estimated_anchors[1, -1]}")

    # 计算最终位置误差
    final_error = np.linalg.norm(
        true_trajectory[0:2, -1] - estimated_trajectory[0:2, -1]
    )
    print(f"最终位置误差: {final_error:.4f} m")

    # 计算平均位置误差
    errors = np.linalg.norm(
        true_trajectory[0:2, :] - estimated_trajectory[0:2, :], axis=0
    )
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    print(f"平均位置误差: {mean_error:.4f} m")
    print(f"最大位置误差: {max_error:.4f} m")

    # ---------------------------
    # 8. 保存结果
    # ---------------------------
    print("\n保存结果到 results_quick.npz...")
    np.savez('results_quick.npz',
             estimated_trajectory=estimated_trajectory,
             true_trajectory=true_trajectory,
             num_estimated_anchors=num_estimated_anchors,
             estimated_anchors=np.array(estimated_anchors, dtype=object),
             posterior_particles_anchors=np.array(posterior_particles_anchors, dtype=object),
             parameters=parameters,
             allow_pickle=True)

    print("\n" + "=" * 60)
    print("✓ 快速测试完成！")
    print("=" * 60)

    # ---------------------------
    # 9. MATLAB风格可视化（3个图表）
    # ---------------------------
    print("\n生成MATLAB风格可视化...")
    from plot_matlab_style import plot_all_matlab_style

    # 获取最后一个时间步的锚点粒子（如果有）
    last_particles = posterior_particles_anchors[-1] if len(posterior_particles_anchors) > 0 and posterior_particles_anchors[-1] is not None else None

    plot_all_matlab_style(
        true_trajectory, estimated_trajectory, estimated_anchors,
        last_particles, num_estimated_anchors,
        data_va, parameters, mode=0, num_steps=parameters['maxSteps']
    )

    print("\n提示：")
    print("- 如需完整运行（900步，100000粒子），请运行: python testbed.py")
    print("- 关闭图表窗口以继续...")

    import matplotlib.pyplot as plt
    plt.show()

    return estimated_trajectory, estimated_anchors, num_estimated_anchors


if __name__ == '__main__':
    main()
