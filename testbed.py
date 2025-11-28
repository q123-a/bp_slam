"""
BP-SLAM 主测试脚本
Main test script for BP-SLAM algorithm

Converted from MATLAB testbed.m
"""

import numpy as np
import scipy.io as sio
from bp_slam.utils.measurements import generate_measurements, generate_cluttered_measurements
from bp_slam.core.slam import bp_based_mint_slam


def main():
    """主测试函数"""

    # ---------------------------
    # 1. 通用参数及数据加载
    # ---------------------------
    parameters = {}
    parameters['known_track'] = 0  # 是否已知轨迹（0表示未知轨迹）

    # 加载场景数据，包括虚拟锚点 dataVA 和真实轨迹 trueTrajectory
    mat_data = sio.loadmat('scenarioCleanM2_new.mat')
    data_va_raw = mat_data['dataVA'][:, 0]  # 修复：获取所有传感器数据
    true_trajectory = mat_data['trueTrajectory']

    # 将所有锚点的可见性设置为全可见（1）
    num_sensors = len(data_va_raw)
    data_va = []
    for sensor in range(num_sensors):
        # 转换MATLAB结构体为Python字典
        sensor_data = {
            'positions': data_va_raw[sensor]['positions'][0, 0],
            'visibility': np.ones((data_va_raw[sensor]['positions'][0, 0].shape[1],
                                  true_trajectory.shape[1]))
        }
        data_va.append(sensor_data)

    # ---------------------------
    # 2. 算法参数配置
    # ---------------------------
    parameters['maxSteps'] = 900  # 最大时间步数
    true_trajectory = true_trajectory[:, :parameters['maxSteps']]  # 取前maxSteps个时间步的轨迹
    parameters['lengthStep'] = 0.03  # 单步移动距离（米）
    parameters['scanTime'] = 1  # 采样时间间隔（秒）

    # 最大速度和过程噪声方差计算
    v_max = parameters['lengthStep'] / parameters['scanTime']
    parameters['drivingNoiseVariance'] = (v_max / 3 / parameters['scanTime'])**2

    # 测量噪声参数
    parameters['measurementVariance'] = 0.1**2  # 距离测量方差
    parameters['measurementVarianceLHF'] = 0.15**2  # 后验测量方差（用于LHF）

    # 检测概率
    parameters['detectionProbability'] = 0.95

    # 区域尺寸及杂波相关参数
    parameters['regionOfInterestSize'] = 30  # 区域边长（米）
    parameters['meanNumberOfClutter'] = 1  # 平均误报数
    parameters['clutterIntensity'] = (parameters['meanNumberOfClutter'] /
                                     parameters['regionOfInterestSize'])  # 杂波强度

    # 新锚点出生率
    parameters['meanNumberOfBirth'] = 1e-4
    parameters['birthIntensity'] = (parameters['meanNumberOfBirth'] /
                                   (2 * parameters['regionOfInterestSize'])**2)

    # 未检测锚点强度
    parameters['meanNumberOfUndetectedAnchors'] = 6
    parameters['undetectedAnchorsIntensity'] = (parameters['meanNumberOfUndetectedAnchors'] /
                                               (2 * parameters['regionOfInterestSize'])**2)

    # 粒子滤波相关参数
    parameters['numParticles'] = 100000  # 粒子数量
    parameters['upSamplingFactor'] = 1  # 粒子上采样因子

    # SLAM相关阈值与先验
    parameters['detectionThreshold'] = 0.5
    parameters['survivalProbability'] = 0.999  # 锚点存活概率
    parameters['unreliabilityThreshold'] = 1e-4  # 锚点存在概率阈值，低于则删除
    parameters['priorKnownAnchors'] = [[0], [0]]  # 传感器已知锚点索引（Python从0开始）
    parameters['priorCovarianceAnchor'] = 0.001**2 * np.eye(2)  # 锚点位置先验协方差
    parameters['anchorRegularNoiseVariance'] = 1e-4**2  # 锚点过程噪声方差

    # agent参数（均匀采样半径）
    parameters['UniformRadius_pos'] = 0.5  # 初始位置均匀采样半径
    parameters['UniformRadius_vel'] = 0.05  # 初始速度均匀采样半径

    # ---------------------------
    # 3. 随机种子设置（保证结果可重复）
    # ---------------------------
    np.random.seed(1)

    # ---------------------------
    # 4. 移动体初始位置均值设定（真实轨迹起点）
    # ---------------------------
    parameters['priorMean'] = np.vstack([true_trajectory[0:2, 0:1], np.zeros((2, 1))])  # 初始位置+速度

    # ---------------------------
    # 5. 生成理想测量数据（无杂波）
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
    print("\n开始运行BP-SLAM算法...\n")
    print("=" * 50)
    (estimated_trajectory, estimated_anchors,
     posterior_particles_anchors, num_estimated_anchors) = bp_based_mint_slam(
        data_va, cluttered_measurements, parameters, true_trajectory
    )

    print("\n" + "=" * 50)
    print("算法运行完成！")
    print(f"最终估计的锚点数量 - 传感器1: {num_estimated_anchors[0, -1]}")
    if num_sensors > 1:
        print(f"最终估计的锚点数量 - 传感器2: {num_estimated_anchors[1, -1]}")

    # ---------------------------
    # 8. 保存结果
    # ---------------------------
    # 创建results文件夹
    from pathlib import Path
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    print("\n保存结果到 results/results.npz...")
    np.savez(results_dir / 'results.npz',
             estimated_trajectory=estimated_trajectory,
             true_trajectory=true_trajectory,
             num_estimated_anchors=num_estimated_anchors,
             estimated_anchors=np.array(estimated_anchors, dtype=object),
             posterior_particles_anchors=np.array(posterior_particles_anchors, dtype=object),
             parameters=parameters,
             allow_pickle=True)

    print("完成！结果已保存到 results/ 文件夹。")

    # ---------------------------
    # 9. 统一可视化模块（3个图表）
    # ---------------------------
    print("\n生成可视化图表...")
    from bp_slam.visualization.visualizer import visualize_online

    # 获取最后一个时间步的锚点粒子（如果有）
    last_particles = posterior_particles_anchors[-1] if len(posterior_particles_anchors) > 0 and posterior_particles_anchors[-1] is not None else None

    # 调用统一可视化模块
    stats = visualize_online(
        true_trajectory, estimated_trajectory, estimated_anchors,
        last_particles, data_va, parameters,
        scene_file='scen_semroom_new.mat',
        output_dir='results',
        save=True,
        show=True
    )

    print("\n提示：关闭图表窗口以继续...")
    import matplotlib.pyplot as plt
    plt.show()

    return estimated_trajectory, estimated_anchors, num_estimated_anchors


if __name__ == '__main__':
    main()
