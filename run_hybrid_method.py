"""
运行混合方案（图注意力 + 自适应RANSAC）
完整的使用示例
"""

import numpy as np
import scipy.io as sio
from bp_slam.utils.measurements import generate_measurements, generate_cluttered_measurements
from bp_slam.core.slam import bp_based_mint_slam


def run_experiment(method_name, use_graph_attention, use_ransac, max_steps=100):
    """
    运行单个实验

    参数:
        method_name: 方法名称（用于显示）
        use_graph_attention: 是否使用图注意力
        use_ransac: 是否使用RANSAC
        max_steps: 运行步数
    """
    print("\n" + "="*70)
    print(f"实验: {method_name}")
    print("="*70)

    # ============================================================
    # 1. 加载场景数据和测量数据
    # ============================================================
    print("加载场景数据...")

    # 加载场景数据（虚拟锚点和真实轨迹）
    scenario_data = sio.loadmat('scenarioCleanM2_new_1500.mat')
    data_va_raw = scenario_data['dataVA'][:, 0]
    true_trajectory = scenario_data['trueTrajectory']

    # 转换虚拟锚点数据格式
    num_sensors = len(data_va_raw)
    data_va = []
    for sensor in range(num_sensors):
        sensor_data = {
            'positions': data_va_raw[sensor]['positions'][0, 0],
            'visibility': np.ones((data_va_raw[sensor]['positions'][0, 0].shape[1],
                                  true_trajectory.shape[1]))
        }
        data_va.append(sensor_data)

    print(f"✓ 场景数据加载完成: {num_sensors} 个传感器")

    # 加载测量数据
    print("加载测量数据...")
    mat_data = sio.loadmat('measurementbadf.mat')
    measurements_raw = mat_data['estimated_measurements_cell']
    num_steps_total, _ = measurements_raw.shape

    # 转换数据格式
    SPEED_OF_LIGHT = 3.0e8
    min_std = 0.05
    variance_floor = min_std ** 2

    cluttered_measurements = [[None for _ in range(num_sensors)] for _ in range(num_steps_total)]

    for step in range(num_steps_total):
        for sensor in range(num_sensors):
            mvalse_data = measurements_raw[step, sensor]
            if mvalse_data.size == 0:
                cluttered_measurements[step][sensor] = np.zeros((2, 0))
            else:
                K_est = mvalse_data.shape[1]
                tracker_input = np.zeros((2, K_est))
                tracker_input[0, :] = mvalse_data[0, :] * SPEED_OF_LIGHT
                tracker_input[1, :] = variance_floor
                cluttered_measurements[step][sensor] = tracker_input

    print(f"✓ 测量数据加载完成: {num_steps_total} 步, {num_sensors} 个传感器")

    # 截取到max_steps
    true_trajectory = true_trajectory[:, :max_steps]
    cluttered_measurements = cluttered_measurements[:max_steps]

    # ============================================================
    # 2. 配置参数
    # ============================================================
    parameters = {}

    # 基本参数
    parameters['known_track'] = 0
    parameters['maxSteps'] = max_steps
    parameters['numParticles'] = 100000
    parameters['lengthStep'] = 0.03
    parameters['scanTime'] = 1

    # 速度和噪声参数
    v_max = parameters['lengthStep'] / parameters['scanTime']
    parameters['drivingNoiseVariance'] = (v_max / 3 / parameters['scanTime'])**2
    parameters['measurementVariance'] = 0.1**2
    parameters['measurementVarianceLHF'] = 0.15**2

    # 检测和存活概率
    parameters['detectionProbability'] = 0.95
    parameters['survivalProbability'] = 0.999

    # 区域和杂波参数
    parameters['regionOfInterestSize'] = 30
    parameters['meanNumberOfClutter'] = 1
    parameters['clutterIntensity'] = parameters['meanNumberOfClutter'] / parameters['regionOfInterestSize']

    # 新锚点出生率
    parameters['meanNumberOfBirth'] = 1e-4
    parameters['birthIntensity'] = parameters['meanNumberOfBirth'] / (2 * parameters['regionOfInterestSize'])**2

    # 未检测锚点强度
    parameters['meanNumberOfUndetectedAnchors'] = 6
    parameters['undetectedAnchorsIntensity'] = parameters['meanNumberOfUndetectedAnchors'] / (2 * parameters['regionOfInterestSize'])**2

    # SLAM相关阈值
    parameters['detectionThreshold'] = 0.5
    parameters['unreliabilityThreshold'] = 1e-4
    parameters['upSamplingFactor'] = 1

    # 锚点先验
    parameters['priorKnownAnchors'] = [[0], [0]]
    parameters['priorCovarianceAnchor'] = 0.001**2 * np.eye(2)
    parameters['anchorRegularNoiseVariance'] = 1e-4**2

    # 先验参数
    parameters['priorMean'] = np.vstack([true_trajectory[0:2, 0:1], np.zeros((2, 1))])
    parameters['UniformRadius_pos'] = 0.5
    parameters['UniformRadius_vel'] = 0.05

    # GNN参数
    parameters['use_gnn'] = True
    parameters['gnn_warmup_steps'] = 50
    parameters['gnn_hidden_dim'] = 64
    parameters['gnn_lr'] = 1e-4
    parameters['gnn_checkpoint_path'] = None

    # GNN改进参数
    parameters['gnn_use_ema'] = True
    parameters['gnn_ema_decay'] = 0.999
    parameters['gnn_use_lr_scheduler'] = True
    parameters['gnn_pseudo_label_mode'] = 'and'
    parameters['gnn_confidence_weighting'] = True
    parameters['gnn_use_temporal_gru'] = False
    parameters['gnn_use_layer_gru'] = False

    # ★★★ 新增参数：图注意力 + RANSAC ★★★
    parameters['gnn_use_graph_attention'] = use_graph_attention
    parameters['gnn_use_ransac'] = use_ransac
    parameters['gnn_ransac_threshold'] = 2.0  # RANSAC距离阈值（米）

    print(f"\n配置:")
    print(f"  - 图注意力: {use_graph_attention}")
    print(f"  - 自适应RANSAC: {use_ransac}")
    print(f"  - 运行步数: {max_steps}")
    print(f"  - 粒子数: {parameters['numParticles']}")

    # ============================================================
    # 3. 运行算法
    # ============================================================
    print("\n开始运行...")

    # 初始化虚拟锚点数据（空列表，算法会自动生成）
    data_va = [[] for _ in range(num_sensors)]

    # 运行SLAM算法
    estimated_trajectory, estimated_anchors, _, num_estimated_anchors = bp_based_mint_slam(
        data_va=data_va,
        cluttered_measurements=cluttered_measurements,
        parameters=parameters,
        true_trajectory=None  # 未知轨迹模式
    )

    print(f"\n✓ 实验完成!")
    print(f"  - 估计轨迹形状: {estimated_trajectory.shape}")
    print(f"  - 估计锚点数量: {num_estimated_anchors.sum()}")

    return estimated_trajectory, estimated_anchors, num_estimated_anchors


def main():
    """主函数：对比三种方案"""

    print("\n" + "="*70)
    print("图注意力 + 自适应RANSAC 对比实验")
    print("="*70)

    max_steps = 100  # 先测试100步，确认无误后可以增加到900

    # ============================================================
    # 方案1: 纯自监督（Baseline）
    # ============================================================
    traj_baseline, anchors_baseline, num_anchors_baseline = run_experiment(
        method_name="方案1: 纯自监督（Baseline）",
        use_graph_attention=False,
        use_ransac=False,
        max_steps=max_steps
    )

    # ============================================================
    # 方案2: 图注意力增强
    # ============================================================
    traj_attention, anchors_attention, num_anchors_attention = run_experiment(
        method_name="方案2: 图注意力增强",
        use_graph_attention=True,
        use_ransac=False,
        max_steps=max_steps
    )

    # ============================================================
    # 方案3: 混合方案（图注意力 + RANSAC）
    # ============================================================
    traj_hybrid, anchors_hybrid, num_anchors_hybrid = run_experiment(
        method_name="方案3: 混合方案（图注意力 + RANSAC）",
        use_graph_attention=True,
        use_ransac=True,
        max_steps=max_steps
    )

    # ============================================================
    # 总结对比
    # ============================================================
    print("\n" + "="*70)
    print("实验总结")
    print("="*70)

    print("\n估计锚点数量对比:")
    print(f"  方案1 (Baseline):  {num_anchors_baseline.sum()}")
    print(f"  方案2 (图注意力):  {num_anchors_attention.sum()}")
    print(f"  方案3 (混合方案):  {num_anchors_hybrid.sum()}")

    print("\n轨迹估计范围:")
    print(f"  方案1 (Baseline):  X=[{traj_baseline[0].min():.2f}, {traj_baseline[0].max():.2f}], "
          f"Y=[{traj_baseline[1].min():.2f}, {traj_baseline[1].max():.2f}]")
    print(f"  方案2 (图注意力):  X=[{traj_attention[0].min():.2f}, {traj_attention[0].max():.2f}], "
          f"Y=[{traj_attention[1].min():.2f}, {traj_attention[1].max():.2f}]")
    print(f"  方案3 (混合方案):  X=[{traj_hybrid[0].min():.2f}, {traj_hybrid[0].max():.2f}], "
          f"Y=[{traj_hybrid[1].min():.2f}, {traj_hybrid[1].max():.2f}]")

    print("\n" + "="*70)
    print("✓ 所有实验完成！")
    print("="*70)

    # 保存结果（可选）
    print("\n保存结果到 results/ 目录...")
    import os
    os.makedirs('results', exist_ok=True)

    np.save('results/trajectory_baseline.npy', traj_baseline)
    np.save('results/trajectory_attention.npy', traj_attention)
    np.save('results/trajectory_hybrid.npy', traj_hybrid)

    print("✓ 结果已保存")


if __name__ == "__main__":
    main()
