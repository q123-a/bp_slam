"""
BP-SLAM 主测试脚本（完整版）
Main test script for BP-SLAM algorithm

支持切换 BP 和 BP+GNN 模式
Converted from MATLAB testbed.m
"""

import numpy as np
import scipy.io as sio
from bp_slam.utils.measurements import generate_measurements, generate_cluttered_measurements
from bp_slam.core.slam import bp_based_mint_slam


def load_measurements_from_mat(mat_file='measurementbadf.mat'):
    """
    从 MAT 文件加载预先生成的检测数据，并应用自适应方差计算

    参数:
        mat_file: MAT 文件路径，默认为 'measurementbadf.mat'

    返回:
        cluttered_measurements: 检测数据，shape (num_steps, num_sensors) 的列表
                               每个元素是 (2, num_detections) 的数组（距离+方差）
    """
    print(f"从 {mat_file} 加载检测数据...")
    mat_data = sio.loadmat(mat_file)

    # 加载 estimated_measurements_cell，形状为 (num_steps, num_sensors)
    measurements_raw = mat_data['estimated_measurements_cell']
    num_steps, num_sensors = measurements_raw.shape

    # 光速常数 (m/s)
    SPEED_OF_LIGHT = 3.0e8

    # 固定方差参数
    min_std = 0.05
    variance_floor = min_std ** 2  # 0.0025 m^2

    # 初始化输出
    cluttered_measurements = [[None for _ in range(num_sensors)] for _ in range(num_steps)]

    # 转换数据格式
    for step in range(num_steps):
        for sensor in range(num_sensors):
            mvalse_data = measurements_raw[step, sensor]

            if mvalse_data.size == 0:
                # 如果没有检测数据，设置为空数组
                cluttered_measurements[step][sensor] = np.zeros((2, 0))
            else:
                # mvalse_data 形状为 (3, num_detections)
                # 第0行: 时延 (s) - 需要转换为距离
                # 第1行: 噪声方差估计 (nu) - 暂不使用
                # 第2行: 信号幅度 (amps) - 暂不使用
                K_est = mvalse_data.shape[1]
                tracker_input = np.zeros((2, K_est))

                # 1. 距离 (m) = 时延 (s) × 光速 (m/s)
                tracker_input[0, :] = mvalse_data[0, :] * SPEED_OF_LIGHT

                # 2. 使用固定方差 0.0025 m^2 (标准差 0.05 m)
                tracker_input[1, :] = variance_floor
                cluttered_measurements[step][sensor] = tracker_input

    print(f"✓ 成功加载检测数据: {num_steps} 步, {num_sensors} 个传感器")
    return cluttered_measurements


def main(use_gnn=False, max_steps=900, num_particles=100000, gnn_warmup=None,
         gnn_load_checkpoint=None, gnn_save_checkpoint=True, gnn_inference_only=False,
         load_measurements=None):
    """
    主测试函数

    参数:
        use_gnn: bool, 是否启用 GNN 模式 (True=BP+GNN, False=纯BP)
        max_steps: int, 运行步数 (默认900)
        num_particles: int, 粒子数量 (默认100000)
        gnn_warmup: int or None, GNN预热步数 (None表示自动计算)
        gnn_load_checkpoint: str or None, GNN权重加载路径 (None表示从头训练)
        gnn_save_checkpoint: bool, 是否保存GNN权重 (默认True)
        gnn_inference_only: bool, 是否仅推理模式 (True=不训练，False=训练)
        load_measurements: str or None, 检测数据文件路径 (None表示生成新数据，否则从文件加载)
    """

    print("=" * 60)
    if use_gnn:
        print("BP-SLAM 测试 (BP + GNN)")
    else:
        print("BP-SLAM 测试 (纯 BP)")
    print(f"Steps: {max_steps}, Particles: {num_particles}")
    print("=" * 60)

    # ---------------------------
    # 1. 通用参数及数据加载
    # ---------------------------
    parameters = {}
    parameters['known_track'] = 0  # 是否已知轨迹（0表示未知轨迹）

    # 加载场景数据，包括虚拟锚点 dataVA 和真实轨迹 trueTrajectory
    mat_data = sio.loadmat('scenarioCleanM2_new901.mat')
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
    parameters['maxSteps'] = max_steps  # 使用传入的步数参数
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
    parameters['numParticles'] = num_particles  # 使用传入的粒子数参数
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
    # [关键] GNN 参数配置
    # ---------------------------
    parameters['use_gnn'] = use_gnn  # 是否启用 FGNN 模式

    if use_gnn:
        # GNN 预热步数：优先使用自定义值，否则根据总步数自适应调整
        if gnn_warmup is not None:
            parameters['gnn_warmup_steps'] = gnn_warmup
            warmup_source = "自定义"
        else:
            # 快速测试(≤100步)用20%，中等测试(≤300步)用10%，完整测试用5%
            if max_steps <= 100:
                warmup_ratio = 0.2
            elif max_steps <= 300:
                warmup_ratio = 0.1
            else:
                warmup_ratio = 0.05
            parameters['gnn_warmup_steps'] = int(max_steps * warmup_ratio)
            warmup_source = "自动计算"

        parameters['gnn_hidden_dim'] = 64  # 隐藏层维度
        parameters['gnn_lr'] = 1e-4  # 学习率

        # 权重加载和保存配置
        parameters['gnn_checkpoint_path'] = gnn_load_checkpoint
        parameters['gnn_save_checkpoint'] = gnn_save_checkpoint
        parameters['gnn_checkpoint_save_path'] = 'checkpoints/gnn_model.pth'
        parameters['gnn_inference_only'] = gnn_inference_only

        print(f"\n[GNN 配置] 预热步数: {parameters['gnn_warmup_steps']} ({warmup_source}), "
              f"隐藏维度: {parameters['gnn_hidden_dim']}, "
              f"学习率: {parameters['gnn_lr']}")

        if gnn_load_checkpoint:
            print(f"  - 加载权重: {gnn_load_checkpoint}")
            if gnn_inference_only:
                print(f"  - 运行模式: 纯推理 (不更新权重)")
            else:
                print(f"  - 运行模式: 继续训练")
        else:
            print(f"  - 训练模式: 从头开始")

        if gnn_save_checkpoint and not gnn_inference_only:
            print(f"  - 保存权重: {parameters['gnn_checkpoint_save_path']}")

    # ---------------------------
    # 3. 随机种子设置（保证结果可重复）
    # ---------------------------
    np.random.seed(1)

    # ---------------------------
    # 4. 移动体初始位置均值设定（真实轨迹起点）
    # ---------------------------
    parameters['priorMean'] = np.vstack([true_trajectory[0:2, 0:1], np.zeros((2, 1))])  # 初始位置+速度

    # ---------------------------
    # 5. 获取检测数据（加载或生成）
    # ---------------------------
    if load_measurements is not None:
        # 从文件加载预先生成的检测数据
        cluttered_measurements = load_measurements_from_mat(load_measurements)
    else:
        # 生成新的检测数据
        print("生成理想测量数据...")
        measurements = generate_measurements(true_trajectory, data_va, parameters)

        print("生成带杂波测量数据...")
        cluttered_measurements = generate_cluttered_measurements(measurements, parameters)
    
    # ---------------------------
    # 7. 调用核心BP-SLAM算法进行估计
    # ---------------------------
    mode_str = "BP + GNN" if use_gnn else "纯 BP"
    print(f"\n开始运行BP-SLAM算法 ({mode_str} 模式)...\n")
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

    # 计算误差统计
    errors = np.linalg.norm(
        true_trajectory[0:2, :] - estimated_trajectory[0:2, :], axis=0
    )
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    final_error = errors[-1]

    print(f"最终位置误差: {final_error:.4f} m")
    print(f"平均位置误差: {mean_error:.4f} m")
    print(f"最大位置误差: {max_error:.4f} m")

    # ---------------------------
    # 8. 保存结果
    # ---------------------------
    # 创建results文件夹
    from pathlib import Path
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # 根据模式保存到不同文件
    result_filename = f'results_{"gnn" if use_gnn else "bp"}.npz'
    print(f"\n保存结果到 results/{result_filename}...")
    np.savez(results_dir / result_filename,
             estimated_trajectory=estimated_trajectory,
             true_trajectory=true_trajectory,
             num_estimated_anchors=num_estimated_anchors,
             estimated_anchors=np.array(estimated_anchors, dtype=object),
             posterior_particles_anchors=np.array(posterior_particles_anchors, dtype=object),
             parameters=parameters,
             use_gnn=use_gnn,
             mean_error=mean_error,
             max_error=max_error,
             final_error=final_error,
             allow_pickle=True)

    print(f"完成！结果已保存到 results/{result_filename}")

    # ---------------------------
    # 9. 可视化（可选）
    # ---------------------------
    try:
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

    except Exception as e:
        print(f"\n可视化跳过（模块未找到或出错）: {e}")
        print("结果已保存到文件，可以稍后使用 visualize_results.py 进行可视化")

    print("\n" + "=" * 60)
    print(f"✓ 完整测试完成 ({mode_str} 模式)！")
    print("=" * 60)

    return estimated_trajectory, estimated_anchors, num_estimated_anchors


if __name__ == '__main__':
    import argparse

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='BP-SLAM 测试脚本（支持自定义步数和粒子数）')
    parser.add_argument('--mode', type=str, default='bp', choices=['bp', 'gnn'],
                        help='运行模式: bp (纯BP) 或 gnn (BP+GNN), 默认: bp')
    parser.add_argument('--steps', type=int, default=900,
                        help='运行步数, 默认: 900')
    parser.add_argument('--particles', type=int, default=100000,
                        help='粒子数量, 默认: 100000')
    parser.add_argument('--warmup', type=int, default=None,
                        help='GNN预热步数 (仅在gnn模式下有效), 默认: 自动计算')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help='GNN权重加载路径 (如 checkpoints/gnn_model.pth), 默认: None (从头训练)')
    parser.add_argument('--no-save-checkpoint', action='store_true',
                        help='不保存GNN权重 (默认会保存)')
    parser.add_argument('--inference-only', action='store_true',
                        help='仅推理模式 (加载权重后不训练，只推理)')
    parser.add_argument('--load-measurements', type=str, default=None,
                        help='从MAT文件加载检测数据 (如 measurementbadf.mat), 默认: None (生成新数据)')

    args = parser.parse_args()

    # 运行测试
    use_gnn = (args.mode == 'gnn')
    main(
        use_gnn=use_gnn,
        max_steps=args.steps,
        num_particles=args.particles,
        gnn_warmup=args.warmup,
        gnn_load_checkpoint=args.load_checkpoint,
        gnn_save_checkpoint=not args.no_save_checkpoint,
        gnn_inference_only=args.inference_only,
        load_measurements=args.load_measurements
    )