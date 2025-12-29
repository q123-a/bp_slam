"""
BP-SLAM 主测试脚本（完整版）
Main test script for BP-SLAM algorithm

支持切换 BP 和 BP+GNN 模式
Converted from MATLAB testbed.m
"""

import sys
import os
from datetime import datetime
import numpy as np
import scipy.io as sio
from scipy.optimize import linear_sum_assignment
from bp_slam.utils.measurements import generate_measurements, generate_cluttered_measurements
from bp_slam.core.slam import bp_based_mint_slam


class Logger:
    """
    日志类：同时输出到终端和文件
    """
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 立即写入文件

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def infer_labels_with_hungarian(measurements_cell, data_va, true_trajectory, num_steps, num_sensors):
    """
    使用匈牙利算法推断测量数据的真实标签

    核心思路：
    1. 对于每个时间步和传感器，计算测量距离和真实距离的代价矩阵
    2. 使用匈牙利算法找到最优匹配
    3. 根据匹配结果推断 true_id
    4. 未匹配的测量标记为杂波

    参数:
        measurements_cell: (num_steps, num_sensors) 测量数据列表
        data_va: 虚拟锚点数据
        true_trajectory: 真实轨迹 (2, num_steps)
        num_steps: 时间步数
        num_sensors: 传感器数量

    返回:
        labels: (num_steps, num_sensors) 标签列表
                每个元素是字典 {'true_id': array, 'is_clutter': array}
    """
    labels = [[None for _ in range(num_sensors)] for _ in range(num_steps)]

    # 统计信息
    total_measurements = 0
    total_matched = 0
    total_unmatched = 0

    for step in range(num_steps):
        for sensor in range(num_sensors):
            meas = measurements_cell[step][sensor]

            if meas is None or meas.size == 0:
                # 没有测量数据
                labels[step][sensor] = {
                    'true_id': np.array([], dtype=int),
                    'is_clutter': np.array([], dtype=bool)
                }
                continue

            M = meas.shape[1]  # 测量数量
            total_measurements += M

            # 获取锚点位置
            anchor_positions = data_va[sensor]['positions']  # (2, K)
            K = anchor_positions.shape[1]  # 锚点数量

            # 获取当前时刻移动体的真实位置
            agent_pos = true_trajectory[:2, step]  # (2,)

            # 计算真实距离（从移动体到每个锚点）
            true_distances = np.zeros(K)
            for k in range(K):
                dx = anchor_positions[0, k] - agent_pos[0]
                dy = anchor_positions[1, k] - agent_pos[1]
                true_distances[k] = np.sqrt(dx**2 + dy**2)

            # 构建代价矩阵 (M, K)
            # 代价 = |测量距离 - 真实距离|
            cost_matrix = np.zeros((M, K))
            for m in range(M):
                meas_dist = meas[0, m]
                for k in range(K):
                    cost_matrix[m, k] = abs(meas_dist - true_distances[k])

            # 使用匈牙利算法求解最优匹配
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # 初始化标签
            true_ids = np.full(M, -1, dtype=int)  # -1 表示杂波
            is_clutter = np.ones(M, dtype=bool)   # 默认全是杂波

            # 根据匹配结果更新标签
            # 只有当匹配代价小于阈值时，才认为是真实匹配
            match_threshold = 2.0  # 2米阈值
            for i, (m, k) in enumerate(zip(row_ind, col_ind)):
                if cost_matrix[m, k] < match_threshold:
                    true_ids[m] = k
                    is_clutter[m] = False
                    total_matched += 1
                else:
                    total_unmatched += 1

            # 保存标签
            labels[step][sensor] = {
                'true_id': true_ids,
                'is_clutter': is_clutter
            }

    # 打印统计信息
    print(f"  - 总测量数: {total_measurements}")
    print(f"  - 成功匹配: {total_matched} ({total_matched/max(total_measurements,1)*100:.1f}%)")
    print(f"  - 未匹配(杂波): {total_unmatched} ({total_unmatched/max(total_measurements,1)*100:.1f}%)")

    return labels


def load_measurements_from_mat(mat_file='measurementbadf.mat', add_synthetic_clutter=False,
                               parameters=None, mismatch_mode=False, return_labels=False,
                               data_va=None, true_trajectory=None):
    """
    从 MAT 文件加载预先生成的检测数据，并应用自适应方差计算

    参数:
        mat_file: MAT 文件路径，默认为 'measurementbadf.mat'
        add_synthetic_clutter: bool, 是否在加载的数据上添加合成杂波 (默认False)
        parameters: 参数字典，当 add_synthetic_clutter=True 时需要提供
        mismatch_mode: bool, 参数失配模式（实际杂波与BP假设不同）
        return_labels: bool, 是否使用匈牙利算法推断标签（用于监督学习）
        data_va: 虚拟锚点数据，当 return_labels=True 时需要提供
        true_trajectory: 真实轨迹，当 return_labels=True 时需要提供

    返回:
        cluttered_measurements: 检测数据，shape (num_steps, num_sensors) 的列表
                               每个元素是 (3, num_detections) 的数组（距离+方差+幅度）
        labels (可选): 如果 return_labels=True，返回标签字典
                      包含 'true_id' 和 'is_clutter'
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
                cluttered_measurements[step][sensor] = np.zeros((3, 0))
            else:
                # mvalse_data 形状为 (3, num_detections)
                # 第0行: 时延 (s) - 需要转换为距离
                # 第1行: 噪声功率 (power) - 暂不使用，使用固定方差
                # 第2行: 信号幅度 (amplitude) - 用于GNN特征
                K_est = mvalse_data.shape[1]
                tracker_input = np.zeros((3, K_est))

                # 1. 距离 (m) = 时延 (s) × 光速 (m/s)
                tracker_input[0, :] = mvalse_data[0, :] * SPEED_OF_LIGHT

                # 2. 使用固定方差 0.0025 m^2 (标准差 0.05 m)
                tracker_input[1, :] = variance_floor

                # 3. 信号幅度 (保留原始值)
                tracker_input[2, :] = mvalse_data[2, :]

                cluttered_measurements[step][sensor] = tracker_input

    print(f"✓ 成功加载检测数据: {num_steps} 步, {num_sensors} 个传感器")

    # [新增] 使用匈牙利算法推断标签
    labels = None
    if return_labels:
        if data_va is None or true_trajectory is None:
            raise ValueError("推断标签时必须提供 data_va 和 true_trajectory 参数")

        print(f"\n使用匈牙利算法推断测量标签...")
        labels = infer_labels_with_hungarian(
            cluttered_measurements, data_va, true_trajectory, num_steps, num_sensors
        )
        print(f"✓ 标签推断完成")

    # [新增] 添加合成杂波
    if add_synthetic_clutter:
        if parameters is None:
            raise ValueError("添加合成杂波时必须提供 parameters 参数")

        print(f"\n添加合成杂波...")
        print(f"  - 平均杂波数: {parameters['meanNumberOfClutter']}")
        print(f"  - 检测概率: {parameters['detectionProbability']}")
        print(f"  - 区域大小: {parameters['regionOfInterestSize']} m")

        if return_labels:
            # 添加杂波并更新标签
            cluttered_measurements, labels = add_synthetic_clutter_to_measurements(
                cluttered_measurements, parameters, mismatch_mode=mismatch_mode,
                return_labels=True, existing_labels=labels
            )
        else:
            cluttered_measurements = add_synthetic_clutter_to_measurements(
                cluttered_measurements, parameters, mismatch_mode=mismatch_mode
            )
        print(f"✓ 合成杂波添加完成")

    if return_labels:
        return cluttered_measurements, labels
    else:
        return cluttered_measurements


def add_synthetic_clutter_to_measurements(measurements_cell, parameters, mismatch_mode=False,
                                         return_labels=False, existing_labels=None):
    """
    在已加载的测量数据上添加合成杂波和漏检

    参数:
        measurements_cell: 原始测量数据，shape (num_steps, num_sensors)的列表
                          每个元素是 (3, num_detections) 的数组（距离+方差+幅度）
        parameters: 参数字典，包括测量方差、检测概率、杂波均值、区域大小等
        mismatch_mode: bool, 是否使用参数失配模式（实际杂波参数与BP假设不同）
        return_labels: bool, 是否返回标签（用于监督学习）
        existing_labels: 已有的标签（如果有的话），会在添加杂波时更新

    返回:
        cluttered_measurements: 添加杂波后的测量数据，shape同输入
        labels (可选): 如果 return_labels=True，返回标签字典
    """
    # 读取参数
    measurement_variance_range = parameters['measurementVariance']
    max_range = parameters['regionOfInterestSize']

    # [新增] 参数失配模式：实际参数与BP假设不同
    if mismatch_mode:
        # 实际杂波率是BP假设的3倍
        actual_mean_clutter = parameters['meanNumberOfClutter'] * 3
        # 实际检测概率比BP假设低10%
        actual_detection_prob = max(0.5, parameters['detectionProbability'] - 0.1)
        print(f"\n⚠ 参数失配模式:")
        print(f"  - BP假设杂波数: {parameters['meanNumberOfClutter']}, 实际: {actual_mean_clutter}")
        print(f"  - BP假设检测率: {parameters['detectionProbability']}, 实际: {actual_detection_prob}")
    else:
        actual_mean_clutter = parameters['meanNumberOfClutter']
        actual_detection_prob = parameters['detectionProbability']

    detection_probability = actual_detection_prob
    mean_number_of_clutter = actual_mean_clutter

    num_steps = len(measurements_cell)
    num_sensors = len(measurements_cell[0])

    # 初始化输出
    cluttered_measurements = [[None for _ in range(num_sensors)] for _ in range(num_steps)]

    # 如果需要返回标签，初始化标签存储
    if return_labels:
        labels = [[None for _ in range(num_sensors)] for _ in range(num_steps)]

    # 统计信息
    total_original = 0
    total_detected = 0
    total_clutter = 0

    # 遍历每个传感器和时间步
    for sensor in range(num_sensors):
        for step in range(num_steps):
            original_measurements = measurements_cell[step][sensor]

            if original_measurements is None or original_measurements.size == 0:
                num_detections = 0
                detected_measurements = np.zeros((3, 0))
                detected_ids = np.array([], dtype=int)
            else:
                num_detections = original_measurements.shape[1]
                total_original += num_detections

                # 按检测概率随机决定哪些测量被检测到（漏检处理）
                detection_indicator = (np.random.rand(num_detections) < detection_probability)

                # 提取被检测到的测量（保留3行：距离、方差、幅度）
                detected_measurements = original_measurements[:, detection_indicator]
                total_detected += detected_measurements.shape[1]

                # 记录被检测到的测量ID（如果有已有标签）
                if return_labels and existing_labels is not None:
                    # 使用已有标签中的 true_id
                    original_true_ids = existing_labels[step][sensor]['true_id']
                    detected_ids = original_true_ids[detection_indicator]
                else:
                    # 假设原始测量按顺序对应锚点 ID
                    detected_ids = np.where(detection_indicator)[0]

            # 生成误报（杂波）数量，符合泊松分布
            num_false_alarms = np.random.poisson(mean_number_of_clutter)
            total_clutter += num_false_alarms

            # 生成误报测量（3行：距离、方差、幅度）
            false_alarms = np.zeros((3, num_false_alarms))
            if num_false_alarms > 0:
                # 误报距离均匀分布在0到maxRange
                false_alarms[0, :] = max_range * np.random.rand(num_false_alarms)
                # 误报测量方差为测距方差
                false_alarms[1, :] = measurement_variance_range
                # [阶段二改进] 物理相关性杂波 (Physics-Based Clutter)
                # 核心思想：真实杂波不是随机的，而是遵循物理规律
                # 杂波通常是反射/多径信号，遵循：RSS_clutter ≈ RSS_theory(d) - Δ
                # 其中 Δ 是反射损耗（3-20dB）
                #
                # 这样生成的杂波：
                # 1. 符合距离衰减规律（看起来像真信号）
                # 2. 只是稍微弱一点（反射损耗）
                # 3. 是GNN最难分辨的对手！

                # [修改] 杂波物理参数：让杂波显著弱于真实信号
                # 问题：原来的 P_tx=15.41 太强，导致杂波和真信号强度相近
                # 解决：降低虚拟发射功率 + 增加反射损耗

                # 1. 降低杂波的虚拟发射功率
                # 真实信号 P_tx=15.41，杂波使用更低的功率模拟非视距传播
                P_tx_clutter = -10.0  # 从 15.41 降低到 -10
                n = 2.0

                # 2. 计算杂波距离对应的理论RSS
                clutter_dists = false_alarms[0, :]
                safe_dists = np.maximum(clutter_dists, 0.1)  # 防止log(0)
                rss_theory = P_tx_clutter - 10 * n * np.log10(safe_dists)

                # 3. 增加反射损耗 (Reflection Loss)
                # 从 3-20dB 增加到 15-35dB，模拟强烈的非视距衰减
                # 15dB: 单次墙面反射
                # 35dB: 多次反射 + 穿墙衰减
                reflection_loss = np.random.uniform(15.0, 35.0, num_false_alarms)

                # 4. 合成杂波RSS = 理论值 - 反射损耗 + 小噪声
                clutter_rss_dbm = rss_theory - reflection_loss + np.random.normal(0, 2.0, num_false_alarms)

                # 5. 强行截断：确保杂波不超过 -30 dBm（远低于真实信号的 -13 dBm）
                clutter_rss_dbm = np.minimum(clutter_rss_dbm, -30.0)

                false_alarms[2, :] = clutter_rss_dbm

                # 结果：
                # - 杂波 RSS 范围：约 -60 ~ -30 dBm
                # - 真实信号 RSS：约 -13 ~ 1 dBm
                # - 差距显著，GNN 可以学习区分

            # 生成标签（在打乱之前）
            if return_labels:
                # 杂波的 true_id = -1, is_clutter = True
                clutter_true_ids = np.full(num_false_alarms, -1, dtype=int)
                clutter_is_clutter = np.ones(num_false_alarms, dtype=bool)

                # 真实测量的 true_id = 锚点ID, is_clutter = False
                detected_true_ids = detected_ids
                detected_is_clutter = np.zeros(len(detected_ids), dtype=bool)

                # 拼接标签
                true_ids = np.concatenate([clutter_true_ids, detected_true_ids])
                is_clutter = np.concatenate([clutter_is_clutter, detected_is_clutter])

            # 将误报和真实检测测量拼接
            if detected_measurements.size > 0:
                cluttered_measurement = np.hstack([false_alarms, detected_measurements])
            else:
                cluttered_measurement = false_alarms

            # 随机打乱测量顺序，模拟实际测量的无序性
            if cluttered_measurement.shape[1] > 0:
                perm = np.random.permutation(cluttered_measurement.shape[1])
                cluttered_measurement = cluttered_measurement[:, perm]

                # 同时打乱标签
                if return_labels:
                    true_ids = true_ids[perm]
                    is_clutter = is_clutter[perm]

            # 保存当前时间步传感器的测量
            cluttered_measurements[step][sensor] = cluttered_measurement

            # 保存标签
            if return_labels:
                labels[step][sensor] = {
                    'true_id': true_ids,
                    'is_clutter': is_clutter
                }

    # 打印统计信息
    print(f"\n杂波添加统计:")
    print(f"  - 原始测量总数: {total_original}")
    print(f"  - 检测到的测量: {total_detected} ({total_detected/max(total_original,1)*100:.1f}%)")
    print(f"  - 漏检数量: {total_original - total_detected}")
    print(f"  - 添加的杂波: {total_clutter}")
    print(f"  - 最终测量总数: {total_detected + total_clutter}")

    if return_labels:
        return cluttered_measurements, labels
    else:
        return cluttered_measurements


def main(use_gnn=False, max_steps=900, num_particles=100000, gnn_warmup=None,
         gnn_load_checkpoint=None, gnn_save_checkpoint=True, gnn_inference_only=False,
         load_measurements=None, add_synthetic_clutter=False, mismatch_mode=False,
         use_sparse_graph=False, use_sparse_graph_v2=False,
         distance_threshold=None, beta_threshold=None):
    """
    主测试函数

    参数:
        use_gnn: bool, 是否启用 GNN 模式 (True=BP+GNN, False=纯BP)
        max_steps: int, 运行步数 (默认900)
        num_particles: int, 粒子数量 (默认100000)
        gnn_warmup: int or None, GNN预热步数 (None表示自动计算)
        use_sparse_graph: bool, 是否使用稀疏图 GAT
        use_sparse_graph_v2: bool, 是否使用稀疏图 GAT V2 (3维边特征，无Dustbin)
        distance_threshold: float or None, 稀疏图距离阈值
        beta_threshold: float or None, 稀疏图 Beta 阈值
        gnn_load_checkpoint: str or None, GNN权重加载路径 (None表示从头训练)
        gnn_save_checkpoint: bool, 是否保存GNN权重 (默认True)
        gnn_inference_only: bool, 是否仅推理模式 (True=不训练，False=训练)
        load_measurements: str or None, 检测数据文件路径 (None表示生成新数据，否则从文件加载)
        add_synthetic_clutter: bool, 是否在加载的数据上添加合成杂波 (默认False)
        mismatch_mode: bool, 参数失配模式 (True=实际参数与BP假设不同，测试鲁棒性)
    """

    # ---------------------------
    # 初始化日志系统
    # ---------------------------
    # 创建日志目录
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # 生成日志文件名：logs/testbed_gnn_20231211_153045.log
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_str = 'gnn' if use_gnn else 'bp'
    mismatch_str = '_mismatch' if mismatch_mode else ''
    log_file = os.path.join(log_dir, f'testbed_{mode_str}{mismatch_str}_{timestamp}.log')

    # 重定向标准输出到日志文件
    logger = Logger(log_file)
    sys.stdout = logger
    sys.stderr = logger

    print(f"日志文件: {log_file}")
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
    mat_data = sio.loadmat('scenarioCleanM2_new_1500.mat')
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

    # [新增] 失配模式下，让BP使用更保守的参数估计
    # 这样可以减少错误关联，改善锚点估计质量
    if mismatch_mode and not use_gnn:
        # BP在失配模式下，假设更高的杂波率（更保守）
        # 这样BP会更谨慎地关联测量，减少误判
        parameters['meanNumberOfClutter'] = 2  # 假设杂波更多
        parameters['detectionProbability'] = 0.90  # 假设检测率更低
        print(f"\n⚠ BP保守模式（失配环境）:")
        print(f"  - BP使用保守参数: 杂波数2, 检测率0.90")
        print(f"  - 实际环境: 杂波数3, 检测率0.85")

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
        parameters['gnn_lr'] = 1e-4  # [修正] 学习率降低到1e-4，防止过拟合

        # [改进版] 新增参数 - 针对连续凸起问题优化
        parameters['gnn_use_ema'] = True  # 使用指数移动平均
        parameters['gnn_ema_decay'] = 0.9995  # EMA衰减率 (从0.999增强到0.9995，更强平滑)
        parameters['gnn_use_lr_scheduler'] = True  # 使用学习率调度器
        parameters['gnn_pseudo_label_mode'] = 'or'  # 伪标签模式: 'and', 'or', 'adaptive'
        parameters['gnn_confidence_weighting'] = True  # 置信度加权损失

        # [关键] GRU控制参数 - 开启节点级GRU时序记忆
        #
        # [改进] 开启跨帧GRU记忆，让模型拥有"记忆"能力
        # 核心优势：
        # 1. 防止过度自信：即使当前帧RSS误差小，但如果前几帧没见过，GRU会保留意见
        # 2. 时序平滑：消除OSPA尖峰，获得更平滑的曲线
        # 3. 节点级记忆：每个测量-锚点对独立记忆（而非全局共享）
        # 4. 梯度截断：trainer内部使用.detach()防止错误传播
        parameters['gnn_use_temporal_gru'] = True  # [改进] 开启跨帧节点级GRU记忆
        parameters['gnn_use_layer_gru'] = False  # 层内GRU更新 (关闭以简化模型)

        # [新增] 双头架构参数
        parameters['gnn_use_dual_head'] = True  # 使用双头架构（质量头+关联头）
        parameters['gnn_quality_threshold'] = 0.25  # [微调] 质量判断阈值降低到0.25，避免过滤真实锚点
        parameters['gnn_quality_weight'] = 1.5  # [修正] 质量损失权重降低到1.5，防止过拟合
        parameters['gnn_assoc_weight'] = 1.0  # [综合方案] 关联损失权重保持1.0
        parameters['gnn_assoc_threshold'] = 3.0  # 关联熔断阈值

        # [新增] 新锚点候选缓冲区参数（防止瞬时噪声被误判为新锚点）
        parameters['gnn_new_anchor_threshold'] = 0.5  # [微调] messages_new 阈值降低到0.5，更容易添加新锚点
        parameters['gnn_new_anchor_min_frames'] = 2  # [微调] 最少连续检测帧数降低到2
        parameters['gnn_new_anchor_max_gap'] = 3  # [微调] 允许的最大间隔帧数增加到3

        # [新增] 匈牙利算法参数 - 针对失配模式优化
        if mismatch_mode:
            parameters['gnn_rejection_threshold'] = 4.0  # [修正] 放宽熔断阈值到4.0，适应预测不准确
            parameters['gnn_positive_weight'] = 10.0  # 更高的正样本权重（从5.0提升到10.0）
            # [关键修正] 双头架构下，质量头不依赖BP预测，可以快速启动
            # 只需要少量预热让模型稳定即可
            if parameters.get('gnn_use_dual_head', False):
                parameters['gnn_warmup_steps'] = int(max_steps * 0.05)  # 双头：5%预热（45步）
            else:
                parameters['gnn_warmup_steps'] = int(max_steps * 0.10)  # 单头：10%预热（90步）
            print(f"  - ⚠ 失配模式优化: 阈值4.0, 权重10.0, 预热{parameters['gnn_warmup_steps']}步")
        else:
            parameters['gnn_rejection_threshold'] = 3.0  # 标准熔断阈值
            parameters['gnn_positive_weight'] = 5.0  # 标准正样本权重

        # 权重加载和保存配置
        parameters['gnn_checkpoint_path'] = gnn_load_checkpoint
        parameters['gnn_save_checkpoint'] = gnn_save_checkpoint
        parameters['gnn_checkpoint_save_path'] = 'checkpoints/gnn_model.pth'
        parameters['gnn_inference_only'] = gnn_inference_only

        # [新增] 稀疏图 GAT 参数
        parameters['gnn_use_sparse_graph'] = use_sparse_graph
        parameters['gnn_use_sparse_graph_v2'] = use_sparse_graph_v2
        parameters['gnn_distance_threshold'] = distance_threshold
        parameters['gnn_beta_threshold'] = beta_threshold

        if use_sparse_graph_v2:
            print(f"\n[GNN 配置 - 稀疏图 GAT V2]")
            print(f"  - 边特征: 3维 [Δx, Δy, Σ]")
            print(f"  - 节点特征: 3维语义特征")
            print(f"  - 无Dustbin节点")
        elif use_sparse_graph:
            print(f"\n[GNN 配置 - 稀疏图 GAT V1]")
            print(f"  - 边特征: 5维")
            print(f"  - 有Dustbin节点")
        else:
            print(f"\n[GNN 配置 - 双头架构版本]")
        if use_sparse_graph:
            print(f"  - 架构模式: 稀疏图 GAT (质量头+关联头)")
            print(f"  - 稀疏图过滤: 距离阈值={distance_threshold}, Beta阈值={beta_threshold}")
        else:
            print(f"  - 架构模式: {'双头 (质量头+关联头)' if parameters['gnn_use_dual_head'] else '单头'}")
        print(f"  - 预热步数: {parameters['gnn_warmup_steps']} ({warmup_source})")
        print(f"  - 隐藏维度: {parameters['gnn_hidden_dim']}")
        print(f"  - 学习率: {parameters['gnn_lr']}")
        print(f"  - EMA: {parameters['gnn_use_ema']} (decay={parameters['gnn_ema_decay']})")
        print(f"  - 学习率调度: {parameters['gnn_use_lr_scheduler']}")
        if parameters['gnn_use_dual_head']:
            print(f"  - 质量阈值: {parameters['gnn_quality_threshold']}")
            print(f"  - 损失权重: 质量={parameters['gnn_quality_weight']}, 关联={parameters['gnn_assoc_weight']}")
            print(f"  - 关联熔断阈值: {parameters['gnn_assoc_threshold']}")
        else:
            print(f"  - 伪标签模式: {parameters['gnn_pseudo_label_mode']}")
            print(f"  - 置信度加权: {parameters['gnn_confidence_weighting']}")
        print(f"  - 跨帧GRU: {parameters['gnn_use_temporal_gru']} {'✓ 节点级时序记忆' if parameters['gnn_use_temporal_gru'] else '(关闭)'}")
        print(f"  - 层内GRU: {parameters['gnn_use_layer_gru']} (关闭以简化模型)")
        print(f"  - 新锚点缓冲区: 阈值={parameters['gnn_new_anchor_threshold']}, 最少帧数={parameters['gnn_new_anchor_min_frames']}, 最大间隔={parameters['gnn_new_anchor_max_gap']}")

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
        if use_gnn:
            # GNN 模式：使用匈牙利算法推断标签
            cluttered_measurements, ground_truth_labels = load_measurements_from_mat(
                load_measurements,
                add_synthetic_clutter=add_synthetic_clutter,
                parameters=parameters,
                mismatch_mode=mismatch_mode,
                return_labels=True,
                data_va=data_va,
                true_trajectory=true_trajectory
            )
            print("✓ 已使用匈牙利算法推断监督学习标签")
        else:
            # 纯 BP 模式：不需要标签
            cluttered_measurements = load_measurements_from_mat(
                load_measurements,
                add_synthetic_clutter=add_synthetic_clutter,
                parameters=parameters,
                mismatch_mode=mismatch_mode
            )
            ground_truth_labels = None
    else:
        # 生成新的检测数据
        print("生成理想测量数据...")
        measurements = generate_measurements(true_trajectory, data_va, parameters)

        print("生成带杂波测量数据...")
        # 如果使用GNN，生成监督学习标签
        if use_gnn:
            cluttered_measurements, ground_truth_labels = generate_cluttered_measurements(
                measurements, parameters, return_labels=True
            )
            print("✓ 已生成监督学习标签 (true_id, is_clutter)")
        else:
            cluttered_measurements = generate_cluttered_measurements(measurements, parameters)
            ground_truth_labels = None

    # ---------------------------
    # 7. 调用核心BP-SLAM算法进行估计
    # ---------------------------
    mode_str = "BP + GNN" if use_gnn else "纯 BP"
    print(f"\n开始运行BP-SLAM算法 ({mode_str} 模式)...\n")
    print("=" * 60)
    (estimated_trajectory, estimated_anchors,
     posterior_particles_anchors, num_estimated_anchors) = bp_based_mint_slam(
        data_va, cluttered_measurements, parameters, true_trajectory,
        ground_truth_labels=ground_truth_labels  # 传递标签
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
    # 7.5. 计算OSPA误差（每个传感器）
    # ---------------------------
    print("\n计算OSPA地图误差...")
    from bp_slam.visualization.plotting import ospa_dist

    num_sensors = len(data_va)
    num_steps = true_trajectory.shape[1]
    ospa_errors = np.zeros((num_sensors, num_steps))

    for sensor in range(num_sensors):
        true_anchor_positions = data_va[sensor]['positions']

        for step in range(num_steps):
            # 获取估计的锚点位置
            if estimated_anchors[sensor][step] is not None and len(estimated_anchors[sensor][step]) > 0:
                estimated_positions = []
                for anchor in estimated_anchors[sensor][step]:
                    if anchor is not None and 'x' in anchor:
                        if anchor['posteriorExistence'] > parameters['detectionThreshold']:
                            estimated_positions.append(anchor['x'])

                if len(estimated_positions) > 0:
                    estimated_anchor_positions = np.array(estimated_positions).T
                else:
                    estimated_anchor_positions = None
            else:
                estimated_anchor_positions = None

            # 计算OSPA距离
            ospa, _, _ = ospa_dist(true_anchor_positions, estimated_anchor_positions, 10, 1)
            ospa_errors[sensor, step] = ospa

    # 打印OSPA统计
    for sensor in range(num_sensors):
        print(f"传感器 {sensor + 1} OSPA误差: 平均={np.mean(ospa_errors[sensor, :]):.4f} m, "
              f"最大={np.max(ospa_errors[sensor, :]):.4f} m")

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
             ospa_errors=ospa_errors,
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
            show=True,
            mode='gnn' if use_gnn else 'bp'
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

    # 关闭日志文件
    print(f"\n日志已保存到: {log_file}")
    logger.close()
    sys.stdout = logger.terminal
    sys.stderr = logger.terminal

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
    parser.add_argument('--add-clutter', action='store_true',
                        help='在加载的数据上添加合成杂波 (仅在使用 --load-measurements 时有效)')
    parser.add_argument('--mismatch-mode', action='store_true',
                        help='参数失配模式：实际杂波参数与BP假设不同，测试BP鲁棒性 (仅在使用 --add-clutter 时有效)')
    parser.add_argument('--use-sparse-graph', action='store_true',
                        help='使用稀疏图 GAT V1（需要 torch_geometric）- 5维边特征，有Dustbin')
    parser.add_argument('--use-sparse-graph-v2', action='store_true',
                        help='使用稀疏图 GAT V2（需要 torch_geometric）- 3维边特征 [Δx, Δy, Σ]，无Dustbin')
    parser.add_argument('--distance-threshold', type=float, default=None,
                        help='稀疏图距离阈值（米），None=不过滤')
    parser.add_argument('--beta-threshold', type=float, default=None,
                        help='稀疏图 Beta 阈值，None=不过滤')

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
        load_measurements=args.load_measurements,
        add_synthetic_clutter=args.add_clutter,
        mismatch_mode=args.mismatch_mode,
        use_sparse_graph=args.use_sparse_graph or args.use_sparse_graph_v2,  # V1 或 V2 都算稀疏图
        use_sparse_graph_v2=args.use_sparse_graph_v2,  # 新增：是否使用 V2
        distance_threshold=args.distance_threshold,
        beta_threshold=args.beta_threshold
    )