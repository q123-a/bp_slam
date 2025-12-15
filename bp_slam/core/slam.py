"""
基于信念传播的多路径SLAM算法核心函数 (集成 Hybrid FGNN)
BP-based Multipath-assisted SLAM core algorithm

Author: Florian Meyer, Erik Leitinger, 20/05/17
Converted to Python: 2025
"""

import numpy as np
import time
import copy
from ..utils.sampling import draw_samples_uniformly_circ, resample_systematic
from ..utils.motion_model import perform_prediction
from ..utils.distance import calc_distance
from .anchors import (init_anchors, predict_anchors, predict_measurements,
                     generate_new_anchors, delete_unreliable_va)
from .association import calculate_association_probabilities_ga

# FGNN 相关导入
try:
    import torch
    from .gnn_trainer_improved import GNNTrainerImproved, JointDualHeadTrainer
    FGNN_AVAILABLE = True
except ImportError:
    FGNN_AVAILABLE = False
    JointDualHeadTrainer = None
    print("Warning: PyTorch not available. FGNN mode disabled.")


def bp_based_mint_slam(data_va, cluttered_measurements, parameters, true_trajectory):
    """
    基于信念传播的多路径SLAM算法核心函数

    参数:
        data_va: 虚拟锚点数据列表
        cluttered_measurements: 带误报的测量数据（距离+方差），列表[num_steps][num_sensors]
        parameters: 算法参数字典
        true_trajectory: 真实轨迹，用于误差计算（已知轨迹模式），shape (n_dims, num_steps)

    返回:
        estimated_trajectory: 估计的移动体状态轨迹（位置+速度），shape (4, num_steps)
        estimated_anchors: 估计的锚点位置和存在概率
        posterior_particles_anchors_storage: 存储部分时刻锚点粒子用于分析
        num_estimated_anchors: 每时刻估计的锚点数量，shape (num_sensors, num_steps)
    """
    # 获取测量时间步数和传感器数量
    num_steps = len(cluttered_measurements)
    num_sensors = len(cluttered_measurements[0])

    # 限制最大时间步数
    num_steps = min(num_steps, parameters['maxSteps'])

    # 读取参数
    num_particles = parameters['numParticles']
    detection_probability = parameters['detectionProbability']
    prior_mean = parameters['priorMean']
    survival_probability = parameters['survivalProbability']
    undetected_anchors_intensity = parameters['undetectedAnchorsIntensity'] * np.ones(num_sensors)
    birth_intensity = parameters['birthIntensity']
    clutter_intensity = parameters['clutterIntensity']
    unreliability_threshold = parameters['unreliabilityThreshold']
    exec_time_per_step = np.zeros(num_steps)
    known_track = parameters['known_track']

    # [新增] 初始化 GNN 训练器
    gnn_trainer = None
    use_gnn = parameters.get('use_gnn', False)
    warmup_steps = parameters.get('gnn_warmup_steps', 50)
    use_dual_head = parameters.get('gnn_use_dual_head', False)  # 是否使用双头架构

    # [新增] GRU隐藏状态管理（每个传感器独立维护）
    # 格式: gnn_hidden_states[sensor] = (M*K, hidden_dim) 的tensor
    gnn_hidden_states = {}

    if use_gnn and FGNN_AVAILABLE:
        gnn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gnn_hidden_dim = parameters.get('gnn_hidden_dim', 64)
        gnn_lr = parameters.get('gnn_lr', 1e-4)
        gnn_checkpoint_path = parameters.get('gnn_checkpoint_path', None)

        # [改进版] 新增参数
        gnn_use_ema = parameters.get('gnn_use_ema', True)
        gnn_ema_decay = parameters.get('gnn_ema_decay', 0.999)
        gnn_use_lr_scheduler = parameters.get('gnn_use_lr_scheduler', True)
        gnn_use_temporal_gru = parameters.get('gnn_use_temporal_gru', False)  # 默认关闭跨帧GRU
        gnn_use_layer_gru = parameters.get('gnn_use_layer_gru', False)  # 默认关闭层内GRU

        if use_dual_head:
            # ===== 双头架构 =====
            quality_threshold = parameters.get('gnn_quality_threshold', 0.5)
            assoc_threshold = parameters.get('gnn_assoc_threshold', 3.0)
            quality_weight = parameters.get('gnn_quality_weight', 1.0)
            assoc_weight = parameters.get('gnn_assoc_weight', 2.0)

            gnn_trainer = JointDualHeadTrainer(
                device=gnn_device,
                lr=gnn_lr,
                hidden_dim=gnn_hidden_dim,
                checkpoint_path=gnn_checkpoint_path,
                seed=42,
                use_ema=gnn_use_ema,
                ema_decay=gnn_ema_decay,
                use_lr_scheduler=gnn_use_lr_scheduler,
                use_temporal_gru=gnn_use_temporal_gru,
                use_layer_gru=gnn_use_layer_gru,
                quality_threshold=quality_threshold,
                assoc_threshold=assoc_threshold,
                quality_weight=quality_weight,
                assoc_weight=assoc_weight
            )

            print(f"✓ 双头GNN训练器已初始化 ({gnn_device})")
            print(f"  - 预热步数: {warmup_steps}")
            print(f"  - 质量阈值: {quality_threshold}")
            print(f"  - 关联阈值: {assoc_threshold}")
        else:
            # ===== 单头架构（原有） =====
            gnn_pseudo_label_mode = parameters.get('gnn_pseudo_label_mode', 'and')
            gnn_confidence_weighting = parameters.get('gnn_confidence_weighting', True)
            gnn_rejection_threshold = parameters.get('gnn_rejection_threshold', 3.0)
            gnn_positive_weight = parameters.get('gnn_positive_weight', 5.0)

            gnn_trainer = GNNTrainerImproved(
                device=gnn_device,
                lr=gnn_lr,
                hidden_dim=gnn_hidden_dim,
                checkpoint_path=gnn_checkpoint_path,
                seed=42,
                use_ema=gnn_use_ema,
                ema_decay=gnn_ema_decay,
                use_lr_scheduler=gnn_use_lr_scheduler,
                pseudo_label_mode=gnn_pseudo_label_mode,
                confidence_weighting=gnn_confidence_weighting,
                use_temporal_gru=gnn_use_temporal_gru,
                use_layer_gru=gnn_use_layer_gru,
                rejection_threshold=gnn_rejection_threshold,
                positive_weight=gnn_positive_weight
            )

            print(f"✓ 单头GNN训练器已初始化 ({gnn_device})")
            print(f"  - 预热步数: {warmup_steps}")

        # [关键] 每次开始新序列前，清空 GRU 记忆
        if hasattr(gnn_trainer, 'reset_hidden_state'):
            gnn_trainer.reset_hidden_state()
        if gnn_checkpoint_path:
            print(f"  - Loaded checkpoint: {gnn_checkpoint_path}")

    # 预分配存储空间
    estimated_trajectory = np.zeros((4, num_steps))  # 状态空间4维：x,y,vx,vy
    num_estimated_anchors = np.zeros((num_sensors, num_steps), dtype=int)
    storing_idx = list(range(29, num_steps, 30))  # 每30步存储一次锚点粒子状态（Python从0开始）
    posterior_particles_anchors_storage = [None] * len(storing_idx)

    # [新增] 新锚点候选缓冲区（防止瞬时噪声被误判为新锚点）
    # 结构: candidate_anchors[sensor] = {meas_idx: {'position': [x, y], 'count': N, 'last_step': step}}
    candidate_anchors = [{} for _ in range(num_sensors)]
    candidate_threshold = parameters.get('gnn_new_anchor_threshold', 0.6)  # messages_new 阈值
    candidate_min_frames = parameters.get('gnn_new_anchor_min_frames', 3)  # 最少连续检测帧数
    candidate_max_gap = parameters.get('gnn_new_anchor_max_gap', 2)  # 允许的最大间隔帧数

    # 初始化移动体粒子
    if known_track:
        # 已知轨迹时，所有粒子初始化为真实轨迹状态，速度为0
        posterior_particles_agent = np.tile(
            np.vstack([true_trajectory[:2, 0:1], np.zeros((2, 1))]),
            (1, num_particles)
        )
    else:
        # 未知轨迹时，均匀采样位置粒子，速度粒子随机采样
        posterior_particles_agent = np.zeros((4, num_particles))
        posterior_particles_agent[0:2, :] = draw_samples_uniformly_circ(
            prior_mean[0:2], parameters['UniformRadius_pos'], num_particles
        )
        posterior_particles_agent[2:4, :] = (
            np.tile(prior_mean[2:4].reshape(-1, 1), (1, num_particles)) +
            2 * parameters['UniformRadius_vel'] * np.random.rand(2, num_particles) -
            parameters['UniformRadius_vel']
        )

    # 记录初始状态估计（粒子均值）
    estimated_trajectory[:, 0] = np.mean(posterior_particles_agent, axis=1)

    # 初始化锚点状态（位置粒子和权重）
    estimated_anchors, posterior_particles_anchors = init_anchors(
        parameters, data_va, num_steps, num_sensors
    )
    for sensor in range(num_sensors):
        num_estimated_anchors[sensor, 0] = len(estimated_anchors[sensor][0])

    # 主循环，遍历每个时间步
    for step in range(1, num_steps):  # Python从0开始，所以从1开始
        start_time = time.time()

        # 预测移动体状态
        if known_track:
            # 已知轨迹，粒子直接用真实轨迹
            predicted_particles_agent = np.tile(
                np.vstack([true_trajectory[:2, step:step+1], np.zeros((2, 1))]),
                (1, num_particles)
            )
        else:
            # 未知轨迹时，基于动力学模型预测粒子状态
            predicted_particles_agent = perform_prediction(posterior_particles_agent, parameters)

        # 初始化存储每个粒子每个传感器权重的矩阵
        weights_sensors = np.full((num_particles, num_sensors), np.nan)

        # 对每个传感器进行锚点估计和数据关联更新
        for sensor in range(num_sensors):
            # 继承上一时刻估计的锚点状态（深拷贝以避免修改历史数据）
            estimated_anchors[sensor][step] = copy.deepcopy(estimated_anchors[sensor][step - 1])
            measurements = cluttered_measurements[step][sensor]  # 当前时刻传感器测量

            if measurements is None or measurements.size == 0:
                num_measurements = 0
            else:
                num_measurements = measurements.shape[1]

            # 预测未检测锚点强度（存活概率衰减+新锚点出生强度）
            undetected_anchors_intensity[sensor] = (
                undetected_anchors_intensity[sensor] * survival_probability + birth_intensity
            )

            # 预测"遗留"锚点粒子状态及权重
            predicted_particles_anchors, weights_anchor = predict_anchors(
                posterior_particles_anchors[sensor], parameters
            )

            # 针对每个测量生成新锚点粒子（新特征）
            new_particles_anchors, new_input_bp = generate_new_anchors(
                measurements, undetected_anchors_intensity[sensor],
                predicted_particles_agent, parameters
            )

            # 预测由锚点到移动体的测量值及其不确定度
            predicted_measurements, predicted_uncertainties, predicted_range = predict_measurements(
                predicted_particles_agent, predicted_particles_anchors, weights_anchor
            )

            # ==================================================================
            # [关键修改] 数据关联：混合特征构建 + 自监督训练
            # ==================================================================

            # 计算锚点存在概率
            num_anchors = predicted_particles_anchors.shape[2]
            existence_probs = np.zeros(num_anchors)
            for a in range(num_anchors):
                existence_probs[a] = np.sum(weights_anchor[:, a])

            # --- A. [策略一] 物理海选：用内在一致性过滤明显杂波 ---
            # 路径损耗参数
            P_tx = 15.41
            n = 2.0

            # 计算所有测量的内在一致性误差
            intrinsic_errors = np.zeros(num_measurements)
            valid_mask = np.ones(num_measurements, dtype=bool)  # 标记哪些测量通过物理检查

            for m in range(num_measurements):
                # 提取测量幅度
                if measurements.shape[0] >= 3:
                    rss_meas = measurements[2, m]
                else:
                    rss_meas = 0.0

                # [内在一致性] 使用测量距离计算理论RSS
                safe_meas_dist = max(measurements[0, m], 0.1)
                rss_theory = P_tx - 10 * n * np.log10(safe_meas_dist)

                # 内在一致性误差（dB）
                intrinsic_error_db = abs(rss_meas - rss_theory)
                intrinsic_errors[m] = intrinsic_error_db / 10.0  # 归一化

                # [修复] 移除硬过滤！让GNN看到真实的杂波数据
                # 只过滤极其离谱的测量（> 50 dB，物理上不可能）
                if intrinsic_error_db > 50.0:
                    valid_mask[m] = False
                # 原来的15dB阈值会导致"幸存者偏差"：
                # GNN只能看到"好点"，无法学习识别真正的杂波

            # 统计过滤结果
            num_filtered = np.sum(~valid_mask)
            if num_filtered > 0 and step % 50 == 0:
                print(f"  [极端值过滤] 过滤掉 {num_filtered}/{num_measurements} 个极端异常测量 (>50dB)")

            # 保留几乎所有测量（包括杂波），让GNN学习识别
            filtered_measurements = measurements[:, valid_mask]
            filtered_intrinsic_errors = intrinsic_errors[valid_mask]
            num_filtered_measurements = filtered_measurements.shape[1]

            # --- B. 计算物理模型概率 (Beta) - 只对过滤后的测量计算 ---
            beta_matrix_filtered = np.zeros((num_filtered_measurements, num_anchors))

            for a in range(num_anchors):
                for m in range(num_filtered_measurements):
                    # 联合方差 S
                    S = predicted_uncertainties[a] + filtered_measurements[1, m]
                    # 残差 nu
                    nu = filtered_measurements[0, m] - predicted_measurements[a]
                    # 高斯似然
                    likelihood = (1.0 / np.sqrt(2 * np.pi * S)) * np.exp(-0.5 * nu**2 / S)
                    # 归一化因子
                    beta_matrix_filtered[m, a] = likelihood * (detection_probability / clutter_intensity)

            # --- C. 构建混合特征张量 (M_filtered, K, 5) ---
            legacy_feat = np.zeros((num_filtered_measurements, num_anchors, 5))

            # 预先计算所有锚点的预测RSS（用于Ch4的对级别RSS匹配）
            rss_pred = np.zeros(num_anchors)
            for a in range(num_anchors):
                safe_pred_dist = max(predicted_measurements[a], 0.1)
                rss_pred[a] = P_tx - 10 * n * np.log10(safe_pred_dist)

            for m in range(num_filtered_measurements):
                # 提取过滤后的测量幅度
                if filtered_measurements.shape[0] >= 3:
                    rss_meas = filtered_measurements[2, m]
                else:
                    rss_meas = 0.0

                for a in range(num_anchors):
                    # Ch0: Log-Prob (物理建议) - 直接使用过滤后的beta矩阵
                    legacy_feat[m, a, 0] = np.log(beta_matrix_filtered[m, a] + 1e-20)

                    # Ch1: 标准化残差 (GNN纠错核心)
                    std_dev = np.sqrt(filtered_measurements[1, m] + predicted_uncertainties[a])
                    legacy_feat[m, a, 1] = (filtered_measurements[0, m] - predicted_measurements[a]) / (std_dev + 1e-6)

                    # Ch2: 对数方差
                    legacy_feat[m, a, 2] = np.log(filtered_measurements[1, m] + predicted_uncertainties[a] + 1e-6)

                    # Ch3: 存在概率
                    legacy_feat[m, a, 3] = existence_probs[a]

                    # Ch4: [对级别] RSS残差（归一化）
                    # 这个特征描述"测量m与锚点a的RSS匹配度"
                    feat_amp_diff = abs(rss_meas - rss_pred[a]) / 5.0
                    legacy_feat[m, a, 4] = feat_amp_diff

            # --- D. 构建 New/Clutter 特征 (M_filtered, 1, 5) ---
            new_feat = np.zeros((num_filtered_measurements, 1, 5))
            # 计算 Xi 参考值 (Log域)
            mu_new = undetected_anchors_intensity[sensor]
            xi_val = np.log(1.0 + mu_new / clutter_intensity)

            new_feat[:, 0, 0] = xi_val  # Ch0
            new_feat[:, 0, 1] = 0.0     # Ch1
            new_feat[:, 0, 2] = 2.0     # Ch2 (背景方差)
            new_feat[:, 0, 3] = 1.0     # Ch3
            # Ch4: 杂波的RSS特征（设为中等值）
            new_feat[:, 0, 4] = 0.5

            # --- E. GNN 训练与推理 ---
            use_gnn_result = False
            message_lhf_ratios = None
            messages_new = None

            if gnn_trainer is not None and num_anchors > 0 and num_filtered_measurements > 0:
                try:
                    # 拼接特征
                    hybrid_input = np.concatenate([legacy_feat, new_feat], axis=1)
                    hybrid_tensor = torch.from_numpy(hybrid_input).float().unsqueeze(0).to(gnn_trainer.device)

                    # 判断是单头还是双头架构
                    if use_dual_head:
                        # ===== 双头架构推理 =====
                        # [改进] GRU状态现在在trainer内部维护，不需要外部传递

                        # 前向推理（返回3个值：assoc_probs, dustbin_probs, loss）
                        assoc_probs, dustbin_probs, loss = gnn_trainer.step(
                            hybrid_tensor,
                            filtered_measurements,
                            predicted_measurements,
                            predicted_uncertainties
                        )

                        # assoc_probs: (M_filtered, K) 关联概率
                        # dustbin_probs: (M_filtered,) 杂波概率 [0, 1]
                        # 注意：dustbin_probs = 1.0 - quality_scores

                        # 打印 Loss 和数据统计
                        if step % 10 == 0:
                            print(f"  [双头GNN] Sensor {sensor+1}, Step {step}, Loss: {loss:.4f}")

                            # 统计杂波识别情况
                            # dustbin_probs高 = 杂波，dustbin_probs低 = 真实信号
                            num_good = np.sum(dustbin_probs < 0.5)  # 杂波概率<0.5 = 好点
                            num_bad = np.sum(dustbin_probs >= 0.5)  # 杂波概率>=0.5 = 杂波
                            print(f"    杂波识别: {num_good}/{len(dustbin_probs)} 个真实信号, {num_bad} 个杂波")

                            # [关键验证] 统计输入数据中的真实杂波比例
                            # 使用内在一致性误差判断真实杂波
                            high_error_count = np.sum(filtered_intrinsic_errors > 2.0)  # >20dB
                            mid_error_count = np.sum((filtered_intrinsic_errors > 1.2) & (filtered_intrinsic_errors <= 2.0))
                            low_error_count = np.sum(filtered_intrinsic_errors <= 1.2)
                            print(f"    输入数据: 高误差(>20dB)={high_error_count}, 中误差(12-20dB)={mid_error_count}, 低误差(<12dB)={low_error_count}")

                            # 验证GNN是否正确识别了杂波
                            if high_error_count > 0:
                                print(f"    ✓ GNN能看到真实杂波！（移除了15dB硬过滤，现在是50dB）")

                        # 决策：预热期后使用 GNN 结果
                        if step > warmup_steps:
                            use_gnn_result = True

                            # [关键] 双头架构的输出处理
                            # 1. 杂波概率（已经由trainer转换好了）
                            gnn_dustbin_filtered = dustbin_probs

                            # 2. 关联概率（已经是概率，不需要再除以dustbin）
                            gnn_probs = assoc_probs

                            # 3. 映射回原始测量索引
                            full_gnn_probs = np.zeros((num_measurements, num_anchors))
                            full_gnn_dustbin = np.ones(num_measurements)  # 被过滤的测量默认为杂波

                            valid_indices = np.where(valid_mask)[0]
                            full_gnn_probs[valid_indices, :] = gnn_probs
                            full_gnn_dustbin[valid_indices] = gnn_dustbin_filtered

                            # 4. 转换为 BP 消息格式
                            # message_lhf_ratios: (M, K) 表示测量-锚点关联强度

                            # [关键修正] 质量头和关联头完全解耦
                            # 核心思想：
                            #   - 质量头：只用于过滤明显的杂波（dustbin > 0.8）和检测新锚点
                            #   - 关联头：完全负责已知锚点的关联，不受质量头影响
                            #
                            # 原问题：message = assoc × (1 - dustbin)
                            #   - 质量头误判（dustbin=0.9）会压制关联消息
                            #   - 导致锚点存在概率剧烈波动（0.09 ↔ 0.99）
                            #   - 产生 OSPA 尖峰
                            #
                            # 改进：对于已知锚点的关联，完全信任关联头的几何匹配
                            #   - 只过滤明显的杂波（dustbin > 0.8）
                            #   - 其他测量完全使用关联概率

                            # 过滤明显的杂波
                            quality_mask = (full_gnn_dustbin < 0.8)  # dustbin < 0.8 才参与关联

                            # 对于通过质量检查的测量，完全使用关联概率
                            message_lhf_ratios = full_gnn_probs.copy()
                            message_lhf_ratios[~quality_mask, :] = 0.0  # 明显杂波的关联消息设为0

                            # 5. 新锚点消息：GNN判断（质量高 + 关联低 → 可能是新锚点）
                            max_assoc_prob = np.max(full_gnn_probs, axis=1)
                            messages_new_gnn = (1.0 - full_gnn_dustbin) * (1.0 - max_assoc_prob)

                            # [修正] 直接使用GNN的判断，不融合BP先验
                            # 原因：
                            # 1. new_input_bp ≈ 1.0-1.2，影响微小
                            # 2. 直接相乘破坏概率语义（结果可能>1）
                            # 3. 失配模式下，应完全信任数据驱动的GNN判断
                            messages_new_raw = messages_new_gnn

                            # [新增] 新锚点候选缓冲区验证机制
                            # 防止瞬时噪声被误判为新锚点，要求连续多帧检测
                            messages_new = np.maximum(messages_new_raw, 1e-6)

                            # 更新候选缓冲区
                            current_candidates = candidate_anchors[sensor]
                            new_candidates = {}

                            for m in range(num_measurements):
                                if messages_new_raw[m] > candidate_threshold:
                                    # 高置信度新锚点候选
                                    # 计算测量位置（粗略估计）
                                    meas_dist = measurements[0, m]
                                    agent_pos = np.mean(predicted_particles_agent[0:2, :], axis=1)

                                    # 查找是否有相近的候选（距离 < 0.5m）
                                    found_match = False
                                    for cand_id, cand_info in current_candidates.items():
                                        # 简单距离检查（这里用测量距离差作为近似）
                                        if abs(meas_dist - cand_info['distance']) < 0.5:
                                            # 找到匹配的候选
                                            if step - cand_info['last_step'] <= candidate_max_gap:
                                                # 间隔不超过最大允许间隔，更新计数
                                                new_candidates[cand_id] = {
                                                    'distance': meas_dist,
                                                    'count': cand_info['count'] + 1,
                                                    'last_step': step
                                                }
                                                found_match = True

                                                # 检查是否达到最小帧数要求
                                                if new_candidates[cand_id]['count'] < candidate_min_frames:
                                                    # 还未达到要求，降低 messages_new
                                                    messages_new[m] = messages_new_raw[m] * 0.1
                                                # else: 达到要求，保持原始 messages_new
                                                break

                                    if not found_match:
                                        # 新候选，创建条目
                                        new_cand_id = f"step{step}_meas{m}"
                                        new_candidates[new_cand_id] = {
                                            'distance': meas_dist,
                                            'count': 1,
                                            'last_step': step
                                        }
                                        # 第一次检测，大幅降低 messages_new
                                        messages_new[m] = messages_new_raw[m] * 0.1

                            # 更新缓冲区（只保留活跃的候选）
                            candidate_anchors[sensor] = new_candidates

                    else:
                        # ===== 单头架构推理（原有逻辑） =====
                        gnn_probs, gnn_dustbin, loss = gnn_trainer.step(
                            hybrid_tensor,
                            filtered_measurements,  # 使用过滤后的测量
                            predicted_measurements,
                            predicted_uncertainties
                        )

                        # 打印 Loss
                        if step % 10 == 0:
                            print(f"  [单头GNN] Sensor {sensor+1}, Step {step}, Loss: {loss:.4f}")

                        # 决策：预热期后使用 GNN 结果
                        if step > warmup_steps:
                            use_gnn_result = True

                            # 映射回原始测量索引
                            full_gnn_probs = np.zeros((num_measurements, num_anchors))
                            full_gnn_dustbin = np.ones(num_measurements)  # 被过滤的测量默认为杂波

                            valid_indices = np.where(valid_mask)[0]
                            full_gnn_probs[valid_indices, :] = gnn_probs
                            full_gnn_dustbin[valid_indices] = gnn_dustbin

                            # 转换为 BP 消息格式
                            message_lhf_ratios = full_gnn_probs / (full_gnn_dustbin[:, np.newaxis] + 1e-10)

                            # 新锚点消息：GNN判断（dustbin低 → 可能是新锚点）
                            messages_new_gnn = np.exp(-0.5 * full_gnn_dustbin)

                            # [修正] 直接使用GNN的判断，不融合BP先验
                            # 原因同双头架构：保持概率语义，完全信任数据驱动判断
                            messages_new_raw = messages_new_gnn

                            # [新增] 新锚点候选缓冲区验证机制（单头架构）
                            messages_new = np.maximum(messages_new_raw, 1e-6)

                            # 更新候选缓冲区
                            current_candidates = candidate_anchors[sensor]
                            new_candidates = {}

                            for m in range(num_measurements):
                                if messages_new_raw[m] > candidate_threshold:
                                    # 高置信度新锚点候选
                                    meas_dist = measurements[0, m]

                                    # 查找是否有相近的候选
                                    found_match = False
                                    for cand_id, cand_info in current_candidates.items():
                                        if abs(meas_dist - cand_info['distance']) < 0.5:
                                            if step - cand_info['last_step'] <= candidate_max_gap:
                                                new_candidates[cand_id] = {
                                                    'distance': meas_dist,
                                                    'count': cand_info['count'] + 1,
                                                    'last_step': step
                                                }
                                                found_match = True

                                                if new_candidates[cand_id]['count'] < candidate_min_frames:
                                                    messages_new[m] = messages_new_raw[m] * 0.1
                                                break

                                    if not found_match:
                                        new_cand_id = f"step{step}_meas{m}"
                                        new_candidates[new_cand_id] = {
                                            'distance': meas_dist,
                                            'count': 1,
                                            'last_step': step
                                        }
                                        messages_new[m] = messages_new_raw[m] * 0.1

                            candidate_anchors[sensor] = new_candidates
                        
                except Exception as e:
                    print(f"  [GNN] Error at step {step}, sensor {sensor}: {e}")

            # --- E. 回退到传统 BP ---
            if not use_gnn_result:
                # [新增] 纯BP模式下的三区间软过滤策略
                # 与GNN训练标签生成逻辑保持一致（gnn_trainer_improved.py lines 814-831）

                # 路径损耗参数（与杂波生成保持一致）
                P_tx = 15.41
                n = 2.0

                # 计算所有测量的RSS内在一致性误差
                bp_intrinsic_errors = np.zeros(num_measurements)
                bp_quality_weights = np.ones(num_measurements)  # 质量权重

                for m in range(num_measurements):
                    # 提取测量RSS
                    if measurements.shape[0] >= 3:
                        rss_meas = measurements[2, m]
                    else:
                        rss_meas = 0.0

                    # 使用测量距离计算理论RSS
                    safe_meas_dist = max(measurements[0, m], 0.1)
                    rss_theory = P_tx - 10 * n * np.log10(safe_meas_dist)

                    # 内在一致性误差（dB）
                    intrinsic_error_db = abs(rss_meas - rss_theory)
                    bp_intrinsic_errors[m] = intrinsic_error_db

                    # [三区间软过滤策略] 与GNN训练标签生成一致
                    # 区间1: 核心真值区 (0-6dB) → 权重 1.0（完全信任）
                    # 区间2: 模糊区 (6-15dB) → 权重 0.5（部分信任）
                    # 区间3: 核心杂波区 (>15dB) → 权重 0.0（完全过滤）

                    if intrinsic_error_db < 6.0:
                        # 核心真值区：肯定是真实信号
                        bp_quality_weights[m] = 1.0
                    elif intrinsic_error_db <= 15.0:
                        # 模糊区：可能是NLoS真信号，也可能是强反射杂波
                        # 给部分权重，让BP的几何匹配去决定
                        bp_quality_weights[m] = 0.5
                    else:
                        # 核心杂波区：肯定是杂波
                        bp_quality_weights[m] = 0.0

                # 统计三区间分布
                if step % 50 == 0:
                    core_true_count = np.sum(bp_intrinsic_errors < 6.0)
                    ambiguous_count = np.sum((bp_intrinsic_errors >= 6.0) & (bp_intrinsic_errors <= 15.0))
                    core_clutter_count = np.sum(bp_intrinsic_errors > 15.0)

                    print(f"\n[纯BP三区间软过滤 - Step {step}]")
                    print(f"  总测量数: {num_measurements}")
                    print(f"  ┌─ 区间1: 核心真值 (RSS误差<6dB): {core_true_count} ({core_true_count/num_measurements*100:.1f}%) → 权重=1.0")
                    print(f"  ├─ 区间2: 模糊区 (6-15dB): {ambiguous_count} ({ambiguous_count/num_measurements*100:.1f}%) → 权重=0.5")
                    print(f"  └─ 区间3: 核心杂波 (RSS误差>15dB): {core_clutter_count} ({core_clutter_count/num_measurements*100:.1f}%) → 权重=0.0")

                # 使用所有测量进行BP关联（不硬过滤）
                (association_probabilities, association_probabilities_new,
                 message_lhf_ratios, messages_new) = calculate_association_probabilities_ga(
                    measurements, predicted_measurements, predicted_uncertainties,
                    weights_anchor, new_input_bp, parameters
                )

                # [关键] 应用质量权重到关联消息
                # 这是软过滤：不是完全删除测量，而是降低其关联权重
                # 核心真值区（权重1.0）：完全信任BP的关联
                # 模糊区（权重0.5）：部分信任，降低影响
                # 核心杂波区（权重0.0）：完全忽略
                message_lhf_ratios = message_lhf_ratios * bp_quality_weights[:, np.newaxis]
                messages_new = messages_new * bp_quality_weights

            # 对每个锚点计算粒子权重，结合检测概率和测量似然
            num_anchors = predicted_particles_anchors.shape[2]
            weights = np.zeros((num_particles, num_anchors))

            # 内存安全版：循环处理每个测量，支持高杂波场景
            if num_measurements > 0:
                # 初始化权重为未检测概率
                weights[:, :] = (1 - detection_probability)

                # 预计算所有测量的方差和因子
                measurement_variances = measurements[1, :]  # shape: (num_measurements,)
                factors = (1 / np.sqrt(2 * np.pi * measurement_variances) *
                          detection_probability / clutter_intensity)  # shape: (num_measurements,)

                # 循环处理每一个测量，避免创建巨大的3D矩阵
                for m in range(num_measurements):
                    z = measurements[0, m]  # 测量距离
                    R = measurement_variances[m]  # 测量方差
                    factor = factors[m]  # 归一化因子
                    # 获取该测量对应的关联比率向量 (num_anchors,)
                    ratio = message_lhf_ratios[m, :]

                    # 计算距离差 (num_particles, num_anchors)
                    diff = z - predicted_range

                    # 累加权重 (利用广播机制)
                    # factor * ratio[None, :] -> (1, num_anchors)
                    # exp term -> (num_particles, num_anchors)
                    weights += (factor * ratio[np.newaxis, :] * np.exp(-0.5 * (diff**2) / R))
            else:
                # 没有测量时，所有权重为未检测概率
                weights[:, :] = (1 - detection_probability)

            # 对每个锚点进行后续处理（这部分仍需循环，因为涉及重采样等操作）
            for anchor in range(num_anchors):

                # 计算该锚点预测存在概率
                predicted_existence = np.sum(weights_anchor[:, anchor])

                # 计算锚点存在的后验概率
                alive_update = np.sum(predicted_existence * (1 / num_particles) * weights[:, anchor])
                dead_update = 1 - predicted_existence
                posterior_particles_anchors[sensor][anchor]['posteriorExistence'] = (
                    alive_update / (alive_update + dead_update)
                )

                # 重采样粒子，依据权重调整粒子集合
                weight_sum = np.sum(weights[:, anchor])
                if weight_sum > 0:
                    idx_resampling = resample_systematic(
                        weights[:, anchor] / weight_sum, num_particles
                    )
                else:
                    idx_resampling = np.arange(num_particles)

                posterior_particles_anchors[sensor][anchor]['x'] = (
                    predicted_particles_anchors[:, idx_resampling, anchor]
                )
                posterior_particles_anchors[sensor][anchor]['w'] = (
                    posterior_particles_anchors[sensor][anchor]['posteriorExistence'] /
                    num_particles * np.ones(num_particles)
                )

                # 计算锚点位置的均值估计和存在概率
                estimated_anchors[sensor][step][anchor]['x'] = np.mean(
                    posterior_particles_anchors[sensor][anchor]['x'], axis=1
                )
                estimated_anchors[sensor][step][anchor]['posteriorExistence'] = (
                    posterior_particles_anchors[sensor][anchor]['posteriorExistence']
                )

                # 计算归一化权重的对数，防止数值溢出
                weights[:, anchor] = predicted_existence * weights[:, anchor] + dead_update
                weights[:, anchor] = np.log(weights[:, anchor])
                weights[:, anchor] = weights[:, anchor] - np.max(weights[:, anchor])

            # 更新锚点数量
            num_estimated_anchors[sensor, step] = len(estimated_anchors[sensor][step])

            # 汇总所有锚点权重，为移动体粒子加权
            weights_sensors[:, sensor] = np.sum(weights, axis=1)
            weights_sensors[:, sensor] = weights_sensors[:, sensor] - np.max(weights_sensors[:, sensor])

            # 更新未检测锚点强度，乘以未检测概率
            undetected_anchors_intensity[sensor] = (
                undetected_anchors_intensity[sensor] * (1 - detection_probability)
            )

            # 更新新锚点的后验存在概率和粒子集
            for measurement in range(num_measurements):
                new_anchor_idx = num_anchors + measurement
                constant = new_particles_anchors[measurement]['constant']
                posterior_existence = (
                    messages_new[measurement] * constant /
                    (messages_new[measurement] * constant + 1)
                )

                # 扩展列表以容纳新锚点
                if new_anchor_idx >= len(posterior_particles_anchors[sensor]):
                    posterior_particles_anchors[sensor].append({
                        'x': new_particles_anchors[measurement]['x'],
                        'w': posterior_existence / num_particles,
                        'posteriorExistence': posterior_existence
                    })
                    estimated_anchors[sensor][step].append({
                        'x': np.mean(new_particles_anchors[measurement]['x'], axis=1),
                        'posteriorExistence': posterior_existence,
                        'generatedAt': step
                    })
                else:
                    posterior_particles_anchors[sensor][new_anchor_idx]['posteriorExistence'] = posterior_existence
                    posterior_particles_anchors[sensor][new_anchor_idx]['x'] = new_particles_anchors[measurement]['x']
                    posterior_particles_anchors[sensor][new_anchor_idx]['w'] = posterior_existence / num_particles
                    estimated_anchors[sensor][step][new_anchor_idx]['x'] = np.mean(
                        new_particles_anchors[measurement]['x'], axis=1
                    )
                    estimated_anchors[sensor][step][new_anchor_idx]['posteriorExistence'] = posterior_existence
                    estimated_anchors[sensor][step][new_anchor_idx]['generatedAt'] = step

            # 删除不可靠的锚点（存在概率低于阈值）
            estimated_anchors[sensor][step], posterior_particles_anchors[sensor] = delete_unreliable_va(
                estimated_anchors[sensor][step],
                posterior_particles_anchors[sensor],
                unreliability_threshold
            )

            # 更新数量记录
            num_estimated_anchors[sensor, step] = len(estimated_anchors[sensor][step])

        # 汇总所有传感器权重，归一化移动体粒子权重
        weights_sensors = np.sum(weights_sensors, axis=1)
        weights_sensors = weights_sensors - np.max(weights_sensors)
        weights_sensors = np.exp(weights_sensors)
        weights_sensors = weights_sensors / np.sum(weights_sensors)

        # 保存部分关键时间步的锚点粒子状态，用于分析
        if step in storing_idx:
            idx = storing_idx.index(step)
            posterior_particles_anchors_storage[idx] = [
                [anchor.copy() for anchor in sensor_anchors]
                for sensor_anchors in posterior_particles_anchors
            ]

        # 更新移动体估计轨迹
        if known_track:
            # 已知轨迹时，直接用预测粒子均值
            estimated_trajectory[:, step] = np.mean(predicted_particles_agent, axis=1)
            posterior_particles_agent = predicted_particles_agent
        else:
            # 未知轨迹时，基于权重重采样粒子，更新估计
            estimated_trajectory[:, step] = predicted_particles_agent @ weights_sensors
            posterior_particles_agent = predicted_particles_agent[
                :, resample_systematic(weights_sensors, num_particles)
            ]

        # 计算估计误差（距离误差）并打印输出
        exec_time_per_step[step] = time.time() - start_time
        error_agent = calc_distance(
            true_trajectory[0:2, step:step+1],
            estimated_trajectory[0:2, step:step+1]
        )
        # 如果error_agent是数组，提取标量值
        if isinstance(error_agent, np.ndarray):
            error_agent = error_agent.item() if error_agent.size == 1 else error_agent[0]

        print(f'Time instance: {step + 1}')  # +1 for MATLAB-style output
        # 自动打印所有传感器的锚点数量
        for sensor in range(num_sensors):
            print(f'Number of Anchors Sensor {sensor + 1}: {num_estimated_anchors[sensor, step]}')
        print(f'Position error agent: {error_agent:.6f}')
        print(f'Execution Time: {exec_time_per_step[step]:.4f}')
        print('---------------------------------------------------\n')

    return estimated_trajectory, estimated_anchors, posterior_particles_anchors_storage, num_estimated_anchors