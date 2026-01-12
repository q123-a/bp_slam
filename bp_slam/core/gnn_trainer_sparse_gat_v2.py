"""
bp_slam/core/gnn_trainer_sparse_gat_v2.py
稀疏图 GAT 训练器 V2 - 简化版

改进：
1. 使用 graph_builder_v2 构建简化的稀疏图
2. 边特征：3维 [Δx, Δy, Σ]
3. 节点特征：3维语义特征
4. 删除 Dustbin 边，杂波判断完全由质量头完成
5. 使用 Focal Loss 处理类别不平衡
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from .gnn_model_sparse_gat_v2 import SparseGAT_V2_DualHead, TORCH_GEOMETRIC_AVAILABLE
from .graph_builder_v2 import build_sparse_graph_v2, sparse_to_dense_output_v2
from ..utils.focal_loss import FocalLossForAssociation, FocalLossForQuality


class SparseGATTrainer_V2:
    """
    稀疏图 GAT 训练器 V2

    支持双头架构（质量头 + 关联头）
    使用简化的 3 维边特征和语义节点特征
    """

    def __init__(self, device='cuda', lr=1e-5, hidden_dim=64, num_layers=2,
                 heads=4, dropout=0.1, use_temporal_gru=False,
                 checkpoint_path=None, seed=42,
                 use_ema=True, ema_decay=0.999,
                 quality_threshold=0.25, assoc_threshold=3.0,
                 quality_weight=1.5, assoc_weight=1.0,
                 adaptive_weighting=True,
                 distance_threshold=None, beta_threshold=None):
        """
        初始化稀疏 GAT 训练器 V2

        参数:
            device: 设备
            lr: 学习率 (默认 1e-5，降低以提高稳定性)
            hidden_dim: 隐藏维度
            num_layers: GAT 层数
            heads: 注意力头数
            dropout: Dropout 概率
            use_temporal_gru: 是否使用 GRU
            checkpoint_path: 权重加载路径
            seed: 随机种子
            use_ema: 是否使用 EMA
            ema_decay: EMA 衰减率
            quality_threshold: 质量阈值
            assoc_threshold: 关联阈值
            quality_weight: 质量损失权重
            assoc_weight: 关联损失权重
            adaptive_weighting: 是否使用自适应权重
            distance_threshold: 稀疏图距离阈值（None=不过滤）
            beta_threshold: 稀疏图 Beta 阈值（None=不过滤）
        """

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required for Sparse GAT V2 trainer")

        self.device = device
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.seed = seed
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.quality_threshold = quality_threshold
        self.assoc_threshold = assoc_threshold
        self.quality_weight = quality_weight
        self.assoc_weight = assoc_weight
        self.adaptive_weighting = adaptive_weighting
        self.distance_threshold = distance_threshold
        self.beta_threshold = beta_threshold

        # 设置随机种子
        if seed is not None:
            self._set_seed(seed)

        # 初始化模型（ReID 增强版：4维特征）
        self.model = SparseGAT_V2_DualHead(
            node_dim=4,  # 4维语义节点特征 [type, feature1, feature2, fingerprint]
            edge_dim=4,  # 4维边特征 [Δx, Δy, Σ, ΔF]
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            use_temporal_gru=use_temporal_gru
        ).to(device)

        # ============================================================
        # [分离优化器] 双头用更高学习率
        # ============================================================
        # 问题：双头学习太慢，关联头全0，质量头全1
        # 解决：双头用 10x 学习率
        # ============================================================
        assoc_params = list(self.model.assoc_head.parameters())
        quality_params = list(self.model.quality_head.parameters())
        other_params = [p for n, p in self.model.named_parameters() 
                       if 'assoc_head' not in n and 'quality_head' not in n]
        
        self.optimizer = optim.AdamW([
            {'params': other_params, 'lr': lr},
            {'params': assoc_params, 'lr': lr * 10},   # 关联头 10x 学习率
            {'params': quality_params, 'lr': lr * 10}  # 质量头 10x 学习率
        ], weight_decay=1e-5)
        
        self.step_count = 0

        # Loss 历史
        self.loss_history = []
        self.quality_loss_history = []
        self.assoc_loss_history = []

        # [新增] 初始化日志文件
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = log_dir / f'training_{timestamp}.log'

        # 配置日志记录器
        self.logger = logging.getLogger(f'SparseGATTrainer_V2_{timestamp}')
        self.logger.setLevel(logging.INFO)

        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.logger.info(f"日志文件已创建: {self.log_file}")
        self.logger.info(f"训练器初始化完成 - 设备: {device}, 学习率: {lr} (双头: {lr*10})")

        # 隐状态（每个传感器独立）
        self.hidden_states = {}

        # EMA 模型（ReID 增强版：4维特征）
        if use_ema:
            self.ema_model = SparseGAT_V2_DualHead(
                node_dim=4,  # 4维节点特征
                edge_dim=4,  # 4维边特征
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                heads=heads,
                dropout=dropout,
                use_temporal_gru=use_temporal_gru
            ).to(device)
            self.ema_model.load_state_dict(self.model.state_dict())
            for param in self.ema_model.parameters():
                param.requires_grad = False
            print(f"✓ EMA 已启用 (decay={ema_decay})")
        else:
            self.ema_model = None

        # ============================================================
        # [Loss Functions] 处理类别不平衡
        # ============================================================
        # 关联头：正样本(真匹配)~14% → 动态加权BCE + 10x学习率
        # 质量头：杂波(0)~15% → 动态加权BCE（neg_weight按比例）
        # ============================================================
        self.focal_loss_assoc = FocalLossForAssociation()  # 动态加权BCE
        self.focal_loss_quality = FocalLossForQuality()    # 动态加权BCE
        print(f"✓ Loss 已配置:")
        print(f"  [关联头] 动态加权BCE (pos_weight=neg/pos, clamp[2,8])")
        print(f"  [质量头] 动态加权BCE (neg_weight=pos/neg, clamp[1,6])")
        print(f"  [质量头] 标签平滑已启用: 信号→0.95, 杂波→0.05")

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=lr * 0.01
        )

        # 加载权重
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        print(f"✓ 稀疏 GAT V2 训练器初始化完成 (ReID 增强版)")
        print(f"  - 节点特征: 4维 [type, feature1, feature2, fingerprint]")
        print(f"  - 边特征: 4维 [Δx, Δy, Σ, ΔF]")
        print(f"  - 稀疏图过滤: 距离阈值={distance_threshold}, Beta阈值={beta_threshold}")

    def _set_seed(self, seed):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def reset_hidden_state(self, sensor_id=None):
        """重置隐状态"""
        if sensor_id is None:
            self.hidden_states = {}
        else:
            if sensor_id in self.hidden_states:
                del self.hidden_states[sensor_id]

    def step(self, filtered_measurements, predicted_measurements, predicted_uncertainties,
             existence_probs, predicted_particles_agent, predicted_particles_anchors,
             weights_anchor, P_tx=15.41, n=2.0,
             num_iterations=1, sensor_id=0, ground_truth_labels=None,
             update_weights=True, reid_fingerprints=None):
        """
        执行一步训练/推理

        参数:
            filtered_measurements: (3, M) [距离, 方差, RSS]
            predicted_measurements: (K,) 预测距离
            predicted_uncertainties: (K,) 预测方差
            existence_probs: (K,) 锚点存在概率
            predicted_particles_agent: (4, num_particles) 移动体粒子
            predicted_particles_anchors: (2, num_particles, K) 锚点粒子
            weights_anchor: (num_particles, K) 锚点权重
            P_tx: 发射功率
            n: 路径损耗指数
            num_iterations: 迭代次数
            sensor_id: 传感器 ID
            ground_truth_labels: 监督学习标签（可选）
            update_weights: 是否更新权重和学习率（多传感器时只在最后一个传感器更新）
            reid_fingerprints: (K,) 锚点历史指纹列表（可选，用于 ReID）

        返回:
            assoc_probs: (M, K) 关联概率矩阵
            dustbin_probs: (M,) Dustbin 概率
            final_scale: (M,) 方差缩放因子（固定为 1）
            loss: 标量损失值
        """

        self.step_count += 1

        # 1. 构建稀疏图 V2（无 Dustbin，带 ReID）
        (node_features, edge_index, edge_attr, node_types,
         num_measurements, num_anchors) = build_sparse_graph_v2(
            filtered_measurements, predicted_measurements, predicted_uncertainties,
            existence_probs, predicted_particles_agent, predicted_particles_anchors,
            weights_anchor, P_tx, n,
            distance_threshold=self.distance_threshold,
            beta_threshold=self.beta_threshold,
            reid_fingerprints=reid_fingerprints  # [ReID] 传递锚点历史指纹
        )

        # 转移到设备
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)
        node_types = node_types.to(self.device)

        # ============================================================
        # [输入检查] 检测 NaN 和异常值
        # ============================================================
        if torch.isnan(node_features).any():
            self.logger.warning("!!! 警告：node_features 包含 NaN !!!")
            print("!!! 警告：node_features 包含 NaN !!!")
        if torch.isnan(edge_attr).any():
            self.logger.warning("!!! 警告：edge_attr 包含 NaN !!!")
            print("!!! 警告：edge_attr 包含 NaN !!!")
        
        # 检查 ReID 特征（edge_attr 第4列）
        if edge_attr.shape[0] > 0 and edge_attr.shape[1] >= 4:
            reid_feat = edge_attr[:, 3]
            reid_min = reid_feat.min().item()
            reid_max = reid_feat.max().item()
            reid_mean = reid_feat.mean().item()
            if reid_max > 10.0:
                self.logger.warning(f"!!! ReID特征异常大: max={reid_max:.2f} !!!")
                print(f"!!! ReID特征异常大: max={reid_max:.2f} !!!")

        # 2. 训练
        total_loss = 0.0
        total_quality_loss = 0.0
        total_assoc_loss = 0.0

        self.model.train()

        for iter_idx in range(num_iterations):
            # 获取隐状态
            h_in = None
            if sensor_id in self.hidden_states:
                cached_h = self.hidden_states[sensor_id].detach()
                # 检查隐状态大小是否匹配（节点数量可能变化）
                if cached_h.shape[0] == node_features.shape[0]:
                    h_in = cached_h
                else:
                    # 节点数量变化，重置隐状态
                    h_in = None

            # 前向传播
            edge_logits, quality_logits, h_out = self.model(
                node_features, edge_index, edge_attr, node_types,
                num_measurements, h_in
            )

            # 更新隐状态
            self.hidden_states[sensor_id] = h_out

            # 计算损失
            loss, quality_loss, assoc_loss = self._compute_joint_loss(
                edge_index, edge_logits, quality_logits,
                filtered_measurements, predicted_measurements, predicted_uncertainties,
                num_measurements, num_anchors,
                ground_truth_labels=ground_truth_labels,  # 传递监督标签
                edge_attr=edge_attr,  # [ReID Sanity Check] 传递边特征
                node_features=node_features  # [DEBUG] 传递节点特征用于调试
            )

            # 反向传播（梯度累积）
            loss.backward()
            
            # [DEBUG] 监控质量头梯度
            if iter_idx == 0:  # 只在第一次迭代打印
                grad_info = []
                for name, param in self.model.quality_head.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        param_norm = param.data.norm().item()
                        grad_info.append(f"{name}: grad={grad_norm:.6f} param={param_norm:.4f}")
                    else:
                        grad_info.append(f"{name}: grad=None")
                if grad_info:
                    self.logger.info(f"[DEBUG-质量头梯度] {' | '.join(grad_info)}")

            # 只在 update_weights=True 时更新权重
            if update_weights:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.05)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # 更新 EMA
                if self.ema_model is not None:
                    self._update_ema()

            total_loss += loss.item()
            total_quality_loss += quality_loss.item()
            total_assoc_loss += assoc_loss.item()

        # 平均损失
        avg_loss = total_loss / num_iterations
        avg_quality_loss = total_quality_loss / num_iterations
        avg_assoc_loss = total_assoc_loss / num_iterations

        # 只在 update_weights=True 时更新学习率和记录损失
        if update_weights:
            self.scheduler.step()

            # 记录
            self.loss_history.append(avg_loss)
            self.quality_loss_history.append(avg_quality_loss)
            self.assoc_loss_history.append(avg_assoc_loss)

        # 3. 推理（使用 EMA 模型）
        with torch.no_grad():
            inference_model = self.ema_model if self.use_ema else self.model
            inference_model.eval()

            eval_h_in = None
            if sensor_id in self.hidden_states:
                cached_h = self.hidden_states[sensor_id].detach()
                # 检查隐状态大小是否匹配
                if cached_h.shape[0] == node_features.shape[0]:
                    eval_h_in = cached_h
                else:
                    # 节点数量变化，重置隐状态
                    eval_h_in = None

            eval_edge_logits, eval_quality_logits, _ = inference_model(
                node_features, edge_index, edge_attr, node_types,
                num_measurements, eval_h_in
            )

            # 温度缩放
            T = 0.1
            edge_probs = torch.sigmoid(eval_edge_logits / T)
            quality_scores = torch.sigmoid(eval_quality_logits)

            # 转换为密集矩阵格式
            assoc_probs, _ = sparse_to_dense_output_v2(
                edge_index.cpu(), edge_probs.cpu(), num_measurements, num_anchors
            )

            # 质量分数 -> Dustbin 概率
            dustbin_probs = 1.0 - quality_scores.cpu().numpy()

            # 方差缩放因子（固定为 1）
            final_scale = np.ones(num_measurements)

        return assoc_probs, dustbin_probs, final_scale, avg_loss

    def _compute_joint_loss(self, edge_index, edge_logits, quality_logits,
                           filtered_measurements, predicted_measurements, predicted_uncertainties,
                           num_measurements, num_anchors, ground_truth_labels=None, edge_attr=None,
                           node_features=None):
        """
        计算联合损失

        参数:
            edge_index: (2, E) 边索引
            edge_logits: (E,) 边的 logits
            quality_logits: (M,) 质量 logits
            filtered_measurements: (3, M)
            predicted_measurements: (K,)
            predicted_uncertainties: (K,)
            num_measurements: M
            num_anchors: K
            ground_truth_labels: 监督学习标签（可选），字典 {'true_id': array, 'is_clutter': array}
            edge_attr: (E, 4) 边特征 [Δx, Δy, Σ, ΔF]（可选，用于 ReID Sanity Check）

        返回:
            total_loss: 总损失
            quality_loss: 质量损失
            assoc_loss: 关联损失
        """

        # 1. 质量损失
        if ground_truth_labels is not None:
            # 使用真实标签（监督学习）
            quality_loss = self._compute_quality_loss_supervised(
                quality_logits, ground_truth_labels, 
                node_features=node_features, num_measurements=num_measurements
            )
        else:
            # 使用 RSS 内在一致性（自监督学习）
            quality_loss = self._compute_quality_loss(
                quality_logits, filtered_measurements
            )

        # 2. 关联损失
        if ground_truth_labels is not None:
            # 使用真实标签（监督学习）
            assoc_loss = self._compute_association_loss_supervised(
                edge_index, edge_logits, ground_truth_labels,
                num_measurements, num_anchors, edge_attr  # [ReID] 传递边特征
            )
        else:
            # 使用几何匹配（自监督学习）
            assoc_loss = self._compute_association_loss(
                edge_index, edge_logits, quality_logits,
                filtered_measurements, predicted_measurements, predicted_uncertainties,
                num_measurements, num_anchors
            )

        # 3. 自适应权重
        if self.adaptive_weighting:
            quality_weight, assoc_weight = self._compute_adaptive_weights(
                quality_loss.item(), assoc_loss.item()
            )
        else:
            quality_weight = self.quality_weight
            assoc_weight = self.assoc_weight

        # 总损失
        total_loss = quality_weight * quality_loss + assoc_weight * assoc_loss

        return total_loss, quality_loss, assoc_loss

    def _compute_quality_loss(self, quality_logits, filtered_measurements):
        """计算质量损失（RSS 内在一致性）"""

        M = filtered_measurements.shape[1]
        P_tx = 15.41
        n = 2.0

        # 计算 RSS 内在一致性误差
        rss_errors = []
        for m in range(M):
            if filtered_measurements.shape[0] >= 3:
                rss_meas = filtered_measurements[2, m]
                safe_dist = max(filtered_measurements[0, m], 0.1)
                rss_theory = P_tx - 10 * n * np.log10(safe_dist)
                rss_error = abs(rss_meas - rss_theory)
                rss_errors.append(rss_error)
            else:
                rss_errors.append(0.0)

        rss_errors = np.array(rss_errors)

        # 三区间软标签
        target_quality = torch.zeros(M, device=self.device)
        quality_mask = torch.zeros(M, device=self.device)

        for m in range(M):
            if rss_errors[m] < 6.0:
                target_quality[m] = 0.95
                quality_mask[m] = 1.0
            elif rss_errors[m] <= 15.0:
                target_quality[m] = 0.5
                quality_mask[m] = 0.5
            else:
                target_quality[m] = 0.05
                quality_mask[m] = 1.0

        # BCE 损失
        quality_probs = torch.sigmoid(quality_logits)
        loss = F.binary_cross_entropy(quality_probs, target_quality, weight=quality_mask)

        return loss

    def _compute_association_loss(self, edge_index, edge_logits, quality_logits,
                                  filtered_measurements, predicted_measurements, predicted_uncertainties,
                                  num_measurements, num_anchors):
        """计算关联损失（几何匹配）"""

        # 简化版：使用距离残差作为标签
        src_nodes = edge_index[0].cpu().numpy()
        dst_nodes = edge_index[1].cpu().numpy()

        target_labels = []
        for i in range(edge_index.shape[1]):
            src = src_nodes[i]
            dst = dst_nodes[i]

            # 所有边都是锚点边（无Dustbin）
            if dst >= num_measurements and dst < num_measurements + num_anchors:
                anchor_idx = dst - num_measurements
                distance_residual = abs(
                    filtered_measurements[0, src] - predicted_measurements[anchor_idx]
                )
                # 距离越小，标签越接近 1
                label = 1.0 if distance_residual < 3.0 else 0.0
                target_labels.append(label)
            else:
                target_labels.append(0.0)

        target_labels = torch.tensor(target_labels, device=self.device, dtype=torch.float32)

        # BCE 损失
        edge_probs = torch.sigmoid(edge_logits)
        loss = F.binary_cross_entropy(edge_probs, target_labels)

        return loss

    def _compute_quality_loss_supervised(self, quality_logits, ground_truth_labels,
                                         node_features=None, num_measurements=None):
        """计算质量损失（监督学习版本）- 使用 Focal Loss 处理类别不平衡"""

        # 从标签中提取 is_clutter
        is_clutter = ground_truth_labels['is_clutter']  # (M,) numpy bool array

        # ============================================================
        # [标签平滑] 防止模型过度自信
        # ============================================================
        # 硬标签 0/1 会让模型输出极端logit，导致梯度消失
        # 软标签让模型保持适度不确定性，梯度更稳定
        # 
        # 真信号: 1.0 -> 0.9 (留10%余地)
        # 杂波:   0.0 -> 0.1 (不完全否定)
        # ============================================================
        label_smoothing = 0.1
        target_quality = torch.from_numpy(~is_clutter).float().to(self.device)  # True信号=1, 杂波=0
        target_quality = target_quality * (1 - label_smoothing) + 0.5 * label_smoothing
        # 结果: 信号=0.95, 杂波=0.05

        # 计算类别数量 (用原始标签)
        num_signals = torch.sum(torch.from_numpy(~is_clutter).float())
        num_clutter = len(is_clutter) - num_signals

        # ============================================================
        # [Focal Loss] 质量头
        # ============================================================
        # 问题：真信号(1)极多，杂波(0)极少 (85:15)
        # 传统BCE：模型会躺平预测全1
        # 
        # Focal Loss 解决方案：
        # - alpha=0.15: 极端偏向杂波(少数类)权重
        # - gamma=1.0: 保持梯度流动，不过度抑制简单样本
        # - 配合标签平滑防止过拟合
        # ============================================================
        
        loss = self.focal_loss_quality(quality_logits, target_quality)

        # 只记录到日志，不打印到控制台
        quality_probs = torch.sigmoid(quality_logits)
        pred_probs_np = quality_probs.detach().cpu().numpy().flatten()
        logits_np = quality_logits.detach().cpu().numpy().flatten()
        # [BUG FIX] 使用原始标签计算真实信号数，而非平滑后的标签
        # target_np 是平滑后的 (信号=0.95, 杂波=0.05)，不能用 == 1.0 判断
        num_pred_signals = np.sum(pred_probs_np > 0.5)
        num_true_signals = int(num_signals)  # 直接使用之前计算的原始值
        
        msg = f"[质量头-Focal] 预测: {num_pred_signals}/{len(pred_probs_np)}, 真实: {num_true_signals}/{len(pred_probs_np)}, 杂波: {int(num_clutter)}, Loss: {loss.item():.4f}"
        self.logger.info(msg)
        
        # ============================================================
        # [DEBUG] 比较杂波和信号的logit值和节点特征
        # ============================================================
        signal_mask = ~is_clutter  # True for signals
        clutter_mask = is_clutter   # True for clutter
        
        if np.any(signal_mask) and np.any(clutter_mask):
            signal_logits = logits_np[signal_mask]
            clutter_logits = logits_np[clutter_mask]
            signal_probs = pred_probs_np[signal_mask]
            clutter_probs = pred_probs_np[clutter_mask]
            
            debug_msg = (
                f"  [DEBUG-质量头] "
                f"信号logit: mean={np.mean(signal_logits):.2f} std={np.std(signal_logits):.2f} | "
                f"杂波logit: mean={np.mean(clutter_logits):.2f} std={np.std(clutter_logits):.2f} | "
                f"信号prob: {np.mean(signal_probs):.3f} | 杂波prob: {np.mean(clutter_probs):.3f} | "
                f"差异: {np.mean(signal_logits) - np.mean(clutter_logits):.2f}"
            )
            self.logger.info(debug_msg)
            
            # [DEBUG] 比较节点特征
            if node_features is not None and num_measurements is not None:
                # 节点特征: [type, feature1, feature2, fingerprint]
                meas_features = node_features[:num_measurements].detach().cpu().numpy()
                signal_features = meas_features[signal_mask]
                clutter_features = meas_features[clutter_mask]
                
                # 计算每个特征维度的均值
                feat_names = ['type', 'RSS_norm', 'uncertainty', 'fingerprint']
                feat_debug = "  [DEBUG-节点特征] "
                for i, name in enumerate(feat_names):
                    sig_mean = np.mean(signal_features[:, i])
                    clu_mean = np.mean(clutter_features[:, i])
                    diff = sig_mean - clu_mean
                    feat_debug += f"{name}: sig={sig_mean:.3f} clu={clu_mean:.3f} (Δ={diff:.3f}) | "
                self.logger.info(feat_debug)

        return loss

    def _compute_association_loss_supervised(self, edge_index, edge_logits, ground_truth_labels,
                                            num_measurements, num_anchors, edge_attr=None):
        """计算关联损失（监督学习版本）- 使用距离匹配生成标签"""

        # 从标签中提取 is_clutter
        true_ids = ground_truth_labels['true_id']  # (M,) numpy int array, -1表示杂波
        is_clutter_arr = (true_ids == -1)  # 用 true_id=-1 判断杂波，更准确

        # 为每条边生成真实标签
        src_nodes = edge_index[0].cpu().numpy()
        dst_nodes = edge_index[1].cpu().numpy()
        
        # 获取边的距离特征 (edge_attr[:, 0:2] 是 Δx, Δy)
        if edge_attr is not None:
            edge_attr_np = edge_attr.detach().cpu().numpy()
            delta_x = edge_attr_np[:, 0]
            delta_y = edge_attr_np[:, 1]
            edge_dist = np.sqrt(delta_x**2 + delta_y**2)
        else:
            edge_dist = np.ones(edge_index.shape[1]) * 999  # 无边特征时默认大距离

        # ============================================================
        # [距离匹配标签生成]
        # 对于每个非杂波测量，找到距离最近的锚点作为正样本
        # 这样可以绑定"正确关联=距离近"的规律
        # ============================================================
        
        # 为每个测量节点找到距离最近的锚点边
        meas_to_best_edge = {}  # meas_idx -> (best_edge_idx, min_dist)
        for i in range(edge_index.shape[1]):
            src = src_nodes[i]
            dst = dst_nodes[i]
            
            if dst >= num_measurements and dst < num_measurements + num_anchors:
                dist = edge_dist[i]
                if src not in meas_to_best_edge or dist < meas_to_best_edge[src][1]:
                    meas_to_best_edge[src] = (i, dist)

        target_labels = []
        edge_indices_positive = []
        edge_indices_negative = []

        for i in range(edge_index.shape[1]):
            src = src_nodes[i]
            dst = dst_nodes[i]

            if dst >= num_measurements and dst < num_measurements + num_anchors:
                # 判断是否为正样本：
                # 1. 测量不是杂波
                # 2. 这条边是该测量距离最近的锚点边
                # 3. 距离小于阈值（可选）
                is_signal = not is_clutter_arr[src]
                is_best_edge = (src in meas_to_best_edge and meas_to_best_edge[src][0] == i)
                dist_ok = edge_dist[i] < 3.0  # 距离阈值
                
                if is_signal and is_best_edge and dist_ok:
                    target_labels.append(1.0)
                    edge_indices_positive.append(i)
                else:
                    target_labels.append(0.0)
                    edge_indices_negative.append(i)
            else:
                target_labels.append(0.0)
                edge_indices_negative.append(i)

        target_labels = torch.tensor(target_labels, device=self.device, dtype=torch.float32)

        # [DEBUG] 检查标签匹配情况和距离分布
        num_signals = np.sum(~is_clutter_arr)
        num_clutter = np.sum(is_clutter_arr)
        num_pos_edges = len(edge_indices_positive)
        num_neg_edges = len(edge_indices_negative)
        
        # 计算正负样本的距离分布
        if len(edge_indices_positive) > 0:
            pos_dists = edge_dist[edge_indices_positive]
            pos_dist_mean = np.mean(pos_dists)
            pos_dist_max = np.max(pos_dists)
        else:
            pos_dist_mean = pos_dist_max = 0.0
            
        if len(edge_indices_negative) > 0:
            neg_dists = edge_dist[edge_indices_negative]
            neg_dist_mean = np.mean(neg_dists)
            neg_dist_min = np.min(neg_dists)
        else:
            neg_dist_mean = neg_dist_min = 0.0
        
        self.logger.info(
            f"[DEBUG-距离匹配标签] 信号={num_signals} 杂波={num_clutter} | "
            f"正样本边={num_pos_edges} (dist: mean={pos_dist_mean:.2f} max={pos_dist_max:.2f}) | "
            f"负样本边={num_neg_edges} (dist: mean={neg_dist_mean:.2f} min={neg_dist_min:.2f}) | "
            f"锚点数={num_anchors}"
        )

        # ============================================================
        # [加权Loss] 强迫GNN预测正样本
        # ============================================================
        # 核心思想：漏选一个真锚点的惩罚 >> 选错一个杂波的惩罚
        #
        # 问题：GNN太保守，宁可不选也不选错
        #       表现：预测 0/42，真实 6/42
        #
        # 解决：大幅提高正样本权重
        #       pos_weight=10.0 → 漏掉一个真1的代价是选错10个杂波的代价
        #
        # 预期行为变化：
        #   阶段1: 预测 0/42 → 适度增加
        #   阶段2: 收敛到准确
        # ============================================================

        num_positive = torch.sum(target_labels)
        num_negative = len(target_labels) - num_positive

        # ============================================================
        # [加权BCE] 关联头 - 不再使用 Focal Loss
        # ============================================================
        # 问题：Focal Loss 导致模型在两个极端之间摇摆
        #       alpha=0.9 → 全预测1
        #       alpha=0.5 → 全预测0
        # 
        # 根本原因：gamma 项压缩了梯度，模型陷入局部最优
        # 
        # 解决方案：简单的加权 BCE (pos_weight=6.0)
        # - 正样本权重是负样本的6倍（匹配实际比例1:6）
        # - 保持正常的梯度流，不压缩梯度
        # ============================================================
        
        loss = self.focal_loss_assoc(edge_logits, target_labels)

        # 统计预测分布
        edge_probs = torch.sigmoid(edge_logits)
        edge_probs_np = edge_probs.detach().cpu().numpy().flatten()
        edge_logits_np = edge_logits.detach().cpu().numpy().flatten()
        target_np = target_labels.detach().cpu().numpy().flatten()
        num_pred_positive = np.sum(edge_probs_np > 0.5)
        num_true_positive = np.sum(target_np == 1.0)

        # ============================================================
        # [DEBUG-关联头] 分析正负样本的logit和概率分布
        # ============================================================
        if len(edge_indices_positive) > 0:
            pos_logits = edge_logits_np[edge_indices_positive]
            pos_probs = edge_probs_np[edge_indices_positive]
            pos_logit_mean = np.mean(pos_logits)
            pos_logit_std = np.std(pos_logits)
            pos_prob_mean = np.mean(pos_probs)
        else:
            pos_logit_mean = pos_logit_std = pos_prob_mean = 0.0
            
        if len(edge_indices_negative) > 0:
            neg_logits = edge_logits_np[edge_indices_negative]
            neg_probs = edge_probs_np[edge_indices_negative]
            neg_logit_mean = np.mean(neg_logits)
            neg_logit_std = np.std(neg_logits)
            neg_prob_mean = np.mean(neg_probs)
        else:
            neg_logit_mean = neg_logit_std = neg_prob_mean = 0.0
        
        logit_diff = pos_logit_mean - neg_logit_mean
        self.logger.info(
            f"[DEBUG-关联头] 正样本logit: mean={pos_logit_mean:.2f} std={pos_logit_std:.2f} | "
            f"负样本logit: mean={neg_logit_mean:.2f} std={neg_logit_std:.2f} | "
            f"正prob: {pos_prob_mean:.3f} | 负prob: {neg_prob_mean:.3f} | logit差异: {logit_diff:.2f}"
        )

        # ============================================================
        # [ReID 指纹分析] 核心调试信息
        # ============================================================
        # 期望看到：
        # - 正样本边 (真关联): ΔF ≈ 0.0 (指纹匹配)
        # - 负样本边 (杂波):   ΔF > 1.0 (指纹不匹配)
        #   - Ghost杂波: ΔF ≈ 1.0-3.0 (物理规律但弱)
        #   - Noise杂波: ΔF ≈ 1.0-3.0 (随机，截断到3.0)
        # ============================================================
        
        msg = f"\n[关联头-Focal] 预测:{num_pred_positive}/{len(edge_probs_np)} 真实:{int(num_positive)} Loss:{loss.item():.4f}"

        if edge_attr is not None and edge_attr.shape[1] >= 4:
            edge_attr_np = edge_attr.detach().cpu().numpy()
            delta_x = edge_attr_np[:, 0]  # Δx
            delta_y = edge_attr_np[:, 1]  # Δy  
            sigma = edge_attr_np[:, 2]    # Σ (不确定性之和)
            reid_diff = edge_attr_np[:, 3]  # ΔF

            # [DEBUG-边特征] 正负样本边特征对比
            if len(edge_indices_positive) > 0 and len(edge_indices_negative) > 0:
                pos_dx = delta_x[edge_indices_positive]
                pos_dy = delta_y[edge_indices_positive]
                pos_sigma = sigma[edge_indices_positive]
                neg_dx = delta_x[edge_indices_negative]
                neg_dy = delta_y[edge_indices_negative]
                neg_sigma = sigma[edge_indices_negative]
                
                # 计算欧式距离 sqrt(Δx² + Δy²)
                pos_dist = np.sqrt(pos_dx**2 + pos_dy**2)
                neg_dist = np.sqrt(neg_dx**2 + neg_dy**2)
                
                self.logger.info(
                    f"[DEBUG-边特征] 正样本: dist={np.mean(pos_dist):.2f}±{np.std(pos_dist):.2f} Σ={np.mean(pos_sigma):.2f} | "
                    f"负样本: dist={np.mean(neg_dist):.2f}±{np.std(neg_dist):.2f} Σ={np.mean(neg_sigma):.2f} | "
                    f"dist差异: {np.mean(neg_dist)-np.mean(pos_dist):.2f}"
                )

            # 正样本的 ReID Diff
            if len(edge_indices_positive) > 0:
                pos_reid = reid_diff[edge_indices_positive]
                pos_mean = np.mean(pos_reid)
                pos_std = np.std(pos_reid)
                pos_max = np.max(pos_reid)
                # 统计ΔF<0.5的比例（应该很高，表示指纹匹配好）
                pos_good_ratio = np.mean(pos_reid < 0.5) * 100
                msg += f"\n  [ReID正] ΔF: mean={pos_mean:.2f} std={pos_std:.2f} max={pos_max:.2f} <0.5占比={pos_good_ratio:.0f}%"

            # 负样本的 ReID Diff
            if len(edge_indices_negative) > 0:
                neg_reid = reid_diff[edge_indices_negative]
                neg_mean = np.mean(neg_reid)
                neg_std = np.std(neg_reid)
                neg_max = np.max(neg_reid)
                # 统计ΔF>1.0的比例（应该较高，表示杂波可区分）
                neg_bad_ratio = np.mean(neg_reid > 1.0) * 100
                # 统计ΔF=1.0的比例（锚点无历史指纹）
                neg_neutral_ratio = np.mean(np.abs(neg_reid - 1.0) < 0.01) * 100
                msg += f"\n  [ReID负] ΔF: mean={neg_mean:.2f} std={neg_std:.2f} max={neg_max:.2f} >1.0占比={neg_bad_ratio:.0f}% =1.0(无历史)={neg_neutral_ratio:.0f}%"

            # 计算正负样本分离度
            if len(edge_indices_positive) > 0 and len(edge_indices_negative) > 0:
                separation = neg_mean - pos_mean
                msg += f"\n  [ReID分离度] 负-正={separation:.2f} (>1.0好)"

        print(msg)
        self.logger.info(msg.replace('\n', ' | '))

        return loss

    def _compute_adaptive_weights(self, quality_loss, assoc_loss):
        """计算自适应权重"""
        # 简单的自适应策略：保持两个损失的比例平衡
        if quality_loss > 0 and assoc_loss > 0:
            ratio = quality_loss / assoc_loss
            if ratio > 2.0:
                quality_weight = 1.0
                assoc_weight = 2.0
            elif ratio < 0.5:
                quality_weight = 2.0
                assoc_weight = 1.0
            else:
                quality_weight = 1.5
                assoc_weight = 1.0
        else:
            quality_weight = self.quality_weight
            assoc_weight = self.assoc_weight

        return quality_weight, assoc_weight

    def _update_ema(self):
        """更新 EMA 模型"""
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def save_checkpoint(self, path):
        """保存权重"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict() if self.ema_model else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'loss_history': self.loss_history,
            'quality_loss_history': self.quality_loss_history,
            'assoc_loss_history': self.assoc_loss_history
        }, path)
        print(f"✓ 权重已保存: {path}")

    def load_checkpoint(self, path):
        """加载权重"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.ema_model and checkpoint['ema_model_state_dict']:
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint.get('step_count', 0)
        self.loss_history = checkpoint.get('loss_history', [])
        self.quality_loss_history = checkpoint.get('quality_loss_history', [])
        self.assoc_loss_history = checkpoint.get('assoc_loss_history', [])
        print(f"✓ 权重已加载: {path}")

    def save_loss_history_to_csv(self, path):
        """保存损失历史到 CSV 文件"""
        import csv
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'Total_Loss', 'Quality_Loss', 'Assoc_Loss'])

            for i in range(len(self.loss_history)):
                total_loss = self.loss_history[i]
                quality_loss = self.quality_loss_history[i] if i < len(self.quality_loss_history) else 0.0
                assoc_loss = self.assoc_loss_history[i] if i < len(self.assoc_loss_history) else 0.0
                writer.writerow([i, total_loss, quality_loss, assoc_loss])

        print(f"✓ 损失历史已保存到 CSV: {path}")

    def print_loss_statistics(self, window_size=50):
        """打印损失统计信息"""
        if len(self.loss_history) == 0:
            print("  [Loss] 暂无损失记录")
            return

        # 最近的损失
        recent_total = self.loss_history[-1]
        recent_quality = self.quality_loss_history[-1] if self.quality_loss_history else 0.0
        recent_assoc = self.assoc_loss_history[-1] if self.assoc_loss_history else 0.0

        # 计算移动平均
        if len(self.loss_history) >= window_size:
            avg_total = np.mean(self.loss_history[-window_size:])
            avg_quality = np.mean(self.quality_loss_history[-window_size:]) if self.quality_loss_history else 0.0
            avg_assoc = np.mean(self.assoc_loss_history[-window_size:]) if self.assoc_loss_history else 0.0

            print(f"  [Loss] 当前: {recent_total:.4f} (质量: {recent_quality:.4f}, 关联: {recent_assoc:.4f})")
            print(f"  [Loss] 平均({window_size}步): {avg_total:.4f} (质量: {avg_quality:.4f}, 关联: {avg_assoc:.4f})")
        else:
            print(f"  [Loss] 当前: {recent_total:.4f} (质量: {recent_quality:.4f}, 关联: {recent_assoc:.4f})")
