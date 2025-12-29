"""
bp_slam/core/gnn_trainer_sparse_gat_v2.py
稀疏图 GAT 训练器 V2 - 简化版

改进：
1. 使用 graph_builder_v2 构建简化的稀疏图
2. 边特征：3维 [Δx, Δy, Σ]
3. 节点特征：3维语义特征
4. 删除 Dustbin 边，杂波判断完全由质量头完成
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


class SparseGATTrainer_V2:
    """
    稀疏图 GAT 训练器 V2

    支持双头架构（质量头 + 关联头）
    使用简化的 3 维边特征和语义节点特征
    """

    def __init__(self, device='cuda', lr=1e-4, hidden_dim=64, num_layers=2,
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
            lr: 学习率
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

        # 初始化模型
        self.model = SparseGAT_V2_DualHead(
            node_dim=3,  # 3维语义节点特征
            edge_dim=3,  # 3维边特征 [Δx, Δy, Σ]
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            use_temporal_gru=use_temporal_gru
        ).to(device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
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
        self.logger.info(f"训练器初始化完成 - 设备: {device}, 学习率: {lr}")

        # 隐状态（每个传感器独立）
        self.hidden_states = {}

        # EMA 模型
        if use_ema:
            self.ema_model = SparseGAT_V2_DualHead(
                node_dim=3,
                edge_dim=3,
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

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=lr * 0.01
        )

        # 加载权重
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        print(f"✓ 稀疏 GAT V2 训练器初始化完成")
        print(f"  - 边特征: 3维 [Δx, Δy, Σ]")
        print(f"  - 节点特征: 3维语义特征")
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
             update_weights=True):
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

        返回:
            assoc_probs: (M, K) 关联概率矩阵
            dustbin_probs: (M,) Dustbin 概率
            final_scale: (M,) 方差缩放因子（固定为 1）
            loss: 标量损失值
        """

        self.step_count += 1

        # 1. 构建稀疏图 V2（无 Dustbin）
        (node_features, edge_index, edge_attr, node_types,
         num_measurements, num_anchors) = build_sparse_graph_v2(
            filtered_measurements, predicted_measurements, predicted_uncertainties,
            existence_probs, predicted_particles_agent, predicted_particles_anchors,
            weights_anchor, P_tx, n,
            distance_threshold=self.distance_threshold,
            beta_threshold=self.beta_threshold
        )

        # 转移到设备
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)
        node_types = node_types.to(self.device)

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
                ground_truth_labels=ground_truth_labels  # 传递监督标签
            )

            # 反向传播（梯度累积）
            loss.backward()

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
                           num_measurements, num_anchors, ground_truth_labels=None):
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

        返回:
            total_loss: 总损失
            quality_loss: 质量损失
            assoc_loss: 关联损失
        """

        # 1. 质量损失
        if ground_truth_labels is not None:
            # 使用真实标签（监督学习）
            quality_loss = self._compute_quality_loss_supervised(
                quality_logits, ground_truth_labels
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
                num_measurements, num_anchors
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

    def _compute_quality_loss_supervised(self, quality_logits, ground_truth_labels):
        """计算质量损失（监督学习版本）- 使用加权 BCE 处理类别不平衡"""

        # 从标签中提取 is_clutter
        is_clutter = ground_truth_labels['is_clutter']  # (M,) numpy bool array

        # 转换为 torch tensor
        target_quality = torch.from_numpy(~is_clutter).float().to(self.device)  # True信号=1, 杂波=0

        # 计算类别数量
        num_signals = torch.sum(target_quality)
        num_clutter = len(target_quality) - num_signals

        # [修改] 极端的类别加权 - 压制模型过度预测真值
        # 问题：模型倾向于把所有测量都预测为真值（概率 > 0.9）
        # 原因：真值样本多（5个），杂波样本少（1个），模型为了降低 Loss 倾向于预测真值
        # 解决：大幅降低真值的权重，提高杂波的权重
        #
        # 策略：pos_weight = 0.15（真值权重很低）
        #      neg_weight = 1.0（杂波权重正常）
        # 效果：猜错一个杂波的惩罚是猜错一个真值的 6-7 倍
        #      模型会变得非常保守，不敢轻易预测真值

        if num_signals > 0 and num_clutter > 0:
            # 极端加权：大幅降低正样本权重
            pos_weight = 0.15  # 真值权重很低
            neg_weight = 1.0   # 杂波权重正常

            # 为每个样本分配权重
            weights = torch.where(target_quality == 1.0, pos_weight, neg_weight)
        else:
            # 如果只有一类，使用均匀权重
            weights = torch.ones_like(target_quality)

        # 加权 BCE 损失
        quality_probs = torch.sigmoid(quality_logits)
        loss = F.binary_cross_entropy(quality_probs, target_quality, weight=weights)

        # [DEBUG] 打印原始概率值和目标值（同时输出到控制台和日志）
        pred_probs_np = quality_probs.detach().cpu().numpy().flatten()
        target_np = target_quality.detach().cpu().numpy().flatten()

        # 统计预测分布
        num_pred_signals = np.sum(pred_probs_np > 0.5)
        num_true_signals = np.sum(target_np == 1.0)

        msg = f"\n[质量头准确率检查]"
        msg += f"\n  预测: {num_pred_signals}/{len(pred_probs_np)} 个真实信号 ({num_pred_signals/len(pred_probs_np)*100:.1f}%)"
        msg += f"\n  真实: {num_true_signals}/{len(target_np)} 个真实信号 ({num_true_signals/len(target_np)*100:.1f}%)"
        msg += f"\n  Loss: {loss.item():.6f}"

        print(msg)
        self.logger.info(msg.replace('\n', ' | '))

        return loss

    def _compute_association_loss_supervised(self, edge_index, edge_logits, ground_truth_labels,
                                            num_measurements, num_anchors):
        """计算关联损失（监督学习版本）- 使用加权 BCE 处理类别不平衡"""

        # 从标签中提取 true_id
        true_ids = ground_truth_labels['true_id']  # (M,) numpy int array, -1表示杂波

        # 为每条边生成真实标签
        src_nodes = edge_index[0].cpu().numpy()
        dst_nodes = edge_index[1].cpu().numpy()

        target_labels = []
        for i in range(edge_index.shape[1]):
            src = src_nodes[i]  # 测量节点索引
            dst = dst_nodes[i]  # 锚点节点索引

            # 锚点节点索引范围：[num_measurements, num_measurements + num_anchors)
            if dst >= num_measurements and dst < num_measurements + num_anchors:
                anchor_idx = dst - num_measurements

                # 检查这条边是否是正确匹配
                if true_ids[src] == anchor_idx:
                    # 正确匹配
                    target_labels.append(1.0)
                else:
                    # 错误匹配
                    target_labels.append(0.0)
            else:
                target_labels.append(0.0)

        target_labels = torch.tensor(target_labels, device=self.device, dtype=torch.float32)

        # 计算类别权重（处理类别不平衡）
        num_positive = torch.sum(target_labels)
        num_negative = len(target_labels) - num_positive

        if num_positive > 0 and num_negative > 0:
            # [修改] 调低正样本权重，避免模型过度预测正样本
            # 原来：pos_weight = total / (2 * num_positive)
            # 问题：当正样本很少时（如 6/42），权重会非常高（42/12=3.5）
            # 解决：使用更温和的权重，限制最大权重
            total = len(target_labels)

            # 方案：使用平方根缩放，降低极端权重
            pos_weight = torch.sqrt(torch.tensor(total / num_positive, device=self.device))
            neg_weight = torch.sqrt(torch.tensor(total / num_negative, device=self.device))

            # 限制最大权重为 3.0，避免过度偏向正样本
            pos_weight = torch.clamp(pos_weight, max=3.0)
            neg_weight = torch.clamp(neg_weight, max=3.0)

            # 为每条边分配权重
            weights = torch.where(target_labels == 1.0, pos_weight, neg_weight)
        else:
            # 如果只有一类，使用均匀权重
            weights = torch.ones_like(target_labels)

        # 加权 BCE 损失
        edge_probs = torch.sigmoid(edge_logits)
        loss = F.binary_cross_entropy(edge_probs, target_labels, weight=weights)

        # [DEBUG] 打印关联头准确率统计（同时输出到控制台和日志）
        edge_probs_np = edge_probs.detach().cpu().numpy().flatten()
        target_np = target_labels.detach().cpu().numpy().flatten()

        # 统计预测分布
        num_pred_positive = np.sum(edge_probs_np > 0.5)
        num_true_positive = np.sum(target_np == 1.0)

        msg = f"\n[关联头准确率检查]"
        msg += f"\n  预测: {num_pred_positive}/{len(edge_probs_np)} 条正确匹配边 ({num_pred_positive/len(edge_probs_np)*100:.1f}%)"
        msg += f"\n  真实: {num_true_positive}/{len(target_np)} 条正确匹配边 ({num_true_positive/len(target_np)*100:.1f}%)"
        msg += f"\n  Loss: {loss.item():.6f}"

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
