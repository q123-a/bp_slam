"""
bp_slam/core/gnn_trainer_sparse_gat.py
稀疏图 GAT 训练器

使用稀疏图格式训练 GAT 模型
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from .gnn_model_sparse_gat import SparseGAT_DualHead, TORCH_GEOMETRIC_AVAILABLE
from .graph_builder import build_sparse_graph, sparse_to_dense_output


class SparseGATTrainer:
    """
    稀疏图 GAT 训练器

    支持双头架构（质量头 + 关联头）
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
        初始化稀疏 GAT 训练器

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
            raise ImportError("torch_geometric is required for Sparse GAT trainer")

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
        self.model = SparseGAT_DualHead(
            node_dim=3,
            edge_dim=5,
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

        # 隐状态（每个传感器独立）
        self.hidden_states = {}

        # EMA 模型
        if use_ema:
            self.ema_model = SparseGAT_DualHead(
                node_dim=3,
                edge_dim=5,
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

        print(f"✓ 稀疏 GAT 训练器初始化完成")
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
             existence_probs, beta_matrix_filtered, undetected_anchors_intensity,
             clutter_intensity, detection_probability, P_tx=15.41, n=2.0,
             num_iterations=1, sensor_id=0):
        """
        执行一步训练/推理

        参数:
            filtered_measurements: (3, M) [距离, 方差, RSS]
            predicted_measurements: (K,) 预测距离
            predicted_uncertainties: (K,) 预测方差
            existence_probs: (K,) 锚点存在概率
            beta_matrix_filtered: (M, K) Beta 矩阵
            undetected_anchors_intensity: 未检测锚点强度
            clutter_intensity: 杂波强度
            detection_probability: 检测概率
            P_tx: 发射功率
            n: 路径损耗指数
            num_iterations: 迭代次数
            sensor_id: 传感器 ID

        返回:
            assoc_probs: (M, K) 关联概率矩阵
            dustbin_probs: (M,) Dustbin 概率
            final_scale: (M,) 方差缩放因子（固定为 1）
            loss: 标量损失值
        """

        self.step_count += 1

        # 1. 构建稀疏图
        (node_features, edge_index, edge_attr, node_types,
         num_measurements, num_anchors) = build_sparse_graph(
            filtered_measurements, predicted_measurements, predicted_uncertainties,
            existence_probs, beta_matrix_filtered, undetected_anchors_intensity,
            clutter_intensity, detection_probability, P_tx, n,
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
                num_measurements, num_anchors
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.05)
            self.optimizer.step()

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

        # 更新学习率
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
            assoc_probs, dustbin_probs = sparse_to_dense_output(
                edge_index.cpu(), edge_probs.cpu(), num_measurements, num_anchors
            )

            # 质量分数 -> Dustbin 概率
            dustbin_probs_from_quality = 1.0 - quality_scores.cpu().numpy()

            # 合并两种 dustbin 概率（取平均）
            dustbin_probs = (dustbin_probs + dustbin_probs_from_quality) / 2.0

            # 方差缩放因子（固定为 1）
            final_scale = np.ones(num_measurements)

        return assoc_probs, dustbin_probs, final_scale, avg_loss

    def _compute_joint_loss(self, edge_index, edge_logits, quality_logits,
                           filtered_measurements, predicted_measurements, predicted_uncertainties,
                           num_measurements, num_anchors):
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

        返回:
            total_loss: 总损失
            quality_loss: 质量损失
            assoc_loss: 关联损失
        """

        # 1. 质量损失（基于 RSS 内在一致性）
        quality_loss = self._compute_quality_loss(
            quality_logits, filtered_measurements
        )

        # 2. 关联损失（基于几何匹配）
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
        # TODO: 实现完整的匈牙利算法

        src_nodes = edge_index[0].cpu().numpy()
        dst_nodes = edge_index[1].cpu().numpy()
        dustbin_idx = num_measurements + num_anchors

        target_labels = []
        for i in range(edge_index.shape[1]):
            src = src_nodes[i]
            dst = dst_nodes[i]

            if dst == dustbin_idx:
                # Dustbin 边：根据质量分数决定
                target_labels.append(0.0)  # 暂时设为 0
            elif dst >= num_measurements and dst < dustbin_idx:
                # 锚点边：根据距离残差决定
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
            'loss_history': self.loss_history
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
        print(f"✓ 权重已加载: {path}")
