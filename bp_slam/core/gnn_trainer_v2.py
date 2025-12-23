"""
bp_slam/core/gnn_trainer_v2.py
双头GNN训练器 V2 - 使用改进的 JointDualHeadGNNV2 模型

核心改进：
1. 使用 JointDualHeadGNNV2 模型（softmax聚合、边特征增强、Skip Connections）
2. 模糊逻辑三区间训练策略（核心真值区、模糊区、核心杂波区）
3. 双老师自监督：物理老师（RSS一致性）+ 几何老师（匈牙利算法）
4. 自适应损失权重平衡
5. 改进的GRU状态管理（自动维度检查）
6. 增强的调试信息输出

模型架构改进（来自 JointDualHeadGNNV2）：
- Softmax 聚合 (smooth max) 替代 Hard Max，梯度流向所有节点
- 边特征增强：拼接 [原始特征, 差分特征]
- Skip Connections：跨层连接提升梯度流动
- 更稳定的训练和更好的收敛性

接口兼容性：
- 完全兼容原版训练器接口
- 输入输出格式保持一致
- 可直接替换使用
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from .gnn_model_v2 import JointDualHeadGNNV2


class JointDualHeadTrainerV2:
    """
    双头联合训练器 V2：质量头 + 关联头的自监督训练
    使用改进的 JointDualHeadGNNV2 模型

    核心创新：
    1. 物理老师（Physics Teacher）：监督质量头，使用RSS内在一致性
       - 不依赖预测位置，只看物理特征
       - 三区间模糊逻辑训练策略

    2. 几何老师（Geometry Teacher）：监督关联头，使用匈牙利算法
       - 全局最优匹配
       - 只对高质量测量进行训练

    3. 自适应权重平衡：
       - 根据训练阶段动态调整质量损失和关联损失的权重
       - 防止某一头过拟合

    4. 模型架构改进：
       - Softmax 聚合提供更平滑的梯度
       - 边特征增强提升表达能力
       - Skip Connections 改善梯度流动
    """

    def __init__(self, device='cuda', lr=1e-3, hidden_dim=64, checkpoint_path=None, seed=42,
                 use_ema=True, ema_decay=0.999, use_lr_scheduler=True,
                 use_temporal_gru=False, use_layer_gru=False,
                 quality_threshold=0.5, assoc_threshold=3.0,
                 quality_weight=1.0, assoc_weight=2.0,
                 adaptive_weighting=True, debug_interval=50,
                 aggregation='softmax', gamma=3.0, edge_mode='concat',
                 skip_connections=None):
        """
        初始化双头训练器 V2

        参数:
            device: 设备 ('cuda' 或 'cpu')
            lr: 初始学习率
            hidden_dim: 隐藏层维度
            checkpoint_path: 权重文件路径
            seed: 随机种子
            use_ema: 是否使用EMA
            ema_decay: EMA衰减率
            use_lr_scheduler: 是否使用学习率调度器
            use_temporal_gru: 是否使用跨帧GRU
            use_layer_gru: 是否使用层内GRU
            quality_threshold: 质量判断阈值 (0.5)
            assoc_threshold: 关联熔断阈值 (3.0)
            quality_weight: 质量损失初始权重 (1.0)
            assoc_weight: 关联损失初始权重 (2.0)
            adaptive_weighting: 是否使用自适应权重平衡
            debug_interval: 调试信息打印间隔（步数）
            aggregation: 聚合方式 ('softmax', 'max', 'mean')
            gamma: softmax 温度参数 (默认 3.0)
            edge_mode: 边特征模式 ('diff', 'concat')
            skip_connections: 跨层连接字典 {source_layer: target_layer}
        """
        self.device = device
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.seed = seed
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.use_lr_scheduler = use_lr_scheduler
        self.use_temporal_gru = use_temporal_gru
        self.use_layer_gru = use_layer_gru
        self.quality_threshold = quality_threshold
        self.assoc_threshold = assoc_threshold
        self.quality_weight = quality_weight
        self.assoc_weight = assoc_weight
        self.adaptive_weighting = adaptive_weighting
        self.debug_interval = debug_interval
        self.aggregation = aggregation
        self.gamma = gamma
        self.edge_mode = edge_mode
        self.skip_connections = skip_connections

        # 设置随机种子
        if seed is not None:
            self._set_seed(seed)

        # 初始化模型（使用改进的 V2 模型）
        self.model = JointDualHeadGNNV2(
            input_dim=5,
            hidden_dim=hidden_dim,
            num_layers=2,
            use_temporal_gru=use_temporal_gru,
            use_layer_gru=use_layer_gru,
            aggregation=aggregation,
            gamma=gamma,
            edge_mode=edge_mode,
            skip_connections=skip_connections
        ).to(device)

        # 优化器
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)

        # EMA模型（可选）
        if use_ema:
            self.ema_model = JointDualHeadGNNV2(
                input_dim=5,
                hidden_dim=hidden_dim,
                num_layers=2,
                use_temporal_gru=use_temporal_gru,
                use_layer_gru=use_layer_gru,
                aggregation=aggregation,
                gamma=gamma,
                edge_mode=edge_mode,
                skip_connections=skip_connections
            ).to(device)
            self.ema_model.load_state_dict(self.model.state_dict())
            for param in self.ema_model.parameters():
                param.requires_grad = False
            print(f"✓ EMA 已启用 (decay={ema_decay})")
        else:
            self.ema_model = None

        # 学习率调度器
        if use_lr_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=1000, eta_min=lr * 0.01
            )
            print(f"✓ 学习率调度器已启用 (CosineAnnealing)")
        else:
            self.scheduler = None

        # 训练状态
        self.step_count = 0
        self.loss_history = []
        self.quality_loss_history = []
        self.assoc_loss_history = []

        # GRU隐藏状态（在trainer内部维护）
        self.hidden_state = None

        # 自适应权重历史
        self.weight_history = []

        # 加载权重（如果提供）
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        print(f"✓ 双头GNN训练器V2初始化完成")
        print(f"  - 模型: JointDualHeadGNNV2")
        print(f"  - 聚合方式: {aggregation} (gamma={gamma})")
        print(f"  - 边特征模式: {edge_mode}")
        print(f"  - 质量阈值: {quality_threshold}")
        print(f"  - 关联阈值: {assoc_threshold}")
        print(f"  - 损失权重: 质量={quality_weight}, 关联={assoc_weight}")
        print(f"  - 自适应权重: {adaptive_weighting}")

    def _set_seed(self, seed):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def reset_hidden_state(self):
        """
        重置GRU隐藏状态

        应在以下情况调用：
        1. 开始新的轨迹序列
        2. 检测到轨迹中断
        3. 测量数量或锚点数量发生显著变化
        """
        self.hidden_state = None

    def step(self, hybrid_tensor, measurements, predicted_measurements, predicted_variances, num_iterations=1, sensor_id=0):
        """
        执行一步训练/推理

        改进：
        1. GRU状态在trainer内部维护（不需要外部传递）
        2. 返回dustbin_probs而不是quality_scores（更符合SLAM接口）
        3. 梯度截断防止BPTT过长
        4. 支持多次迭代训练

        输入:
            hybrid_tensor: (1, M, K+1, 5) 混合特征
            measurements: (3, M) 测量数据 [距离, 方差, 幅度]
            sensor_id: int, 传感器ID（V2不使用，仅为接口兼容）
            predicted_measurements: (K,) 预测距离
            predicted_variances: (K,) 预测方差
            num_iterations: 每个时间步的迭代次数（默认1次）

        输出:
            assoc_probs: (M, K) 关联概率
            dustbin_probs: (M,) 杂波概率 [0, 1]
            loss: 标量损失值
        """
        self.step_count += 1
        hybrid_tensor = hybrid_tensor.to(self.device)

        # 多次迭代训练
        total_loss = 0.0
        total_quality_loss = 0.0
        total_assoc_loss = 0.0

        self.model.train()

        for iter_idx in range(num_iterations):
            # 梯度截断：防止BPTT过长导致梯度爆炸
            # 检查hidden state维度是否匹配
            h_in = None
            if self.hidden_state is not None:
                B, M, K_plus_1, _ = hybrid_tensor.shape
                K = K_plus_1 - 1
                expected_size = B * M * K

                if self.hidden_state.shape[0] == expected_size:
                    h_in = self.hidden_state.detach()
                else:
                    h_in = None

            # 前向推理
            assoc_logits, quality_scores, h_out = self.model(hybrid_tensor, h_in)

            # 更新内部GRU状态
            self.hidden_state = h_out

            # 计算双老师自监督损失
            loss, quality_loss, assoc_loss = self._compute_joint_loss(
                assoc_logits, quality_scores, measurements,
                predicted_measurements, predicted_variances
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.05)
            self.optimizer.step()

            # 更新EMA模型
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
        if self.scheduler is not None:
            self.scheduler.step()

        # 记录
        self.loss_history.append(avg_loss)
        self.quality_loss_history.append(avg_quality_loss)
        self.assoc_loss_history.append(avg_assoc_loss)

        # 格式化输出供SLAM使用
        with torch.no_grad():
            # 使用EMA模型进行推理（如果启用）
            inference_model = self.ema_model if self.use_ema else self.model
            inference_model.eval()

            # 重新前向推理（使用EMA模型）
            eval_h_in = None
            if self.hidden_state is not None:
                B, M, K_plus_1, _ = hybrid_tensor.shape
                K = K_plus_1 - 1
                expected_size = B * M * K
                if self.hidden_state.shape[0] == expected_size:
                    eval_h_in = self.hidden_state.detach()

            eval_assoc_logits, eval_quality_scores, _ = inference_model(hybrid_tensor, eval_h_in)

            # 关联概率: Softmax (M, K)
            assoc_probs = F.softmax(eval_assoc_logits, dim=2).squeeze(0).cpu().numpy()

            # 质量分数: Sigmoid (M,)
            quality = eval_quality_scores.squeeze(0).cpu().numpy()

            # 将双头输出转换为SLAM格式
            # 杂波概率 (Dustbin) = 1.0 - Quality
            dustbin_probs = 1.0 - quality

            # No variance inflation - return ones (no scaling)
            final_scale = np.ones(len(dustbin_probs))

        return assoc_probs, dustbin_probs, final_scale, avg_loss

    def _compute_joint_loss(self, assoc_logits, quality_scores, measurements,
                           predicted_measurements, predicted_variances):
        """
        计算联合损失：质量损失 + 关联损失

        双老师自监督：
        1. 物理老师：用RSS内在一致性监督质量头
        2. 几何老师：用匈牙利算法监督关联头
        """
        M = measurements.shape[1]
        K = predicted_measurements.shape[0]

        # 物理老师：生成质量标签
        quality_loss = self._compute_quality_loss(quality_scores, measurements)

        # 几何老师：生成关联标签（只对质量好的测量进行关联训练）
        assoc_loss = self._compute_association_loss(
            assoc_logits, quality_scores, measurements,
            predicted_measurements, predicted_variances
        )

        # 自适应权重平衡
        if self.adaptive_weighting:
            quality_weight, assoc_weight = self._compute_adaptive_weights(
                quality_loss.item(), assoc_loss.item()
            )
        else:
            quality_weight = self.quality_weight
            assoc_weight = self.assoc_weight

        # 总损失
        total_loss = quality_weight * quality_loss + assoc_weight * assoc_loss

        # 记录权重历史
        self.weight_history.append({
            'quality_weight': quality_weight,
            'assoc_weight': assoc_weight
        })

        # 调试信息
        if self.step_count % self.debug_interval == 0:
            print(f"\n[损失分解 - Step {self.step_count}]")
            print(f"  质量损失: {quality_loss.item():.4f} (权重={quality_weight:.3f})")
            print(f"  关联损失: {assoc_loss.item():.4f} (权重={assoc_weight:.3f})")
            print(f"  总损失: {total_loss.item():.4f}")
            print(f"  加权贡献: 质量={quality_weight * quality_loss.item():.4f}, 关联={assoc_weight * assoc_loss.item():.4f}")
            if self.scheduler is not None:
                print(f"  当前学习率: {self.scheduler.get_last_lr()[0]:.6f}")

        return total_loss, quality_loss, assoc_loss

    def _compute_adaptive_weights(self, quality_loss_val, assoc_loss_val):
        """
        自适应权重平衡

        策略：
        1. 如果某个损失过大，增加其权重
        2. 使用指数移动平均平滑权重变化
        3. 限制权重范围，防止极端值
        """
        # 计算损失比例
        total = quality_loss_val + assoc_loss_val + 1e-8
        quality_ratio = quality_loss_val / total
        assoc_ratio = assoc_loss_val / total

        # 反比例调整权重（损失大的给更大权重）
        quality_weight = 1.0 + assoc_ratio
        assoc_weight = 1.0 + quality_ratio

        # 归一化（保持总权重恒定）
        total_weight = quality_weight + assoc_weight
        quality_weight = quality_weight / total_weight * (self.quality_weight + self.assoc_weight)
        assoc_weight = assoc_weight / total_weight * (self.quality_weight + self.assoc_weight)

        # 限制权重范围
        quality_weight = np.clip(quality_weight, 0.5, 3.0)
        assoc_weight = np.clip(assoc_weight, 0.5, 3.0)

        return quality_weight, assoc_weight

    def _compute_quality_loss(self, quality_scores, measurements):
        """
        物理老师：用RSS内在一致性监督质量头

        三区间模糊逻辑训练策略：
        - 区间1: 核心真值区 (RSS误差<6dB) → 标签=0.95, 权重=1.0
        - 区间2: 模糊区 (6-15dB) → 标签=0.5, 权重=0.5
        - 区间3: 核心杂波区 (>15dB) → 标签=0.05, 权重=1.0
        """
        M = measurements.shape[1]

        # 提取数据
        z_dist = torch.from_numpy(measurements[0, :]).float().to(self.device)
        z_rss = torch.from_numpy(measurements[2, :]).float().to(self.device)

        # 物理模型参数
        P_tx = 15.41  # 发射功率 (dBm)
        n = 2.0       # 路径损耗指数

        # 计算理论RSS
        rss_theory = P_tx - 10.0 * n * torch.log10(z_dist + 1e-6)

        # 计算RSS误差（dB）
        rss_error = torch.abs(z_rss - rss_theory)

        # 生成质量伪标签
        target_quality = torch.zeros(M, device=self.device)
        quality_mask = torch.zeros(M, device=self.device)

        # 区间1: 核心真值区 (0-6dB)
        mask_core_true = (rss_error < 6.0)
        target_quality[mask_core_true] = 0.95
        quality_mask[mask_core_true] = 1.0

        # 区间3: 核心杂波区 (>15dB)
        mask_core_clutter = (rss_error > 15.0)
        target_quality[mask_core_clutter] = 0.05
        quality_mask[mask_core_clutter] = 1.0

        # 区间2: 模糊区 (6-15dB)
        mask_ambiguous = (~mask_core_true) & (~mask_core_clutter)
        target_quality[mask_ambiguous] = 0.5
        quality_mask[mask_ambiguous] = 0.5

        # 调试信息
        if self.step_count % self.debug_interval == 0:
            core_true_count = mask_core_true.sum().item()
            core_clutter_count = mask_core_clutter.sum().item()
            ambiguous_count = mask_ambiguous.sum().item()
            avg_quality = quality_scores.squeeze(0).mean().item()

            print(f"\n[质量头诊断 - 模糊逻辑三区间 - Step {self.step_count}]")
            print(f"  总测量数: {M}")
            print(f"  ┌─ 区间1: 核心真值 (RSS误差<6dB): {core_true_count} ({core_true_count/M*100:.1f}%)")
            print(f"  ├─ 区间2: 模糊区 (6-15dB): {ambiguous_count} ({ambiguous_count/M*100:.1f}%)")
            print(f"  └─ 区间3: 核心杂波 (RSS误差>15dB): {core_clutter_count} ({core_clutter_count/M*100:.1f}%)")
            print(f"  平均质量分数: {avg_quality:.3f}")
            print(f"  RSS误差范围: [{rss_error.min().item():.1f}, {rss_error.max().item():.1f}] dB")

        # 计算BCE损失
        if quality_mask.sum() > 0:
            loss = F.binary_cross_entropy(
                quality_scores.squeeze(0),
                target_quality,
                weight=quality_mask,
                reduction='sum'
            ) / quality_mask.sum()
        else:
            loss = torch.tensor(0.0, device=self.device)

        return loss

    def _compute_association_loss(self, assoc_logits, quality_scores, measurements,
                                  predicted_measurements, predicted_variances):
        """
        几何老师：用匈牙利算法监督关联头

        关键：只对质量好的测量进行关联训练
        """
        M = measurements.shape[1]
        K = predicted_measurements.shape[0]

        # 提取质量好的测量
        quality_np = quality_scores.squeeze(0).cpu().detach().numpy()
        valid_mask = quality_np > self.quality_threshold
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=self.device)

        # 提取有效测量的子集
        z_dist = torch.from_numpy(measurements[0, valid_indices]).float().to(self.device)
        var_meas = torch.from_numpy(measurements[1, valid_indices]).float().to(self.device)

        z_pred = torch.from_numpy(predicted_measurements).float().to(self.device)
        var_pred = torch.from_numpy(predicted_variances).float().to(self.device)

        # 计算几何代价矩阵
        joint_std = torch.sqrt(var_meas.unsqueeze(1) + var_pred.unsqueeze(0))
        diff_mat = torch.abs(z_dist.unsqueeze(1) - z_pred.unsqueeze(0))
        cost_matrix = diff_mat / (joint_std + 1e-6)

        # 匈牙利算法生成匹配标签
        cost_np = cost_matrix.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)

        # 熔断：过滤掉代价太大的匹配
        target_assoc = torch.full((len(valid_indices),), -1, dtype=torch.long, device=self.device)

        for i, r in enumerate(row_ind):
            if cost_np[r, col_ind[i]] < self.assoc_threshold:
                target_assoc[r] = col_ind[i]

        # 计算关联损失（只对匹配成功的）
        match_mask = (target_assoc != -1)

        # 调试信息
        if self.step_count % self.debug_interval == 0:
            matched_count = match_mask.sum().item()
            print(f"\n[关联头诊断 - Step {self.step_count}]")
            print(f"  质量好的测量: {len(valid_indices)}/{M} ({len(valid_indices)/M*100:.1f}%)")
            print(f"  匹配成功 (cost<{self.assoc_threshold}): {matched_count}/{len(valid_indices)} ({matched_count/max(len(valid_indices),1)*100:.1f}%)")
            if matched_count > 0:
                matched_costs = [cost_np[r, col_ind[i]] for i, r in enumerate(row_ind) if cost_np[r, col_ind[i]] < self.assoc_threshold]
                print(f"  匹配代价范围: [{min(matched_costs):.2f}, {max(matched_costs):.2f}]")

        if match_mask.sum() > 0:
            sub_logits = assoc_logits.squeeze(0)[valid_indices]
            loss = F.cross_entropy(
                sub_logits[match_mask],
                target_assoc[match_mask],
                label_smoothing=0.1
            )
        else:
            loss = torch.tensor(0.0, device=self.device)

        return loss

    def _update_ema(self):
        """更新EMA模型"""
        if self.ema_model is None:
            return
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def save_checkpoint(self, checkpoint_path, epoch=None, additional_info=None):
        """保存模型权重"""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'hidden_dim': self.hidden_dim,
            'lr': self.lr,
            'loss_history': self.loss_history,
            'quality_loss_history': self.quality_loss_history,
            'assoc_loss_history': self.assoc_loss_history,
            'weight_history': self.weight_history,
            'quality_threshold': self.quality_threshold,
            'assoc_threshold': self.assoc_threshold,
            'aggregation': self.aggregation,
            'gamma': self.gamma,
            'edge_mode': self.edge_mode,
        }

        if self.ema_model is not None:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if epoch is not None:
            checkpoint['epoch'] = epoch

        if additional_info is not None:
            checkpoint['additional_info'] = additional_info

        torch.save(checkpoint, checkpoint_path)
        print(f"✓ 双头GNN V2权重已保存: {checkpoint_path}")
        print(f"  - 训练步数: {self.step_count}")

    def load_checkpoint(self, checkpoint_path):
        """加载模型权重"""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            print(f"⚠ 权重文件不存在: {checkpoint_path}")
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])

            if self.ema_model is not None and 'ema_model_state_dict' in checkpoint:
                self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.step_count = checkpoint.get('step_count', 0)
            self.loss_history = checkpoint.get('loss_history', [])
            self.quality_loss_history = checkpoint.get('quality_loss_history', [])
            self.assoc_loss_history = checkpoint.get('assoc_loss_history', [])
            self.weight_history = checkpoint.get('weight_history', [])

            print(f"✓ 双头GNN V2权重已加载: {checkpoint_path}")
            print(f"  - 训练步数: {self.step_count}")
            print(f"  - Loss 历史记录: {len(self.loss_history)} 个数据点")

            return True

        except Exception as e:
            print(f"✗ 加载权重失败: {e}")
            return False
