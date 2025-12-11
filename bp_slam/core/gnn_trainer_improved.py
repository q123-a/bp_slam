"""
bp_slam/core/gnn_trainer_improved.py
改进的自监督训练器 - 增强稳定性版本
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from .gnn_model import FactorGraphNeuralNetwork

class GNNTrainerImproved:
    def __init__(self, device='cuda', lr=1e-3, hidden_dim=64, checkpoint_path=None, seed=42,
                 use_ema=True, ema_decay=0.999, use_lr_scheduler=True,
                 pseudo_label_mode='and', confidence_weighting=True,
                 use_temporal_gru=False, use_layer_gru=False,
                 rejection_threshold=3.0, positive_weight=5.0):
        """
        初始化改进的 GNN 训练器

        参数:
            device: 设备 ('cuda' 或 'cpu')
            lr: 初始学习率
            hidden_dim: 隐藏层维度 (默认 64)
            checkpoint_path: 权重文件路径 (如果提供，则加载预训练权重)
            seed: 随机种子 (默认 42)，设为 None 则不固定种子
            use_ema: 是否使用指数移动平均 (EMA) 平滑参数
            ema_decay: EMA 衰减率 (默认 0.999)
            use_lr_scheduler: 是否使用学习率调度器
            pseudo_label_mode: 伪标签生成模式 ('and', 'or', 'adaptive')
            confidence_weighting: 是否使用置信度加权损失
            use_temporal_gru: 是否使用跨帧GRU记忆 (默认False，避免错误传播)
            use_layer_gru: 是否使用层内GRU更新 (默认False，避免错误传播)
        """
        self.device = device
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.seed = seed
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.use_lr_scheduler = use_lr_scheduler
        self.pseudo_label_mode = pseudo_label_mode
        self.confidence_weighting = confidence_weighting
        self.use_temporal_gru = use_temporal_gru
        self.use_layer_gru = use_layer_gru
        self.rejection_threshold = rejection_threshold
        self.positive_weight = positive_weight

        # 设置随机种子以确保可复现性
        if seed is not None:
            self._set_seed(seed)

        # input_dim=5 (混合特征: LogProb, Residual, Variance, Existence, Amplitude)
        # [关键修改] 默认关闭GRU以避免错误信息的时间传播
        self.model = FactorGraphNeuralNetwork(
            input_dim=5,
            hidden_dim=hidden_dim,
            use_temporal_gru=use_temporal_gru,
            use_layer_gru=use_layer_gru
        ).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.step_count = 0

        # Loss 历史记录
        self.loss_history = []

        # [新增] 跨帧记忆状态
        self.hidden_state = None

        # [新增] EMA 模型
        if use_ema:
            self.ema_model = FactorGraphNeuralNetwork(
                input_dim=5,
                hidden_dim=hidden_dim,
                use_temporal_gru=use_temporal_gru,
                use_layer_gru=use_layer_gru
            ).to(device)
            self.ema_model.load_state_dict(self.model.state_dict())
            for param in self.ema_model.parameters():
                param.requires_grad = False
            print(f"✓ EMA 已启用 (decay={ema_decay})")
        else:
            self.ema_model = None

        # [新增] 学习率调度器 - 余弦退火
        if use_lr_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=1000, eta_min=lr * 0.01
            )
            print(f"✓ 学习率调度器已启用 (CosineAnnealing)")
        else:
            self.scheduler = None

        # 如果提供了权重文件，则加载
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        print(f"✓ GNN 训练器初始化完成")
        print(f"  - 损失函数: 匈牙利算法全局最优匹配")
        print(f"  - 熔断阈值: {self.rejection_threshold} (几何+物理综合代价)")
        print(f"  - 样本权重: 正样本 {self.positive_weight} / 杂波 1.0")

    def _set_seed(self, seed):
        """设置所有随机种子以确保可复现性"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f"✓ 随机种子已设置: {seed}")

    def _update_ema(self):
        """更新 EMA 模型参数"""
        if self.ema_model is None:
            return

        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)

    def reset_hidden_state(self):
        """在新的序列开始时调用，清空 GRU 记忆"""
        self.hidden_state = None

    def step(self, hybrid_tensor, measurements, predicted_measurements, predicted_variances, num_iterations=5):
        """
        执行一步训练/推理（支持多次迭代）

        参数:
            hybrid_tensor: (1, M, K+1, 5) 混合特征张量
            measurements: (3, M) 测量数据 [距离, 方差, 幅度]
            predicted_measurements: (K,) 预测测量
            predicted_variances: (K,) 预测方差
            num_iterations: int, 每个时间步的迭代次数 (默认5次)

        返回:
            legacy_probs: (M, K) 锚点关联概率
            dustbin_probs: (M,) 杂波概率
            loss: 标量损失值
        """
        self.step_count += 1
        hybrid_tensor = hybrid_tensor.to(self.device)

        # 处理 hidden_state 的梯度截断
        if self.hidden_state is not None:
            h_in = self.hidden_state.detach()
        else:
            h_in = None

        # 多次迭代训练
        total_loss = 0.0
        self.model.train()

        for iter_idx in range(num_iterations):
            # 1. 前向推理
            logits, h_out = self.model(hybrid_tensor, h_in)  # (1, M, K+1), (1, hidden_dim)

            # 2. 计算自监督 Loss (通用匹配版 - 匈牙利算法)
            loss = self._compute_universal_matching_loss(
                logits, measurements, predicted_measurements, predicted_variances, hybrid_tensor
            )

            # 3. 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            # [改进] 更强的梯度裁剪 (从0.1降到0.05，防止参数震荡)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.05)
            self.optimizer.step()

            # [新增] 更新 EMA
            if self.use_ema:
                self._update_ema()

            total_loss += loss.item()

            # 在迭代过程中更新 h_in
            h_in = h_out.detach() if h_out is not None else None

        # 更新内部记忆
        self.hidden_state = h_out

        # [新增] 学习率调度
        if self.scheduler is not None:
            self.scheduler.step()

        # 4. 推理模式：使用 EMA 模型（如果启用）
        with torch.no_grad():
            inference_model = self.ema_model if self.use_ema else self.model
            inference_model.eval()
            # 处理 hidden_state 为 None 的情况
            eval_hidden = self.hidden_state.detach() if self.hidden_state is not None else None
            eval_logits, _ = inference_model(hybrid_tensor, eval_hidden)

            # Softmax 归一化
            all_probs = F.softmax(eval_logits[0], dim=-1)  # (M, K+1)

            # 分离 Legacy 和 Dustbin
            legacy_probs = all_probs[:, :-1]  # (M, K)
            dustbin_probs = all_probs[:, -1]   # (M,)

        # 返回平均损失
        avg_loss = total_loss / num_iterations
        self.loss_history.append(avg_loss)

        return legacy_probs.cpu().numpy(), dustbin_probs.cpu().numpy(), avg_loss

    def _compute_improved_loss(self, logits, measurements, predicted_measurements, predicted_variances):
        """
        改进的损失函数：支持多种伪标签生成策略和置信度加权

        参数:
            logits: (1, M, K+1) 模型输出
            measurements: (3, M) 测量数据
            predicted_measurements: (K,) 预测测量
            predicted_variances: (K,) 预测方差

        返回:
            loss: 标量损失值
        """
        # 准备数据（只使用几何信息，不使用幅度）
        z_meas = torch.from_numpy(measurements[0, :]).float().to(self.device)  # (M,) 距离
        z_pred = torch.from_numpy(predicted_measurements).float().to(self.device) # (K,)
        var_meas = torch.from_numpy(measurements[1, :]).float().to(self.device)
        var_pred = torch.from_numpy(predicted_variances).float().to(self.device)

        # 1. 计算几何一致性
        joint_std = torch.sqrt(var_meas.unsqueeze(1) + var_pred.unsqueeze(0))
        diff_mat = z_meas.unsqueeze(1) - z_pred.unsqueeze(0)
        normalized_residuals = torch.abs(diff_mat) / (joint_std + 1e-6)

        # 找到几何上最近的锚点
        min_residuals, min_idx = torch.min(normalized_residuals, dim=1)

        # 2. 生成伪标签（只使用几何条件）
        target_indices = torch.full((z_meas.shape[0],), predicted_measurements.shape[0],
                                    dtype=torch.long, device=self.device)

        # [修改] 只使用几何条件，忽略幅度信息
        # 2σ (95%置信度) 或 1m 绝对距离差
        is_geo_valid = (min_residuals < 2.0) | (torch.abs(torch.gather(diff_mat, 1, min_idx.unsqueeze(1)).squeeze(1)) < 1.0)

        # 直接使用几何条件作为有效性判断
        valid_mask = is_geo_valid

        # 保留模式选择逻辑的框架（但现在只用几何）
        if self.pseudo_label_mode == 'and':
            valid_mask = is_geo_valid  # 只有几何条件
        elif self.pseudo_label_mode == 'or':
            valid_mask = is_geo_valid  # 只有几何条件
        elif self.pseudo_label_mode == 'adaptive':
            # adaptive模式现在也只使用几何条件
            valid_mask = is_geo_valid
        else:
            raise ValueError(f"Unknown pseudo_label_mode: {self.pseudo_label_mode}")

        target_indices[valid_mask] = min_idx[valid_mask]

        # 4. [修改] 置信度加权 - 只使用几何置信度
        if self.confidence_weighting:
            # 只使用几何置信度
            confidence = torch.exp(-min_residuals)

            # 对正样本和负样本分别加权
            sample_weights = torch.ones_like(confidence)
            sample_weights[valid_mask] = confidence[valid_mask]  # 正样本：高置信度高权重
            sample_weights[~valid_mask] = 1.0  # 负样本：固定权重

            # 归一化权重
            sample_weights = sample_weights / sample_weights.mean()

            # 计算加权损失
            loss_per_sample = F.cross_entropy(
                logits.view(z_meas.shape[0], -1),
                target_indices,
                reduction='none',
                label_smoothing=0.1
            )
            loss = (loss_per_sample * sample_weights).mean()
        else:
            # 标准损失
            loss = F.cross_entropy(
                logits.view(z_meas.shape[0], -1),
                target_indices,
                label_smoothing=0.1
            )

        return loss

    def _compute_universal_matching_loss(self, logits, measurements, predicted_measurements, predicted_variances, hybrid_tensor):
        """
        Universal Physics-Aware Matching Loss
        通用物理感知匹配损失：适用于无杂波和有杂波环境。

        参数:
            logits: (1, M, K+1) 模型输出
            measurements: (3, M) 测量数据 [距离, 方差, 幅度]
            predicted_measurements: (K,) 预测测量
            predicted_variances: (K,) 预测方差
            hybrid_tensor: (1, M, K+1, 5) 混合特征张量

        返回:
            loss: 标量损失值
        """
        M = measurements.shape[1]
        K = predicted_measurements.shape[0]

        # ------------------------------------------------------
        # 1. 准备数据
        # ------------------------------------------------------
        z_geo = torch.from_numpy(measurements[0, :]).float().to(self.device)   # (M,) 距离
        z_rss = torch.from_numpy(measurements[2, :]).float().to(self.device)   # (M,) 幅度

        z_pred_geo = torch.from_numpy(predicted_measurements).float().to(self.device) # (K,)
        var_meas = torch.from_numpy(measurements[1, :]).float().to(self.device)
        var_pred = torch.from_numpy(predicted_variances).float().to(self.device)

        # ------------------------------------------------------
        # 2. 构建"几何+物理"综合代价矩阵
        # ------------------------------------------------------

        # A. 几何代价 (Mahalanobis Distance)
        # 衡量空间位置的匹配度
        joint_std = torch.sqrt(var_meas.unsqueeze(1) + var_pred.unsqueeze(0))
        diff_geo = z_geo.unsqueeze(1) - z_pred_geo.unsqueeze(0)
        cost_geo = torch.abs(diff_geo) / (joint_std + 1e-6) # (M, K)

        # B. 物理代价 (Amplitude/RSS Compatibility)
        # 衡量信号特征的匹配度 (从 hybrid_tensor 提取或直接计算)
        # 假设 hybrid_tensor 第4维是归一化的 RSS 残差
        cost_phy = torch.abs(hybrid_tensor[0, :, :K, 4]) # (M, K)

        # C. 动态加权 (可选，进阶)
        # 如果信号强，几何权重大；如果信号弱，几何权重小
        # [关键修改] 在失配模式下，降低几何权重，提高物理权重
        # 因为几何代价依赖predicted_measurements，而失配模式下预测可能不准
        # 物理代价（幅度）是直接测量，不受预测影响
        if self.rejection_threshold < 2.5:  # 失配模式的标志
            w_geo = 0.5  # 降低几何权重
            w_phy = 2.0  # 提高物理权重
        else:
            w_geo = 1.0
            w_phy = 1.5

        # D. 总代价矩阵
        total_cost_matrix = w_geo * cost_geo + w_phy * cost_phy

        # ------------------------------------------------------
        # 3. 匈牙利算法 (Global Assignment)
        # ------------------------------------------------------
        # 即使 M > K (有杂波)，它也会选出 Top-K 个"嫌疑人"
        # 即使 M < K (漏检)，它也会尽力匹配
        cost_np = total_cost_matrix.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)

        # ------------------------------------------------------
        # 4. 熔断机制 (Gating / Rejection) - 关键！
        # ------------------------------------------------------
        # 初始化所有目标为 "垃圾桶/杂波" (索引 K)
        target_indices = torch.full((M,), K, dtype=torch.long, device=self.device)

        # 定义熔断门槛 (Cost Threshold)
        # 只有 Cost 小于此值的匹配，才被承认
        # 使用可配置的阈值，失配模式下会更严格
        REJECTION_THRESHOLD = self.rejection_threshold

        for i, match_idx in enumerate(row_ind):
            anchor_idx = col_ind[i]
            match_cost = total_cost_matrix[match_idx, anchor_idx]

            # [通用逻辑核心]
            # 无杂波时：真值 Cost 比如 0.5 < threshold -> 匹配成功 (Target = anchor_idx)
            # 有杂波时：杂波 Cost 比如 5.1 > threshold -> 熔断拒绝 (Target 保持为 K)
            if match_cost < REJECTION_THRESHOLD:
                target_indices[match_idx] = anchor_idx

        # ------------------------------------------------------
        # 5. 计算 Loss (包含垃圾桶列)
        # ------------------------------------------------------
        # 既然要处理杂波，必须允许网络输出第 K+1 列 (索引 K)
        logits_all = logits.view(M, K+1)

        # 加权 Loss (可选)：给正样本更高权重，防止被大量杂波淹没
        weights = torch.ones(M, device=self.device)
        # 找到正样本 (不是杂波的)
        pos_mask = (target_indices != K)

        # [调试] 统计正负样本比例
        num_positive = pos_mask.sum().item()
        num_negative = M - num_positive

        if pos_mask.sum() > 0:
            weights[pos_mask] = self.positive_weight # 强迫网络关注那几个真匹配

        # [调试] 每100步打印一次统计信息
        if hasattr(self, 'step_count') and self.step_count % 100 == 0:
            print(f"\n[Step {self.step_count}] 样本统计:")
            print(f"  正样本: {num_positive}/{M} ({num_positive/M*100:.1f}%)")
            print(f"  负样本: {num_negative}/{M} ({num_negative/M*100:.1f}%)")
            print(f"  失衡比例: 1:{num_negative/max(num_positive,1):.2f}")

        loss = F.cross_entropy(logits_all, target_indices, reduction='none', label_smoothing=0.1)
        loss = (loss * weights).mean()

        return loss

    def save_checkpoint(self, checkpoint_path, epoch=None, additional_info=None):
        """保存模型权重和训练状态"""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'hidden_dim': self.hidden_dim,
            'lr': self.lr,
            'loss_history': self.loss_history,
            'pseudo_label_mode': self.pseudo_label_mode,
            'confidence_weighting': self.confidence_weighting,
        }

        if self.ema_model is not None:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if epoch is not None:
            checkpoint['epoch'] = epoch

        if additional_info is not None:
            checkpoint.update(additional_info)

        torch.save(checkpoint, checkpoint_path)
        print(f"✓ GNN 权重已保存: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """加载模型权重和训练状态"""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            print(f"⚠ 权重文件不存在: {checkpoint_path}")
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # 加载模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # 加载 EMA 模型
            if self.ema_model is not None and 'ema_model_state_dict' in checkpoint:
                self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

            # 加载优化器状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 加载调度器状态
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # 加载训练步数
            self.step_count = checkpoint.get('step_count', 0)

            # 加载 Loss 历史
            self.loss_history = checkpoint.get('loss_history', [])

            print(f"✓ GNN 权重已加载: {checkpoint_path}")
            print(f"  - 训练步数: {self.step_count}")
            print(f"  - Loss 历史记录: {len(self.loss_history)} 个数据点")

            return True

        except Exception as e:
            print(f"✗ 加载权重失败: {e}")
            return False
