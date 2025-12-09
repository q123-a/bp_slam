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
from .gnn_model import FactorGraphNeuralNetwork

class GNNTrainerImproved:
    def __init__(self, device='cuda', lr=1e-3, hidden_dim=64, checkpoint_path=None, seed=42,
                 use_ema=True, ema_decay=0.999, use_lr_scheduler=True,
                 pseudo_label_mode='and', confidence_weighting=True,
                 use_temporal_gru=False, use_layer_gru=False):
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
        print(f"  - 伪标签模式: {pseudo_label_mode}")
        print(f"  - 置信度加权: {confidence_weighting}")

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

            # 2. 计算自监督 Loss (改进版)
            loss = self._compute_improved_loss(
                logits, measurements, predicted_measurements, predicted_variances
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
