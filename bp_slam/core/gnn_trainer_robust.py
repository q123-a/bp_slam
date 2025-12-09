"""
bp_slam/core/gnn_trainer_robust.py
鲁棒版GNN训练器 - 带伪标签质量评估和自适应训练
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
from .gnn_model import FactorGraphNeuralNetwork

class GNNTrainerRobust:
    def __init__(self, device='cuda', lr=1e-3, hidden_dim=128, checkpoint_path=None, seed=42,
                 use_ema=True, ema_decay=0.999, use_lr_scheduler=True,
                 pseudo_label_mode='and', confidence_weighting=True,
                 use_temporal_gru=True, use_layer_gru=True,
                 quality_threshold=0.3, skip_low_quality=True):
        """
        初始化鲁棒版 GNN 训练器

        参数:
            ... (其他参数同GNNTrainerImproved)
            quality_threshold: 伪标签质量阈值 (0-1)，低于此值的样本不参与训练
            skip_low_quality: 是否跳过低质量时间步的训练
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
        self.quality_threshold = quality_threshold
        self.skip_low_quality = skip_low_quality

        # 设置随机种子
        if seed is not None:
            self._set_seed(seed)

        # 初始化模型
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
        self.quality_history = []  # 记录伪标签质量
        self.skipped_steps = []    # 记录跳过的步数

        # 跨帧记忆状态
        self.hidden_state = None

        # EMA 模型
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

        # 学习率调度器
        if use_lr_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=1000, eta_min=lr * 0.01
            )
            print(f"✓ 学习率调度器已启用")
        else:
            self.scheduler = None

        # 加载权重
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        print(f"✓ 鲁棒版GNN训练器初始化完成")
        print(f"  - 伪标签质量阈值: {quality_threshold}")
        print(f"  - 跳过低质量样本: {skip_low_quality}")

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

    def _update_ema(self):
        """更新 EMA 模型参数"""
        if self.ema_model is None:
            return
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)

    def reset_hidden_state(self):
        """清空 GRU 记忆"""
        self.hidden_state = None

    def _assess_pseudo_label_quality(self, measurements, predicted_measurements, predicted_variances):
        """
        评估伪标签质量

        返回:
            quality_score: 0-1之间的质量分数
            quality_info: 质量信息字典
        """
        z_meas = torch.from_numpy(measurements[0, :]).float().to(self.device)
        z_rss = torch.from_numpy(measurements[2, :]).float().to(self.device)
        z_pred = torch.from_numpy(predicted_measurements).float().to(self.device)
        var_meas = torch.from_numpy(measurements[1, :]).float().to(self.device)
        var_pred = torch.from_numpy(predicted_variances).float().to(self.device)

        # 1. 计算几何一致性
        joint_std = torch.sqrt(var_meas.unsqueeze(1) + var_pred.unsqueeze(0))
        diff_mat = z_meas.unsqueeze(1) - z_pred.unsqueeze(0)
        normalized_residuals = torch.abs(diff_mat) / (joint_std + 1e-6)
        min_residuals, min_idx = torch.min(normalized_residuals, dim=1)

        # 2. 计算幅度一致性
        P_tx = 15.41
        n = 2.0
        safe_pred_dist = torch.clamp(z_pred, min=0.1)
        rss_pred = P_tx - 10 * n * torch.log10(safe_pred_dist)
        rss_diff_mat = rss_pred.unsqueeze(0) - z_rss.unsqueeze(1)
        selected_rss_diff = torch.gather(rss_diff_mat, 1, min_idx.unsqueeze(1)).squeeze(1)

        # 3. 计算质量指标
        # 几何质量：残差越小越好
        geo_quality = torch.exp(-min_residuals).mean().item()

        # 幅度质量：差异越小越好
        phy_quality = torch.exp(-torch.abs(selected_rss_diff) / 5.0).mean().item()

        # 测量数量质量：测量越多越好（归一化到0-1）
        num_measurements = len(z_meas)
        meas_quality = min(num_measurements / 10.0, 1.0)

        # 综合质量分数（加权平均）
        quality_score = 0.4 * geo_quality + 0.3 * phy_quality + 0.3 * meas_quality

        quality_info = {
            'geo_quality': geo_quality,
            'phy_quality': phy_quality,
            'meas_quality': meas_quality,
            'num_measurements': num_measurements,
            'min_residual': min_residuals.mean().item()
        }

        return quality_score, quality_info

    def step(self, hybrid_tensor, measurements, predicted_measurements, predicted_variances, num_iterations=5):
        """
        执行一步训练/推理（带质量评估）

        返回:
            legacy_probs: (M, K) 锚点关联概率
            dustbin_probs: (M,) 杂波概率
            loss: 标量损失值（如果跳过训练则为0）
        """
        self.step_count += 1
        hybrid_tensor = hybrid_tensor.to(self.device)

        # [新增] 评估伪标签质量
        quality_score, quality_info = self._assess_pseudo_label_quality(
            measurements, predicted_measurements, predicted_variances
        )
        self.quality_history.append(quality_score)

        # [关键] 决定是否训练
        should_train = (not self.skip_low_quality) or (quality_score >= self.quality_threshold)

        if not should_train:
            self.skipped_steps.append(self.step_count)
            # 跳过训练，只做推理
            with torch.no_grad():
                inference_model = self.ema_model if self.use_ema else self.model
                inference_model.eval()
                eval_hidden = self.hidden_state.detach() if self.hidden_state is not None else None
                eval_logits, new_hidden_state = inference_model(hybrid_tensor, eval_hidden)

                # 更新hidden_state（即使不训练也要更新，保持一致性）
                self.hidden_state = new_hidden_state

                all_probs = F.softmax(eval_logits[0], dim=-1)
                legacy_probs = all_probs[:, :-1]
                dustbin_probs = all_probs[:, -1]

            return legacy_probs.cpu().numpy(), dustbin_probs.cpu().numpy(), 0.0

        # 正常训练流程
        if self.hidden_state is not None:
            h_in = self.hidden_state.detach()
        else:
            h_in = None

        total_loss = 0.0
        self.model.train()

        for iter_idx in range(num_iterations):
            logits, h_out = self.model(hybrid_tensor, h_in)

            loss = self._compute_improved_loss(
                logits, measurements, predicted_measurements, predicted_variances
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.05)
            self.optimizer.step()

            if self.use_ema:
                self._update_ema()

            total_loss += loss.item()
            h_in = h_out.detach() if h_out is not None else None

        self.hidden_state = h_out

        if self.scheduler is not None:
            self.scheduler.step()

        # 推理
        with torch.no_grad():
            inference_model = self.ema_model if self.use_ema else self.model
            inference_model.eval()
            eval_hidden = self.hidden_state.detach() if self.hidden_state is not None else None
            eval_logits, _ = inference_model(hybrid_tensor, eval_hidden)

            all_probs = F.softmax(eval_logits[0], dim=-1)
            legacy_probs = all_probs[:, :-1]
            dustbin_probs = all_probs[:, -1]

        avg_loss = total_loss / num_iterations
        self.loss_history.append(avg_loss)

        return legacy_probs.cpu().numpy(), dustbin_probs.cpu().numpy(), avg_loss

    def _compute_improved_loss(self, logits, measurements, predicted_measurements, predicted_variances):
        """改进的损失函数（同GNNTrainerImproved）"""
        z_meas = torch.from_numpy(measurements[0, :]).float().to(self.device)
        z_rss = torch.from_numpy(measurements[2, :]).float().to(self.device)
        z_pred = torch.from_numpy(predicted_measurements).float().to(self.device)
        var_meas = torch.from_numpy(measurements[1, :]).float().to(self.device)
        var_pred = torch.from_numpy(predicted_variances).float().to(self.device)

        joint_std = torch.sqrt(var_meas.unsqueeze(1) + var_pred.unsqueeze(0))
        diff_mat = z_meas.unsqueeze(1) - z_pred.unsqueeze(0)
        normalized_residuals = torch.abs(diff_mat) / (joint_std + 1e-6)
        min_residuals, min_idx = torch.min(normalized_residuals, dim=1)

        P_tx = 15.41
        n = 2.0
        safe_pred_dist = torch.clamp(z_pred, min=0.1)
        rss_pred = P_tx - 10 * n * torch.log10(safe_pred_dist)
        rss_diff_mat = rss_pred.unsqueeze(0) - z_rss.unsqueeze(1)
        selected_rss_diff = torch.gather(rss_diff_mat, 1, min_idx.unsqueeze(1)).squeeze(1)

        target_indices = torch.full((z_meas.shape[0],), predicted_measurements.shape[0],
                                    dtype=torch.long, device=self.device)

        is_geo_valid = (min_residuals < 1.5) | (torch.abs(torch.gather(diff_mat, 1, min_idx.unsqueeze(1)).squeeze(1)) < 0.5)
        is_phy_valid = selected_rss_diff < 6.0

        if self.pseudo_label_mode == 'and':
            valid_mask = is_geo_valid & is_phy_valid
        elif self.pseudo_label_mode == 'or':
            valid_mask = is_geo_valid | is_phy_valid
        else:
            valid_mask = is_geo_valid & is_phy_valid

        target_indices[valid_mask] = min_idx[valid_mask]

        if self.confidence_weighting:
            geo_confidence = torch.exp(-min_residuals)
            phy_confidence = torch.exp(-torch.abs(selected_rss_diff) / 5.0)
            confidence = (geo_confidence + phy_confidence) / 2.0

            sample_weights = torch.ones_like(confidence)
            sample_weights[valid_mask] = confidence[valid_mask]
            sample_weights = sample_weights / sample_weights.mean()

            loss_per_sample = F.cross_entropy(
                logits.view(z_meas.shape[0], -1),
                target_indices,
                reduction='none',
                label_smoothing=0.1
            )
            loss = (loss_per_sample * sample_weights).mean()
        else:
            loss = F.cross_entropy(
                logits.view(z_meas.shape[0], -1),
                target_indices,
                label_smoothing=0.1
            )

        return loss

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
            'quality_history': self.quality_history,
            'skipped_steps': self.skipped_steps,
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
        print(f"  - 跳过的步数: {len(self.skipped_steps)}")

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
            self.quality_history = checkpoint.get('quality_history', [])
            self.skipped_steps = checkpoint.get('skipped_steps', [])

            print(f"✓ GNN 权重已加载: {checkpoint_path}")
            return True

        except Exception as e:
            print(f"✗ 加载权重失败: {e}")
            return False
