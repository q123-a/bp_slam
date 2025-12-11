"""
bp_slam/core/gnn_trainer_improved.py
改进的自监督训练器 - 增强稳定性版本 + 图注意力 + 自适应RANSAC
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
from .gnn_model import FactorGraphNeuralNetwork
from .graph_attention import AdaptiveRANSAC

class GNNTrainerImproved:
    def __init__(self, device='cuda', lr=1e-3, hidden_dim=64, checkpoint_path=None, seed=42,
                 use_ema=True, ema_decay=0.999, use_lr_scheduler=True,
                 pseudo_label_mode='and', confidence_weighting=True,
                 use_temporal_gru=False, use_layer_gru=False,
                 use_graph_attention=False, use_ransac=False, ransac_threshold=2.0):
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
            use_graph_attention: 是否使用图注意力模块 (默认False)
            use_ransac: 是否使用自适应RANSAC后处理 (默认False)
            ransac_threshold: RANSAC距离阈值 (默认2.0米)
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
        self.use_graph_attention = use_graph_attention
        self.use_ransac = use_ransac

        # 设置随机种子以确保可复现性
        if seed is not None:
            self._set_seed(seed)

        # input_dim=5 (混合特征: LogProb, Residual, Variance, Existence, Amplitude)
        # [关键修改] 默认关闭GRU以避免错误信息的时间传播
        self.model = FactorGraphNeuralNetwork(
            input_dim=5,
            hidden_dim=hidden_dim,
            use_temporal_gru=use_temporal_gru,
            use_layer_gru=use_layer_gru,
            use_graph_attention=use_graph_attention
        ).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.step_count = 0

        # Loss 历史记录
        self.loss_history = []
        self.detailed_loss_history = []  # [新增] 详细记录: [(step, sensor, loss), ...]

        # [新增] 跨帧记忆状态
        self.hidden_state = None

        # [新增] EMA 模型
        if use_ema:
            self.ema_model = FactorGraphNeuralNetwork(
                input_dim=5,
                hidden_dim=hidden_dim,
                use_temporal_gru=use_temporal_gru,
                use_layer_gru=use_layer_gru,
                use_graph_attention=use_graph_attention
            ).to(device)
            self.ema_model.load_state_dict(self.model.state_dict())
            for param in self.ema_model.parameters():
                param.requires_grad = False
            print(f"✓ EMA 已启用 (decay={ema_decay})")
        else:
            self.ema_model = None

        # [新增] RANSAC 后处理模块
        if use_ransac:
            self.ransac = AdaptiveRANSAC(
                distance_threshold=ransac_threshold,
                enable_adaptive=True
            )
            print(f"✓ 自适应RANSAC已启用 (threshold={ransac_threshold}m)")
        else:
            self.ransac = None

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
        print(f"  - 图注意力: {use_graph_attention}")
        print(f"  - 自适应RANSAC: {use_ransac}")

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
            logits, h_out, attention_info = self.model(hybrid_tensor, h_in)  # (1, M, K+1), (1, hidden_dim), dict

            # 2. 计算自监督 Loss (改进版)
            # 只在第一次迭代时输出和保存伪标签统计
            loss = self._compute_improved_loss(
                logits, measurements, predicted_measurements, predicted_variances,
                attention_info=attention_info,
                log_stats=(iter_idx == 0)
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
            eval_logits, _, eval_attention_info = inference_model(hybrid_tensor, eval_hidden)

            # Softmax 归一化
            all_probs = F.softmax(eval_logits[0], dim=-1)  # (M, K+1)

            # 分离 Legacy 和 Dustbin
            legacy_probs = all_probs[:, :-1]  # (M, K)
            dustbin_probs = all_probs[:, -1]   # (M,)

            # [新增] RANSAC 后处理（可选）
            if self.ransac is not None and self.use_ransac:
                z_meas = torch.from_numpy(measurements[0, :]).float().to(self.device)
                z_pred = torch.from_numpy(predicted_measurements).float().to(self.device)

                # 提取杂波分数（如果有图注意力）
                clutter_scores = eval_attention_info.get('clutter_scores', None)
                if clutter_scores is not None:
                    clutter_scores = clutter_scores[0]  # (M,)

                # 执行RANSAC过滤
                inlier_mask, ransac_info = self.ransac.filter_measurements(
                    z_meas, z_pred,
                    clutter_scores=clutter_scores
                )

                # 将杂波测量的dustbin概率设为1.0
                dustbin_probs = dustbin_probs.clone()
                dustbin_probs[~inlier_mask] = 1.0

                # 重新归一化
                all_probs_adjusted = torch.cat([legacy_probs, dustbin_probs.unsqueeze(1)], dim=1)
                all_probs_adjusted = all_probs_adjusted / all_probs_adjusted.sum(dim=1, keepdim=True)
                legacy_probs = all_probs_adjusted[:, :-1]
                dustbin_probs = all_probs_adjusted[:, -1]

                # 输出RANSAC统计信息（仅第一次迭代）
                if ransac_info is not None and self.step_count % 10 == 1:
                    clutter_ratio = ransac_info.get('clutter_ratio', 0)
                    num_inliers = inlier_mask.sum().item()
                    num_total = len(inlier_mask)
                    print(f"  [RANSAC] Step {self.step_count}: Inliers={num_inliers}/{num_total}, "
                          f"Clutter={clutter_ratio*100:.1f}%, Method={ransac_info.get('method', 'ransac')}")

        # 返回平均损失
        avg_loss = total_loss / num_iterations
        self.loss_history.append(avg_loss)

        return legacy_probs.cpu().numpy(), dustbin_probs.cpu().numpy(), avg_loss

    def save_loss_history(self, filepath):
        """保存 loss 历史到文件"""
        import json
        loss_data = {
            'loss_history': self.loss_history,
            'detailed_loss_history': self.detailed_loss_history
        }
        with open(filepath, 'w') as f:
            json.dump(loss_data, f, indent=2)
        print(f"✓ Loss 历史已保存到: {filepath}")

    def save_pseudo_label_history(self, filepath):
        """保存正负样本统计到单独的文件"""
        import json
        from pathlib import Path

        # 确保目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        pseudo_label_data = {
            'pseudo_label_history': getattr(self, 'pseudo_label_history', []),
            'summary': {}
        }

        # 计算统计摘要
        if hasattr(self, 'pseudo_label_history') and self.pseudo_label_history:
            pos_ratios = [item['positive_ratio'] for item in self.pseudo_label_history]
            neg_ratios = [item['negative_ratio'] for item in self.pseudo_label_history]

            pseudo_label_data['summary'] = {
                'total_steps': len(self.pseudo_label_history),
                'positive_ratio': {
                    'mean': sum(pos_ratios) / len(pos_ratios),
                    'min': min(pos_ratios),
                    'max': max(pos_ratios)
                },
                'negative_ratio': {
                    'mean': sum(neg_ratios) / len(neg_ratios),
                    'min': min(neg_ratios),
                    'max': max(neg_ratios)
                }
            }

        with open(filepath, 'w') as f:
            json.dump(pseudo_label_data, f, indent=2)

        print(f"✓ 正负样本统计已保存到: {filepath}")

        # 打印统计摘要
        if pseudo_label_data['summary']:
            summary = pseudo_label_data['summary']
            print(f"  正样本比例: 平均 {summary['positive_ratio']['mean']:.1f}%, "
                  f"最小 {summary['positive_ratio']['min']:.1f}%, "
                  f"最大 {summary['positive_ratio']['max']:.1f}%")
            print(f"  负样本比例: 平均 {summary['negative_ratio']['mean']:.1f}%, "
                  f"最小 {summary['negative_ratio']['min']:.1f}%, "
                  f"最大 {summary['negative_ratio']['max']:.1f}%")

    def _compute_improved_loss(self, logits, measurements, predicted_measurements, predicted_variances, attention_info=None, log_stats=False):
        """
        改进的损失函数：支持多种伪标签生成策略和置信度加权

        参数:
            logits: (1, M, K+1) 模型输出
            measurements: (3, M) 测量数据
            predicted_measurements: (K,) 预测测量
            predicted_variances: (K,) 预测方差
            log_stats: 是否输出和保存伪标签统计（默认False，只在第一次迭代时为True）

        返回:
            loss: 标量损失值
        """
        # 准备数据（几何 + 幅度）
        z_meas = torch.from_numpy(measurements[0, :]).float().to(self.device)  # (M,) 距离
        z_pred = torch.from_numpy(predicted_measurements).float().to(self.device) # (K,)
        var_meas = torch.from_numpy(measurements[1, :]).float().to(self.device)
        var_pred = torch.from_numpy(predicted_variances).float().to(self.device)

        # 1. 计算距离矩阵和归一化残差
        diff_mat = z_meas.unsqueeze(1) - z_pred.unsqueeze(0)  # (M, K) 测量 - 预测
        abs_dist_mat = torch.abs(diff_mat)  # (M, K) 绝对距离差

        joint_std = torch.sqrt(var_meas.unsqueeze(1) + var_pred.unsqueeze(0))
        normalized_residuals = abs_dist_mat / (joint_std + 1e-6)

        # 找到每个测量的最佳匹配锚点
        min_abs_dist, min_idx = torch.min(abs_dist_mat, dim=1)  # (M,)
        min_norm_res = torch.gather(normalized_residuals, 1, min_idx.unsqueeze(1)).squeeze(1)  # (M,)

        # 2. 计算幅度一致性（RSS）和信号质量
        has_amplitude = measurements.shape[0] >= 3
        if has_amplitude:
            # 测量幅度（线性）转换为 dB
            z_rss_linear = torch.from_numpy(measurements[2, :]).float().to(self.device)
            z_rss_power = z_rss_linear ** 2
            z_rss_power = torch.clamp(z_rss_power, min=1e-10)
            z_rss_db = 10 * torch.log10(z_rss_power)  # (M,) dB

            # 预测幅度（基于 Friis 公式）
            P_tx = 15.41  # 校准后的发射功率 (dBm)
            n = 2.0
            z_pred_safe = torch.clamp(z_pred, min=0.1)
            rss_pred_db = P_tx - 10 * n * torch.log10(z_pred_safe)  # (K,) dB

            # 计算 RSS 差异矩阵
            rss_diff_mat = torch.abs(z_rss_db.unsqueeze(1) - rss_pred_db.unsqueeze(0))  # (M, K)

            # 对于几何最近的锚点，检查 RSS 是否一致
            rss_diff_min = torch.gather(rss_diff_mat, 1, min_idx.unsqueeze(1)).squeeze(1)  # (M,)
            is_rss_valid = rss_diff_min < 8.0  # RSS 差异 < 8dB

            # [新增] 基于信号质量的动态距离门限
            # 信号强 → 门限严格；信号弱 → 门限宽松
            rss_min = z_rss_linear.min()
            rss_max = z_rss_linear.max()
            quality_score = (z_rss_linear - rss_min) / (rss_max - rss_min + 1e-6)  # (M,) 0~1

            # 动态门限：信号好时0.5m，信号差时3.0m
            TH_STRICT = 0.5
            TH_LOOSE = 3.0
            dynamic_dist_th = TH_LOOSE - (TH_LOOSE - TH_STRICT) * quality_score  # (M,)
        else:
            is_rss_valid = torch.ones(z_meas.shape[0], dtype=torch.bool, device=self.device)
            # 没有幅度信息时，使用基于方差的动态门限
            dynamic_dist_th = 2.0 * torch.sqrt(var_meas)  # (M,)
            dynamic_dist_th = torch.clamp(dynamic_dist_th, min=0.6, max=2.0)

        # 3. 生成伪标签（双重判定：统计 + 物理）
        target_indices = torch.full((z_meas.shape[0],), predicted_measurements.shape[0],
                                    dtype=torch.long, device=self.device)

        # 统计一致性：归一化残差 < 3σ (99.7%置信度)
        is_stat_valid = min_norm_res < 3.0

        # 物理距离：绝对距离在动态门限内
        is_phy_valid = min_abs_dist < dynamic_dist_th

        # 几何条件：统计 AND 物理都满足
        is_geo_valid = is_stat_valid & is_phy_valid

        # 根据伪标签模式组合条件
        if self.pseudo_label_mode == 'and':
            # AND 模式：几何 AND 幅度都满足
            valid_mask = is_geo_valid & is_rss_valid
        elif self.pseudo_label_mode == 'or':
            # OR 模式：几何 OR 幅度满足其一
            valid_mask = is_geo_valid | is_rss_valid
        elif self.pseudo_label_mode == 'adaptive':
            # Adaptive 模式：优先使用 AND，如果正样本太少则降级为 OR
            and_mask = is_geo_valid & is_rss_valid
            if and_mask.sum() < z_meas.shape[0] * 0.3:  # 正样本少于 30%
                valid_mask = is_geo_valid | is_rss_valid
            else:
                valid_mask = and_mask
        else:
            raise ValueError(f"Unknown pseudo_label_mode: {self.pseudo_label_mode}")

        target_indices[valid_mask] = min_idx[valid_mask]

        # [新增] 统计正负样本数量 - 只在第一次迭代时输出并保存
        if log_stats:
            num_positives = valid_mask.sum().item()
            num_negatives = (~valid_mask).sum().item()
            num_measurements = z_meas.shape[0]
            positive_ratio = num_positives / num_measurements * 100 if num_measurements > 0 else 0
            negative_ratio = num_negatives / num_measurements * 100 if num_measurements > 0 else 0

            # 输出到控制台
            print(f"  [Pseudo-label] Step {self.step_count}: Pos={num_positives}({positive_ratio:.1f}%), Neg={num_negatives}({negative_ratio:.1f}%), Total={num_measurements}")

            # 保存到详细历史（用于后续分析）
            if not hasattr(self, 'pseudo_label_history'):
                self.pseudo_label_history = []
            self.pseudo_label_history.append({
                'step': self.step_count,
                'num_positives': num_positives,
                'num_negatives': num_negatives,
                'num_measurements': num_measurements,
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio
            })

        # 4. 置信度加权 - 几何 + 幅度
        if self.confidence_weighting:
            # 几何置信度
            geo_confidence = torch.exp(-min_norm_res)

            # 幅度置信度
            if has_amplitude:
                rss_confidence = torch.exp(-rss_diff_min / 8.0)  # 归一化到 8dB
                # 综合置信度（几何权重 0.6，幅度权重 0.4）
                confidence = 0.6 * geo_confidence + 0.4 * rss_confidence
            else:
                confidence = geo_confidence

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
