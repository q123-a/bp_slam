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
            return False


# ==============================================================================
# 双头联合训练器 (Joint Dual-Head Trainer)
# ==============================================================================

from .gnn_model import JointDualHeadGNN

class JointDualHeadTrainer:
    """
    双头联合训练器：质量头 + 关联头的自监督训练
    
    训练策略：
    1. 物理老师（Physics Teacher）：监督质量头，使用RSS内在一致性
    2. 几何老师（Geometry Teacher）：监督关联头，使用匈牙利算法
    
    关键优势：
    - 质量头不依赖预测位置，只看物理特征
    - 即使关联头被错误预测误导，质量头仍能正确识别杂波
    - 两个头互不干扰，但共享底层特征
    """
    
    def __init__(self, device='cuda', lr=1e-3, hidden_dim=64, checkpoint_path=None, seed=42,
                 use_ema=True, ema_decay=0.999, use_lr_scheduler=True,
                 use_temporal_gru=False, use_layer_gru=False,
                 quality_threshold=0.5, assoc_threshold=3.0, 
                 quality_weight=1.0, assoc_weight=2.0):
        """
        初始化双头训练器
        
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
            quality_weight: 质量损失权重 (1.0)
            assoc_weight: 关联损失权重 (2.0，因为更难学)
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
        
        # 设置随机种子
        if seed is not None:
            self._set_seed(seed)
        
        # 初始化模型
        self.model = JointDualHeadGNN(
            input_dim=5,
            hidden_dim=hidden_dim,
            num_layers=2,
            use_temporal_gru=use_temporal_gru,
            use_layer_gru=use_layer_gru
        ).to(device)
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # EMA模型（可选）
        if use_ema:
            self.ema_model = JointDualHeadGNN(
                input_dim=5,
                hidden_dim=hidden_dim,
                num_layers=2,
                use_temporal_gru=use_temporal_gru,
                use_layer_gru=use_layer_gru
            ).to(device)
            self.ema_model.load_state_dict(self.model.state_dict())
            for param in self.ema_model.parameters():
                param.requires_grad = False
        else:
            self.ema_model = None
        
        # 学习率调度器
        if use_lr_scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=300, gamma=0.5
            )
        else:
            self.scheduler = None
        
        # 训练状态
        self.step_count = 0
        self.loss_history = []
        self.quality_loss_history = []
        self.assoc_loss_history = []

        # [新增] GRU隐藏状态（在trainer内部维护）
        self.hidden_state = None

        # 加载权重（如果提供）
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        
        print(f"✓ 双头GNN训练器初始化完成")
        print(f"  - 质量阈值: {quality_threshold}")
        print(f"  - 关联阈值: {assoc_threshold}")
        print(f"  - 损失权重: 质量={quality_weight}, 关联={assoc_weight}")
    
    def _set_seed(self, seed):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def reset_hidden_state(self):
        """
        重置GRU隐藏状态

        应在以下情况调用：
        1. 开始新的轨迹序列
        2. 检测到轨迹中断
        3. 测量数量或锚点数量发生显著变化
        """
        self.hidden_state = None
    
    def step(self, hybrid_tensor, measurements, predicted_measurements, predicted_variances):
        """
        执行一步训练/推理

        改进：
        1. GRU状态在trainer内部维护（不需要外部传递）
        2. 返回dustbin_probs而不是quality_scores（更符合SLAM接口）
        3. 梯度截断防止BPTT过长

        输入:
            hybrid_tensor: (1, M, K+1, 5) 混合特征
            measurements: (3, M) 测量数据 [距离, 方差, 幅度]
            predicted_measurements: (K,) 预测距离
            predicted_variances: (K,) 预测方差

        输出:
            assoc_probs: (M, K) 关联概率
            dustbin_probs: (M,) 杂波概率 [0, 1]
            loss: 标量损失值
        """
        self.model.train()

        # 将输入移到设备
        hybrid_tensor = hybrid_tensor.to(self.device)

        # [关键改进] 梯度截断：防止BPTT过长导致梯度爆炸
        # 使用detach()切断梯度流，只保留数值
        #
        # [修复] 检查hidden state维度是否匹配
        # 问题：每一帧的M和K都在变化，导致hidden_state维度不匹配
        # 解决：如果维度不匹配，重置hidden_state为None
        h_in = None
        if self.hidden_state is not None:
            # 计算当前帧的期望维度：(B*M*K, hidden_dim)
            B, M, K_plus_1, _ = hybrid_tensor.shape
            K = K_plus_1 - 1  # 去掉垃圾桶列
            expected_size = B * M * K

            # 检查维度是否匹配
            if self.hidden_state.shape[0] == expected_size:
                h_in = self.hidden_state.detach()
            else:
                # 维度不匹配，重置为None（让模型自动初始化）
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
        
        # 更新学习率
        if self.scheduler is not None:
            self.scheduler.step()
        
        # 记录
        self.step_count += 1
        self.loss_history.append(loss.item())
        self.quality_loss_history.append(quality_loss.item())
        self.assoc_loss_history.append(assoc_loss.item())
        
        # [关键改进] 格式化输出供SLAM使用
        with torch.no_grad():
            # 关联概率: Softmax (M, K)
            assoc_probs = F.softmax(assoc_logits, dim=2).squeeze(0).cpu().numpy()

            # 质量分数: Sigmoid (M,)
            quality = quality_scores.squeeze(0).cpu().numpy()

            # [关键适配] 将双头输出转换为SLAM格式
            # 杂波概率 (Dustbin) = 1.0 - Quality
            # 这样SLAM端可以直接使用: message = assoc_probs * (1 - dustbin)
            dustbin_probs = 1.0 - quality

        return assoc_probs, dustbin_probs, loss.item()
    
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
        
        # ===== 物理老师：生成质量标签 =====
        quality_loss = self._compute_quality_loss(quality_scores, measurements)
        
        # ===== 几何老师：生成关联标签 =====
        # 关键：只对质量好的测量进行关联训练
        assoc_loss = self._compute_association_loss(
            assoc_logits, quality_scores, measurements,
            predicted_measurements, predicted_variances
        )
        
        # 总损失
        total_loss = self.quality_weight * quality_loss + self.assoc_weight * assoc_loss

        # [调试] 每50步打印损失分解
        if hasattr(self, 'step_count') and self.step_count % 50 == 0:
            print(f"\n[损失分解 - Step {self.step_count}]")
            print(f"  质量损失: {quality_loss.item():.4f} (权重={self.quality_weight})")
            print(f"  关联损失: {assoc_loss.item():.4f} (权重={self.assoc_weight})")
            print(f"  总损失: {total_loss.item():.4f}")
            print(f"  加权贡献: 质量={self.quality_weight * quality_loss.item():.4f}, 关联={self.assoc_weight * assoc_loss.item():.4f}")

        return total_loss, quality_loss, assoc_loss
    
    def _compute_quality_loss(self, quality_scores, measurements):
        """
        物理老师：用RSS内在一致性监督质量头
        
        原理：
        - 真实测量：RSS与距离符合路径损耗模型（误差 < 6 dB）
        - 杂波：RSS与距离不符合（误差 > 10 dB）
        """
        M = measurements.shape[1]
        
        # 提取数据
        z_dist = torch.from_numpy(measurements[0, :]).float().to(self.device)
        z_rss = torch.from_numpy(measurements[2, :]).float().to(self.device)
        
        # [改进] 使用更宽松的物理模型参数
        # 原因：实际环境中RSS波动较大，过于严格的阈值会误判
        P_tx = 15.41  # 发射功率 (dBm)
        n = 2.0       # 路径损耗指数

        # 计算理论RSS
        rss_theory = P_tx - 10.0 * n * torch.log10(z_dist + 1e-6)

        # 计算RSS误差（dB）
        rss_error = torch.abs(z_rss - rss_theory)

        # 生成质量伪标签
        target_quality = torch.zeros(M, device=self.device)
        quality_mask = torch.zeros(M, device=self.device)

        # [阶段二改进] 模糊逻辑训练策略 (Fuzzy Logic Training)
        # 核心思想：引入物理相关性杂波后，简单的二分法（好/坏）不再适用
        # 需要用三区间策略来处理不同置信度的样本
        #
        # 背景：
        # - 物理相关性杂波：RSS_clutter = RSS_theory - Δ (Δ=3-20dB)
        # - 近距离杂波（Δ=3dB）：RSS误差只有3dB，非常难分辨
        # - 远距离杂波（Δ=20dB）：RSS误差20dB，容易识别
        # - 真实信号：RSS误差通常<6dB（测量噪声、模型误差）
        #
        # 三区间策略：
        # ┌─────────────────────────────────────────────────────────┐
        # │ 区间1: 核心真值区 (0-6dB)                                │
        # │   - 肯定是真实信号（直达波）                             │
        # │   - 标签 = 0.95（高质量）                               │
        # │   - 权重 = 1.0（强训练）                                │
        # ├─────────────────────────────────────────────────────────┤
        # │ 区间2: 模糊区 (6-15dB)                                  │
        # │   - 可能是衰减的真值（阴影衰落、NLOS）                   │
        # │   - 也可能是强反射杂波（Δ=3-9dB）                       │
        # │   - 标签 = 0.5（不确定）                                │
        # │   - 权重 = 0.5（弱训练，让几何头去决定）                 │
        # ├─────────────────────────────────────────────────────────┤
        # │ 区间3: 核心杂波区 (>15dB)                               │
        # │   - 肯定是杂波（反射损耗>15dB）                          │
        # │   - 标签 = 0.05（低质量）                               │
        # │   - 权重 = 1.0（强训练）                                │
        # └─────────────────────────────────────────────────────────┘
        #
        # 关键优势：
        # 1. 不强迫模型在模糊区做决定（避免过拟合）
        # 2. 让几何关联头处理模糊样本（利用空间信息）
        # 3. 只在确定的样本上强训练（提高鲁棒性）
        # 4. 软标签防止过度自信（0.05/0.95而非0/1）

        # 区间1: 核心真值区 (0-6dB)
        # RSS误差很小，肯定是真实信号
        mask_core_true = (rss_error < 6.0)
        target_quality[mask_core_true] = 0.95  # 高质量分数
        quality_mask[mask_core_true] = 1.0     # 强训练权重

        # 区间3: 核心杂波区 (>15dB)
        # RSS误差很大，肯定是杂波（反射损耗>15dB）
        mask_core_clutter = (rss_error > 15.0)
        target_quality[mask_core_clutter] = 0.05  # 低质量分数
        quality_mask[mask_core_clutter] = 1.0     # 强训练权重

        # 区间2: 模糊区 (6-15dB)
        # 可能是衰减的真值，也可能是强反射杂波
        # 策略：给中间分（0.5），降低权重（0.5），让几何头去决定
        mask_ambiguous = (~mask_core_true) & (~mask_core_clutter)
        target_quality[mask_ambiguous] = 0.5   # 中性分数（不确定）
        quality_mask[mask_ambiguous] = 0.5     # 弱训练权重（不强迫模型）

        # [调试] 每50步打印质量头统计（三区间策略）
        if hasattr(self, 'step_count') and self.step_count % 50 == 0:
            core_true_count = mask_core_true.sum().item()
            core_clutter_count = mask_core_clutter.sum().item()
            ambiguous_count = mask_ambiguous.sum().item()
            avg_quality = quality_scores.squeeze(0).mean().item()

            print(f"\n[质量头诊断 - 模糊逻辑三区间 - Step {self.step_count}]")
            print(f"  总测量数: {M}")
            print(f"  ┌─ 区间1: 核心真值 (RSS误差<6dB): {core_true_count} ({core_true_count/M*100:.1f}%) → 标签=0.95, 权重=1.0")
            print(f"  ├─ 区间2: 模糊区 (6-15dB): {ambiguous_count} ({ambiguous_count/M*100:.1f}%) → 标签=0.5, 权重=0.5")
            print(f"  └─ 区间3: 核心杂波 (RSS误差>15dB): {core_clutter_count} ({core_clutter_count/M*100:.1f}%) → 标签=0.05, 权重=1.0")
            print(f"  平均质量分数: {avg_quality:.3f}")
            print(f"  RSS误差范围: [{rss_error.min().item():.1f}, {rss_error.max().item():.1f}] dB")

            # [新增] 统计物理相关性杂波的分布
            # 理论上，反射损耗3-20dB的杂波会分布在各个区间
            print(f"  [物理相关性杂波分析]")
            print(f"    - 强反射杂波 (Δ=3-9dB) → 落入区间1或2")
            print(f"    - 中等反射杂波 (Δ=9-15dB) → 落入区间2")
            print(f"    - 弱反射杂波 (Δ>15dB) → 落入区间3")

        # 计算BCE损失（只对确定的样本）
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
        原理：如果把杂波强行拿去匹配，会教坏关联头
        """
        M = measurements.shape[1]
        K = predicted_measurements.shape[0]
        
        # 提取质量好的测量
        quality_np = quality_scores.squeeze(0).cpu().detach().numpy()
        valid_mask = quality_np > self.quality_threshold
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            # 没有质量好的测量，跳过关联训练
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

        # [调试] 每50步打印关联头统计
        if hasattr(self, 'step_count') and self.step_count % 50 == 0:
            matched_count = match_mask.sum().item()
            print(f"\n[关联头诊断 - Step {self.step_count}]")
            print(f"  质量好的测量: {len(valid_indices)}/{M} ({len(valid_indices)/M*100:.1f}%)")
            print(f"  匹配成功 (cost<{self.assoc_threshold}): {matched_count}/{len(valid_indices)} ({matched_count/max(len(valid_indices),1)*100:.1f}%)")
            if matched_count > 0:
                matched_costs = [cost_np[r, col_ind[i]] for i, r in enumerate(row_ind) if cost_np[r, col_ind[i]] < self.assoc_threshold]
                print(f"  匹配代价范围: [{min(matched_costs):.2f}, {max(matched_costs):.2f}]")

        if match_mask.sum() > 0:
            # 提取对应的logits
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
            'loss_history': self.loss_history,
            'quality_loss_history': self.quality_loss_history,
            'assoc_loss_history': self.assoc_loss_history,
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
        print(f"✓ 双头GNN权重已保存: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载模型权重"""
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
            
            print(f"✓ 双头GNN权重已加载: {checkpoint_path}")
            print(f"  - 训练步数: {self.step_count}")
            
            return True
        
        except Exception as e:
            print(f"✗ 加载权重失败: {e}")
            return False
