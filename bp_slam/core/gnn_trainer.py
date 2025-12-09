"""
bp_slam/core/gnn_trainer.py
自监督训练器 - Hard-EM 版本
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
from .gnn_model import FactorGraphNeuralNetwork

class GNNTrainer:
    def __init__(self, device='cuda', lr=1e-3, hidden_dim=64, checkpoint_path=None, seed=42):
        """
        初始化 GNN 训练器

        参数:
            device: 设备 ('cuda' 或 'cpu')
            lr: 学习率
            hidden_dim: 隐藏层维度 (默认 128)
            checkpoint_path: 权重文件路径 (如果提供，则加载预训练权重)
            seed: 随机种子 (默认 42)，设为 None 则不固定种子
        """
        self.device = device
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.seed = seed

        # 设置随机种子以确保可复现性
        if seed is not None:
            self._set_seed(seed)

        # input_dim=5 (混合特征: LogProb, Residual, Variance, Existence, Amplitude)
        self.model = FactorGraphNeuralNetwork(input_dim=5, hidden_dim=hidden_dim).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.step_count = 0

        # Loss 历史记录
        self.loss_history = []

        # [新增] 跨帧记忆状态
        self.hidden_state = None

        # 如果提供了权重文件，则加载
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def _set_seed(self, seed):
        """
        设置所有随机种子以确保可复现性

        参数:
            seed: 随机种子值
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # 多GPU情况

        # 确保 CUDA 操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f"✓ 随机种子已设置: {seed}")
        print(f"  - Python random: {seed}")
        print(f"  - NumPy: {seed}")
        print(f"  - PyTorch: {seed}")
        if torch.cuda.is_available():
            print(f"  - CUDA: {seed}")
            print(f"  - cuDNN deterministic: True")

    def reset_hidden_state(self):
        """在新的序列开始时调用，清空 GRU 记忆"""
        self.hidden_state = None

    def step(self, hybrid_tensor, measurements, predicted_measurements, predicted_variances, num_iterations=5):
        """
        执行一步训练/推理（支持多次迭代）

        参数:
            hybrid_tensor: (1, M, K+1, 5) 混合特征张量 [LogProb, Residual, Variance, Existence, Amplitude]
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

        # [关键修改] 处理 hidden_state 的梯度截断
        # 我们只做单步更新，或者短时序 BPTT
        # 在进入 step 之前，必须 detach 掉上一帧的梯度，否则 PyTorch 会试图回传到第 0 步
        if self.hidden_state is not None:
            h_in = self.hidden_state.detach()
        else:
            h_in = None

        # 多次迭代训练
        total_loss = 0.0
        self.model.train()

        for iter_idx in range(num_iterations):
            # 1. [修改] 前向推理 - 接收两个返回值
            logits, h_out = self.model(hybrid_tensor, h_in)  # (1, M, K+1), (1, hidden_dim)

            # 2. 计算自监督 Loss (Hard-EM)
            loss = self._compute_hard_em_loss(
                logits, measurements, predicted_measurements, predicted_variances
            )

            # 3. 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_loss += loss.item()

            # [关键] 在迭代过程中更新 h_in，但要 detach
            # 这样每次迭代都能利用最新的记忆，但不会累积梯度
            h_in = h_out.detach()

        # [关键] 更新内部记忆，供下一帧使用
        # 注意：这里我们存下 h_out，但在下一次 step 开始时我们会 detach 它
        self.hidden_state = h_out

        # 4. 推理模式：解析输出（使用最后一次迭代的结果）
        with torch.no_grad():
            self.model.eval()
            eval_logits, _ = self.model(hybrid_tensor, self.hidden_state.detach())

            # Softmax 归一化
            all_probs = F.softmax(eval_logits[0], dim=-1)  # (M, K+1)

            # 分离 Legacy (前K列) 和 Dustbin (最后1列)
            legacy_probs = all_probs[:, :-1]  # (M, K)
            dustbin_probs = all_probs[:, -1]   # (M,)

        # 返回平均损失
        avg_loss = total_loss / num_iterations

        # 记录 Loss 历史
        self.loss_history.append(avg_loss)

        return legacy_probs.cpu().numpy(), dustbin_probs.cpu().numpy(), avg_loss

    def _compute_hard_em_loss(self, logits, measurements, predicted_measurements, predicted_variances):
        """
        Hard-EM Loss: 使用几何一致性 + 幅度信息生成伪标签

        规则:
        1. 计算标准化残差 (几何一致性)
        2. 计算幅度一致性 (物理一致性)
        3. 必须同时满足几何和物理一致性才是正样本
        """
        # 准备数据
        z_meas = torch.from_numpy(measurements[0, :]).float().to(self.device)  # (M,) 距离
        # [新增] 提取幅度
        z_rss = torch.from_numpy(measurements[2, :]).float().to(self.device)   # 幅度 (M,)

        z_pred = torch.from_numpy(predicted_measurements).float().to(self.device) # 预测距离 (K,)
        var_meas = torch.from_numpy(measurements[1, :]).float().to(self.device)
        var_pred = torch.from_numpy(predicted_variances).float().to(self.device)

        # 1. 计算联合标准差 & 几何残差 (保持不变)
        joint_std = torch.sqrt(var_meas.unsqueeze(1) + var_pred.unsqueeze(0))
        diff_mat = z_meas.unsqueeze(1) - z_pred.unsqueeze(0)
        normalized_residuals = torch.abs(diff_mat) / (joint_std + 1e-6)

        # 2. 找到几何上最近的锚点
        min_residuals, min_idx = torch.min(normalized_residuals, dim=1)

        # === [新增] 计算幅度一致性 ===
        P_tx = 15.41  # 校准后的发射功率 (dBm)
        n = 2.0
        # 根据预测距离算出理论幅度
        safe_pred_dist = torch.clamp(z_pred, min=0.1)
        rss_pred = P_tx - 10 * n * torch.log10(safe_pred_dist) # (K,)

        # 计算幅度差 (M, K)
        rss_diff_mat = rss_pred.unsqueeze(0) - z_rss.unsqueeze(1)

        # 提取最佳几何匹配对应的幅度差 (M,)
        # 我们只关心那个距离最近的锚点，它的幅度对不对
        selected_rss_diff = torch.gather(rss_diff_mat, 1, min_idx.unsqueeze(1)).squeeze(1)

        # === [修改] 判定逻辑 ===
        target_indices = torch.full((z_meas.shape[0],), predicted_measurements.shape[0], dtype=torch.long, device=self.device)

        # 规则 A: 几何上要吻合 (3倍标准差 或 2米内)
        is_geo_valid = (min_residuals < 3.0) | (torch.abs(torch.gather(diff_mat, 1, min_idx.unsqueeze(1)).squeeze(1)) < 2.0)

        # 规则 B: 物理上要吻合 (幅度差不能太大)
        # 允许 10dB 的误差。如果衰减超过 10dB，说明是反射/遮挡
        # 注意：这里取绝对值或者单边判断都可以，通常 ghost 是衰减大，所以 diff 是正的大值
        is_phy_valid = selected_rss_diff < 10.0

        # [核心] 满足任一条件即可为正样本（OR 逻辑）
        valid_mask = is_geo_valid | is_phy_valid

        target_indices[valid_mask] = min_idx[valid_mask]

        # 4. 计算 Loss (保持不变)
        loss = F.cross_entropy(logits.view(z_meas.shape[0], -1), target_indices, label_smoothing=0.1)

        return loss

    def save_checkpoint(self, checkpoint_path, epoch=None, additional_info=None):
        """
        保存模型权重和训练状态

        参数:
            checkpoint_path: 保存路径 (如 'checkpoints/gnn_model.pth')
            epoch: 当前训练轮数 (可选)
            additional_info: 额外信息字典 (可选)
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'hidden_dim': self.hidden_dim,
            'lr': self.lr,
            'loss_history': self.loss_history,  # 保存 Loss 历史
        }

        if epoch is not None:
            checkpoint['epoch'] = epoch

        if additional_info is not None:
            checkpoint.update(additional_info)

        torch.save(checkpoint, checkpoint_path)
        print(f"✓ GNN 权重已保存: {checkpoint_path}")
        print(f"  - Loss 历史记录: {len(self.loss_history)} 个数据点")

    def load_checkpoint(self, checkpoint_path):
        """
        加载模型权重和训练状态

        参数:
            checkpoint_path: 权重文件路径
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            print(f"⚠ 权重文件不存在: {checkpoint_path}")
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # 加载模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # 加载优化器状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 加载训练步数
            self.step_count = checkpoint.get('step_count', 0)

            # 加载 Loss 历史
            self.loss_history = checkpoint.get('loss_history', [])

            print(f"✓ GNN 权重已加载: {checkpoint_path}")
            print(f"  - 训练步数: {self.step_count}")
            print(f"  - Loss 历史记录: {len(self.loss_history)} 个数据点")
            if 'epoch' in checkpoint:
                print(f"  - 训练轮数: {checkpoint['epoch']}")

            return True

        except Exception as e:
            print(f"✗ 加载权重失败: {e}")
            return False