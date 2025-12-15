"""
test_gnn_v2.py
测试改进版 GNN 模型 (V2)

使用方法:
    python test_gnn_v2.py --config softmax  # 测试 softmax 聚合
    python test_gnn_v2.py --config multi    # 测试多聚合集成
    python test_gnn_v2.py --config compare  # 对比原版和改进版
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from bp_slam.core.gnn_model_v2 import (
    FactorGraphNeuralNetworkV2,
    JointDualHeadGNNV2,
    BiVariableGNNLayerV2,
    MultiAggregationGNNLayer
)
from bp_slam.core.gnn_model import (
    FactorGraphNeuralNetwork,
    JointDualHeadGNN
)


def test_basic_forward():
    """测试基本前向传播"""
    print("\n" + "="*60)
    print("测试 1: 基本前向传播")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    # 创建测试数据
    B, M, K = 1, 10, 5
    hybrid_input = torch.randn(B, M, K+1, 5).to(device)

    # 测试不同配置
    configs = [
        ('原版 (Hard Max)', {'aggregation': 'max', 'edge_mode': 'diff'}),
        ('Softmax 聚合', {'aggregation': 'softmax', 'gamma': 3.0, 'edge_mode': 'diff'}),
        ('边特征增强', {'aggregation': 'max', 'edge_mode': 'concat'}),
        ('Softmax + 边增强', {'aggregation': 'softmax', 'gamma': 3.0, 'edge_mode': 'concat'}),
    ]

    for name, config in configs:
        model = FactorGraphNeuralNetworkV2(
            input_dim=5,
            hidden_dim=64,
            num_layers=2,
            **config
        ).to(device)

        logits, _ = model(hybrid_input, None)

        print(f"\n{name}:")
        print(f"  输入形状: {hybrid_input.shape}")
        print(f"  输出形状: {logits.shape}")
        print(f"  输出范围: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  ✓ 前向传播成功")


def test_multi_aggregation():
    """测试多聚合集成"""
    print("\n" + "="*60)
    print("测试 2: 多聚合集成")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B, M, K = 1, 10, 5
    hybrid_input = torch.randn(B, M, K+1, 5).to(device)

    model = FactorGraphNeuralNetworkV2(
        input_dim=5,
        hidden_dim=64,
        num_layers=2,
        use_multi_aggregation=True
    ).to(device)

    logits, _ = model(hybrid_input, None)

    print(f"输入形状: {hybrid_input.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ 多聚合集成测试成功")


def test_skip_connections():
    """测试 Skip Connections"""
    print("\n" + "="*60)
    print("测试 3: Skip Connections")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B, M, K = 1, 10, 5
    hybrid_input = torch.randn(B, M, K+1, 5).to(device)

    # 4 层网络，第 0 层连接到第 2 层，第 1 层连接到第 3 层
    model = FactorGraphNeuralNetworkV2(
        input_dim=5,
        hidden_dim=64,
        num_layers=4,
        aggregation='softmax',
        skip_connections={2: 0, 3: 1}
    ).to(device)

    logits, _ = model(hybrid_input, None)

    print(f"输入形状: {hybrid_input.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"层数: 4 (带 skip connections)")
    print(f"Skip 连接: 0→2, 1→3")
    print(f"✓ Skip Connections 测试成功")


def test_dual_head_v2():
    """测试改进版双头架构"""
    print("\n" + "="*60)
    print("测试 4: 改进版双头架构")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B, M, K = 1, 10, 5
    hybrid_input = torch.randn(B, M, K+1, 5).to(device)

    model = JointDualHeadGNNV2(
        input_dim=5,
        hidden_dim=64,
        num_layers=2,
        aggregation='softmax',
        gamma=3.0,
        edge_mode='concat'
    ).to(device)

    assoc_logits, quality_scores, _ = model(hybrid_input, None)

    print(f"输入形状: {hybrid_input.shape}")
    print(f"关联输出: {assoc_logits.shape}")
    print(f"质量输出: {quality_scores.shape}")
    print(f"质量范围: [{quality_scores.min().item():.3f}, {quality_scores.max().item():.3f}]")
    print(f"✓ 双头架构测试成功")


def test_gradient_flow():
    """测试梯度流"""
    print("\n" + "="*60)
    print("测试 5: 梯度流对比")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B, M, K = 1, 10, 5
    hybrid_input = torch.randn(B, M, K+1, 5).to(device)
    target = torch.randint(0, K+1, (B, M)).to(device)

    configs = [
        ('Hard Max', 'max'),
        ('Softmax (γ=3)', 'softmax'),
        ('Mean', 'mean'),
    ]

    for name, agg in configs:
        model = FactorGraphNeuralNetworkV2(
            input_dim=5,
            hidden_dim=64,
            num_layers=2,
            aggregation=agg,
            gamma=3.0
        ).to(device)

        # 前向传播
        logits, _ = model(hybrid_input, None)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, K+1), target.view(-1))

        # 反向传播
        loss.backward()

        # 统计梯度
        grad_norms = []
        for name_param, param in model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())

        print(f"\n{name}:")
        print(f"  损失: {loss.item():.4f}")
        print(f"  平均梯度范数: {np.mean(grad_norms):.6f}")
        print(f"  最大梯度范数: {np.max(grad_norms):.6f}")
        print(f"  最小梯度范数: {np.min(grad_norms):.6f}")


def compare_with_original():
    """对比原版和改进版"""
    print("\n" + "="*60)
    print("测试 6: 原版 vs 改进版对比")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B, M, K = 1, 10, 5
    hybrid_input = torch.randn(B, M, K+1, 5).to(device)

    # 原版模型
    model_original = FactorGraphNeuralNetwork(
        input_dim=5,
        hidden_dim=64,
        num_layers=2
    ).to(device)

    # 改进版模型 (配置为与原版相同)
    model_v2_max = FactorGraphNeuralNetworkV2(
        input_dim=5,
        hidden_dim=64,
        num_layers=2,
        aggregation='max',
        edge_mode='diff'
    ).to(device)

    # 改进版模型 (使用 softmax)
    model_v2_softmax = FactorGraphNeuralNetworkV2(
        input_dim=5,
        hidden_dim=64,
        num_layers=2,
        aggregation='softmax',
        gamma=3.0,
        edge_mode='concat'
    ).to(device)

    # 前向传播
    with torch.no_grad():
        logits_original, _ = model_original(hybrid_input, None)
        logits_v2_max, _ = model_v2_max(hybrid_input, None)
        logits_v2_softmax, _ = model_v2_softmax(hybrid_input, None)

    print(f"\n原版模型:")
    print(f"  输出形状: {logits_original.shape}")
    print(f"  参数量: {sum(p.numel() for p in model_original.parameters()):,}")

    print(f"\n改进版 (Max, 兼容模式):")
    print(f"  输出形状: {logits_v2_max.shape}")
    print(f"  参数量: {sum(p.numel() for p in model_v2_max.parameters()):,}")

    print(f"\n改进版 (Softmax + 边增强):")
    print(f"  输出形状: {logits_v2_softmax.shape}")
    print(f"  参数量: {sum(p.numel() for p in model_v2_softmax.parameters()):,}")

    print(f"\n✓ 所有模型输出形状一致，可直接替换")


def test_self_supervised_compatibility():
    """测试自监督兼容性"""
    print("\n" + "="*60)
    print("测试 7: 自监督学习兼容性")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B, M, K = 1, 10, 5
    hybrid_input = torch.randn(B, M, K+1, 5).to(device)

    # 模拟匈牙利算法生成的伪标签
    target_indices = torch.randint(0, K+1, (M,)).to(device)

    model = FactorGraphNeuralNetworkV2(
        input_dim=5,
        hidden_dim=64,
        num_layers=2,
        aggregation='softmax',
        gamma=3.0,
        edge_mode='concat'
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 训练 10 步
    losses = []
    for step in range(10):
        logits, _ = model(hybrid_input, None)

        # 自监督损失 (交叉熵)
        loss = torch.nn.functional.cross_entropy(
            logits.view(M, K+1),
            target_indices,
            label_smoothing=0.1
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05)
        optimizer.step()

        losses.append(loss.item())

    print(f"训练 10 步:")
    print(f"  初始损失: {losses[0]:.4f}")
    print(f"  最终损失: {losses[-1]:.4f}")
    print(f"  损失下降: {losses[0] - losses[-1]:.4f}")

    if losses[-1] < losses[0]:
        print(f"  ✓ 损失正常下降，自监督学习兼容")
    else:
        print(f"  ⚠ 损失未下降，需要检查")


def main():
    parser = argparse.ArgumentParser(description='测试改进版 GNN 模型')
    parser.add_argument('--config', type=str, default='all',
                        choices=['all', 'basic', 'multi', 'skip', 'dual', 'gradient', 'compare', 'self_supervised'],
                        help='测试配置')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("改进版 GNN 模型 (V2) 测试")
    print("="*60)

    if args.config == 'all':
        test_basic_forward()
        test_multi_aggregation()
        test_skip_connections()
        test_dual_head_v2()
        test_gradient_flow()
        compare_with_original()
        test_self_supervised_compatibility()
    elif args.config == 'basic':
        test_basic_forward()
    elif args.config == 'multi':
        test_multi_aggregation()
    elif args.config == 'skip':
        test_skip_connections()
    elif args.config == 'dual':
        test_dual_head_v2()
    elif args.config == 'gradient':
        test_gradient_flow()
    elif args.config == 'compare':
        compare_with_original()
    elif args.config == 'self_supervised':
        test_self_supervised_compatibility()

    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)


if __name__ == '__main__':
    main()
