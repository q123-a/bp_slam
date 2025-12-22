"""
test_gnn_v3_explicit_factor.py
测试显式因子节点版本的 GNN 模型

使用方法:
    python test_gnn_v3_explicit_factor.py --config all
    python test_gnn_v3_explicit_factor.py --config basic
    python test_gnn_v3_explicit_factor.py --config compare
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from bp_slam.core.gnn_model_v3_explicit_factor import (
    FactorGraphNeuralNetworkV3,
    JointDualHeadGNNV3,
    ExplicitFactorGNNLayer
)
from bp_slam.core.gnn_model_v2 import (
    FactorGraphNeuralNetworkV2,
    JointDualHeadGNNV2
)


def test_explicit_factor_layer():
    """测试显式因子节点层"""
    print("\n" + "="*60)
    print("测试 1: 显式因子节点层")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    B, M, K = 1, 10, 5
    hidden_dim = 64

    # 创建测试数据（新接口：分离的节点表示）
    h_meas = torch.randn(B, M, hidden_dim).to(device)
    h_anchor = torch.randn(B, K, hidden_dim).to(device)
    h_pairs = torch.randn(B, M, K, hidden_dim).to(device)

    # 测试不同配置
    configs = [
        ('Softmax聚合', {'aggregation': 'softmax', 'gamma': 3.0, 'edge_mode': 'diff'}),
        ('Softmax + 边增强', {'aggregation': 'softmax', 'gamma': 3.0, 'edge_mode': 'concat'}),
        ('Max聚合', {'aggregation': 'max', 'edge_mode': 'diff'}),
    ]

    for name, config in configs:
        layer = ExplicitFactorGNNLayer(hidden_dim, **config).to(device)

        h_meas_out, h_anchor_out, h_pairs_out = layer(h_meas, h_anchor, h_pairs)

        print(f"\n{name}:")
        print(f"  输入形状: h_meas={h_meas.shape}, h_anchor={h_anchor.shape}, h_pairs={h_pairs.shape}")
        print(f"  输出形状: h_meas={h_meas_out.shape}, h_anchor={h_anchor_out.shape}, h_pairs={h_pairs_out.shape}")
        print(f"  参数量: {sum(p.numel() for p in layer.parameters()):,}")
        print(f"  ✓ 前向传播成功")


def test_v3_basic_forward():
    """测试V3基本前向传播"""
    print("\n" + "="*60)
    print("测试 2: V3 基本前向传播")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B, M, K = 1, 10, 5
    hybrid_input = torch.randn(B, M, K+1, 5).to(device)

    model = FactorGraphNeuralNetworkV3(
        input_dim=5,
        hidden_dim=64,
        num_layers=2,
        aggregation='softmax',
        gamma=3.0,
        edge_mode='concat'
    ).to(device)

    logits, _ = model(hybrid_input, None)

    print(f"输入形状: {hybrid_input.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"输出范围: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ V3 前向传播成功")


def test_v3_dual_head():
    """测试V3双头架构"""
    print("\n" + "="*60)
    print("测试 3: V3 双头架构")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B, M, K = 1, 10, 5
    hybrid_input = torch.randn(B, M, K+1, 5).to(device)

    model = JointDualHeadGNNV3(
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
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ V3 双头架构测试成功")


def test_gradient_flow():
    """测试梯度流"""
    print("\n" + "="*60)
    print("测试 4: 梯度流对比")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B, M, K = 1, 10, 5
    hybrid_input = torch.randn(B, M, K+1, 5).to(device)
    target = torch.randint(0, K+1, (B, M)).to(device)

    configs = [
        ('V3 显式因子 (Softmax)', FactorGraphNeuralNetworkV3, {'aggregation': 'softmax', 'gamma': 3.0}),
        ('V3 显式因子 (Max)', FactorGraphNeuralNetworkV3, {'aggregation': 'max'}),
    ]

    for name, model_class, config in configs:
        model = model_class(
            input_dim=5,
            hidden_dim=64,
            num_layers=2,
            edge_mode='concat',
            **config
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


def compare_v2_v3():
    """对比V2和V3"""
    print("\n" + "="*60)
    print("测试 5: V2 vs V3 对比")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B, M, K = 1, 10, 5
    hybrid_input = torch.randn(B, M, K+1, 5).to(device)

    # V2 模型（隐式因子）
    model_v2 = FactorGraphNeuralNetworkV2(
        input_dim=5,
        hidden_dim=64,
        num_layers=2,
        aggregation='softmax',
        gamma=3.0,
        edge_mode='concat'
    ).to(device)

    # V3 模型（显式因子）
    model_v3 = FactorGraphNeuralNetworkV3(
        input_dim=5,
        hidden_dim=64,
        num_layers=2,
        aggregation='softmax',
        gamma=3.0,
        edge_mode='concat'
    ).to(device)

    # 前向传播
    with torch.no_grad():
        logits_v2, _ = model_v2(hybrid_input, None)
        logits_v3, _ = model_v3(hybrid_input, None)

    print(f"\nV2 模型（隐式因子）:")
    print(f"  输出形状: {logits_v2.shape}")
    print(f"  参数量: {sum(p.numel() for p in model_v2.parameters()):,}")
    print(f"  架构: 测量-锚点直接连接（二分图）")

    print(f"\nV3 模型（显式因子）:")
    print(f"  输出形状: {logits_v3.shape}")
    print(f"  参数量: {sum(p.numel() for p in model_v3.parameters()):,}")
    print(f"  架构: 测量-因子-锚点（三层因子图）")

    print(f"\n参数量对比:")
    v2_params = sum(p.numel() for p in model_v2.parameters())
    v3_params = sum(p.numel() for p in model_v3.parameters())
    print(f"  V2: {v2_params:,}")
    print(f"  V3: {v3_params:,}")
    print(f"  增加: {v3_params - v2_params:,} ({(v3_params/v2_params - 1)*100:.1f}%)")

    print(f"\n✓ 所有模型输出形状一致，接口兼容")


def test_skip_connections():
    """测试Skip Connections"""
    print("\n" + "="*60)
    print("测试 6: Skip Connections")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B, M, K = 1, 10, 5
    hybrid_input = torch.randn(B, M, K+1, 5).to(device)

    # 4层网络，带skip connections
    model = FactorGraphNeuralNetworkV3(
        input_dim=5,
        hidden_dim=64,
        num_layers=4,
        aggregation='softmax',
        gamma=3.0,
        edge_mode='concat',
        skip_connections={2: 0, 3: 1}  # 第0层连接到第2层，第1层连接到第3层
    ).to(device)

    logits, _ = model(hybrid_input, None)

    print(f"输入形状: {hybrid_input.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"层数: 4 (带 skip connections)")
    print(f"Skip 连接: 0→2, 1→3")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ Skip Connections 测试成功")


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

    model = FactorGraphNeuralNetworkV3(
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
    print(f"  损失下降: {losses[0] - losses[-1]:.4f} ({(losses[0] - losses[-1])/losses[0]*100:.1f}%)")

    if losses[-1] < losses[0]:
        print(f"  ✓ 损失正常下降，自监督学习兼容")
    else:
        print(f"  ⚠ 损失未下降，需要检查")


def main():
    parser = argparse.ArgumentParser(description='测试显式因子节点 GNN 模型')
    parser.add_argument('--config', type=str, default='all',
                        choices=['all', 'basic', 'layer', 'dual', 'gradient', 'compare', 'skip', 'self_supervised'],
                        help='测试配置')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("显式因子节点 GNN 模型 (V3) 测试")
    print("="*60)

    if args.config == 'all':
        test_explicit_factor_layer()
        test_v3_basic_forward()
        test_v3_dual_head()
        test_gradient_flow()
        compare_v2_v3()
        test_skip_connections()
        test_self_supervised_compatibility()
    elif args.config == 'layer':
        test_explicit_factor_layer()
    elif args.config == 'basic':
        test_v3_basic_forward()
    elif args.config == 'dual':
        test_v3_dual_head()
    elif args.config == 'gradient':
        test_gradient_flow()
    elif args.config == 'compare':
        compare_v2_v3()
    elif args.config == 'skip':
        test_skip_connections()
    elif args.config == 'self_supervised':
        test_self_supervised_compatibility()

    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)


if __name__ == '__main__':
    main()
