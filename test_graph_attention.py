"""
测试图注意力 + 自适应RANSAC功能
"""

import torch
import numpy as np
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(__file__))

from bp_slam.core.gnn_trainer_improved import GNNTrainerImproved

def test_basic_functionality():
    """测试基本功能：不使用图注意力和RANSAC"""
    print("=" * 60)
    print("测试 1: 基本功能（无图注意力，无RANSAC）")
    print("=" * 60)

    trainer = GNNTrainerImproved(
        device='cpu',
        lr=1e-3,
        hidden_dim=64,
        use_graph_attention=False,
        use_ransac=False,
        seed=42
    )

    # 创建模拟数据
    M = 10  # 10个测量
    K = 5   # 5个锚点

    hybrid_tensor = torch.randn(1, M, K+1, 5)
    measurements = np.random.randn(3, M)
    predicted_measurements = np.random.randn(K)
    predicted_variances = np.ones(K) * 0.1

    # 执行一步训练
    legacy_probs, dustbin_probs, loss = trainer.step(
        hybrid_tensor, measurements, predicted_measurements, predicted_variances
    )

    print(f"✓ 基本功能测试通过")
    print(f"  - Legacy probs shape: {legacy_probs.shape}")
    print(f"  - Dustbin probs shape: {dustbin_probs.shape}")
    print(f"  - Loss: {loss:.4f}")
    print()

def test_graph_attention():
    """测试图注意力功能"""
    print("=" * 60)
    print("测试 2: 图注意力功能")
    print("=" * 60)

    trainer = GNNTrainerImproved(
        device='cpu',
        lr=1e-3,
        hidden_dim=64,
        use_graph_attention=True,
        use_ransac=False,
        seed=42
    )

    # 创建模拟数据
    M = 10
    K = 5

    hybrid_tensor = torch.randn(1, M, K+1, 5)
    measurements = np.random.randn(3, M)
    predicted_measurements = np.random.randn(K)
    predicted_variances = np.ones(K) * 0.1

    # 执行一步训练
    legacy_probs, dustbin_probs, loss = trainer.step(
        hybrid_tensor, measurements, predicted_measurements, predicted_variances
    )

    print(f"✓ 图注意力功能测试通过")
    print(f"  - Legacy probs shape: {legacy_probs.shape}")
    print(f"  - Dustbin probs shape: {dustbin_probs.shape}")
    print(f"  - Loss: {loss:.4f}")
    print()

def test_ransac():
    """测试RANSAC功能"""
    print("=" * 60)
    print("测试 3: 自适应RANSAC功能")
    print("=" * 60)

    trainer = GNNTrainerImproved(
        device='cpu',
        lr=1e-3,
        hidden_dim=64,
        use_graph_attention=False,
        use_ransac=True,
        ransac_threshold=2.0,
        seed=42
    )

    # 创建模拟数据（包含杂波）
    M = 15
    K = 5

    hybrid_tensor = torch.randn(1, M, K+1, 5)

    # 创建测量：前10个是真实测量，后5个是杂波
    measurements = np.random.randn(3, M)
    measurements[0, :10] = np.random.randn(10) * 0.5  # 真实测量，小噪声
    measurements[0, 10:] = np.random.randn(5) * 10.0  # 杂波，大噪声

    predicted_measurements = np.random.randn(K)
    predicted_variances = np.ones(K) * 0.1

    # 执行一步训练
    legacy_probs, dustbin_probs, loss = trainer.step(
        hybrid_tensor, measurements, predicted_measurements, predicted_variances
    )

    print(f"✓ RANSAC功能测试通过")
    print(f"  - Legacy probs shape: {legacy_probs.shape}")
    print(f"  - Dustbin probs shape: {dustbin_probs.shape}")
    print(f"  - Loss: {loss:.4f}")
    print(f"  - Dustbin probs (前10个): {dustbin_probs[:10].mean():.4f}")
    print(f"  - Dustbin probs (后5个杂波): {dustbin_probs[10:].mean():.4f}")
    print()

def test_hybrid():
    """测试混合功能：图注意力 + RANSAC"""
    print("=" * 60)
    print("测试 4: 混合功能（图注意力 + RANSAC）")
    print("=" * 60)

    trainer = GNNTrainerImproved(
        device='cpu',
        lr=1e-3,
        hidden_dim=64,
        use_graph_attention=True,
        use_ransac=True,
        ransac_threshold=2.0,
        seed=42
    )

    # 创建模拟数据
    M = 15
    K = 5

    hybrid_tensor = torch.randn(1, M, K+1, 5)

    # 创建测量：前10个是真实测量，后5个是杂波
    measurements = np.random.randn(3, M)
    measurements[0, :10] = np.random.randn(10) * 0.5
    measurements[0, 10:] = np.random.randn(5) * 10.0

    predicted_measurements = np.random.randn(K)
    predicted_variances = np.ones(K) * 0.1

    # 执行多步训练
    print("执行5步训练...")
    for step in range(5):
        legacy_probs, dustbin_probs, loss = trainer.step(
            hybrid_tensor, measurements, predicted_measurements, predicted_variances
        )
        if step == 0 or step == 4:
            print(f"  Step {step+1}: Loss={loss:.4f}, "
                  f"Dustbin(真实)={dustbin_probs[:10].mean():.4f}, "
                  f"Dustbin(杂波)={dustbin_probs[10:].mean():.4f}")

    print(f"✓ 混合功能测试通过")
    print()

def test_clutter_detection():
    """测试杂波检测能力"""
    print("=" * 60)
    print("测试 5: 杂波检测能力")
    print("=" * 60)

    # 测试不同杂波率
    clutter_ratios = [0.0, 0.2, 0.5, 0.8]

    for clutter_ratio in clutter_ratios:
        print(f"\n杂波率: {clutter_ratio*100:.0f}%")

        trainer = GNNTrainerImproved(
            device='cpu',
            lr=1e-3,
            hidden_dim=64,
            use_graph_attention=True,
            use_ransac=True,
            ransac_threshold=2.0,
            seed=42
        )

        M = 20
        K = 5
        num_clutter = int(M * clutter_ratio)
        num_inliers = M - num_clutter

        hybrid_tensor = torch.randn(1, M, K+1, 5)
        measurements = np.random.randn(3, M)

        # 前num_inliers个是真实测量
        measurements[0, :num_inliers] = np.random.randn(num_inliers) * 0.5
        # 后num_clutter个是杂波
        measurements[0, num_inliers:] = np.random.randn(num_clutter) * 10.0

        predicted_measurements = np.random.randn(K)
        predicted_variances = np.ones(K) * 0.1

        # 训练10步
        for step in range(10):
            legacy_probs, dustbin_probs, loss = trainer.step(
                hybrid_tensor, measurements, predicted_measurements, predicted_variances
            )

        # 统计结果
        if num_inliers > 0:
            inlier_dustbin = dustbin_probs[:num_inliers].mean()
        else:
            inlier_dustbin = 0.0

        if num_clutter > 0:
            clutter_dustbin = dustbin_probs[num_inliers:].mean()
        else:
            clutter_dustbin = 0.0

        print(f"  真实测量的Dustbin概率: {inlier_dustbin:.4f}")
        print(f"  杂波测量的Dustbin概率: {clutter_dustbin:.4f}")
        print(f"  分离度: {clutter_dustbin - inlier_dustbin:.4f}")

    print(f"\n✓ 杂波检测能力测试完成")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("图注意力 + 自适应RANSAC 功能测试")
    print("=" * 60 + "\n")

    try:
        test_basic_functionality()
        test_graph_attention()
        test_ransac()
        test_hybrid()
        test_clutter_detection()

        print("=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
