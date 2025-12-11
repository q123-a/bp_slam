"""
运行完整的混合方案实验（1500步）
图注意力 + 自适应RANSAC
"""

from testbed import main

print("\n" + "="*70)
print("完整混合方案实验（1500步）")
print("图注意力 + 自适应RANSAC")
print("="*70)

# 运行完整的1500步实验
main(
    use_gnn=True,
    max_steps=1500,              # 完整的1500步
    num_particles=100000,
    gnn_warmup=75,               # 预热步数（1500的5%）
    gnn_load_checkpoint=None,    # 从头训练
    gnn_save_checkpoint=True,    # 保存权重
    gnn_inference_only=False,    # 训练模式
    load_measurements='measurement1500zhen.mat'
)

print("\n" + "="*70)
print("✓ 完整实验完成！")
print("="*70)
print("\n结果文件：")
print("  - results/figure1_trajectory_anchors.png - 轨迹和锚点可视化")
print("  - results/figure2_ospa_error.png - OSPA误差曲线")
print("  - results/figure3_position_error.png - 位置误差曲线")
print("  - results/gnn_loss_curve.png - GNN训练损失曲线")
print("  - checkpoints/gnn_model.pth - 训练好的模型权重")
