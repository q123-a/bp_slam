"""
对比三种方案：纯自监督 vs 图注意力 vs 混合方案
使用现有的testbed.py框架
"""

import sys
import os

# 修改testbed.py以支持新参数
def run_comparison():
    """运行三种方案的对比实验"""

    print("\n" + "="*70)
    print("图注意力 + 自适应RANSAC 对比实验")
    print("="*70)

    # 实验参数
    max_steps = 100  # 先测试100步
    num_particles = 100000
    gnn_warmup = 50

    # ============================================================
    # 方案1: 纯自监督（Baseline）
    # ============================================================
    print("\n" + "="*70)
    print("方案1: 纯自监督（Baseline）")
    print("="*70)

    # 临时修改slam.py的参数
    import bp_slam.core.slam as slam_module

    # 保存原始的bp_based_mint_slam函数
    original_bp_slam = slam_module.bp_based_mint_slam

    def run_with_params(use_graph_attention, use_ransac, method_name):
        """运行实验的辅助函数"""
        print(f"\n{'='*70}")
        print(f"实验: {method_name}")
        print(f"{'='*70}")
        print(f"配置: 图注意力={use_graph_attention}, RANSAC={use_ransac}")

        # 动态修改参数
        def modified_bp_slam(data_va, cluttered_measurements, parameters, true_trajectory):
            # 添加新参数
            parameters['gnn_use_graph_attention'] = use_graph_attention
            parameters['gnn_use_ransac'] = use_ransac
            parameters['gnn_ransac_threshold'] = 2.0

            # 调用原始函数
            return original_bp_slam(data_va, cluttered_measurements, parameters, true_trajectory)

        # 临时替换函数
        slam_module.bp_based_mint_slam = modified_bp_slam

        try:
            # 导入并运行testbed
            from testbed import main

            main(
                use_gnn=True,
                max_steps=max_steps,
                num_particles=num_particles,
                gnn_warmup=gnn_warmup,
                gnn_load_checkpoint=None,
                gnn_save_checkpoint=False,
                gnn_inference_only=False,
                load_measurements='measurementbadf.mat'
            )

            print(f"\n✓ {method_name} 完成!")

        finally:
            # 恢复原始函数
            slam_module.bp_based_mint_slam = original_bp_slam

    # 运行三种方案
    run_with_params(False, False, "方案1: 纯自监督（Baseline）")
    run_with_params(True, False, "方案2: 图注意力增强")
    run_with_params(True, True, "方案3: 混合方案（图注意力 + RANSAC）")

    print("\n" + "="*70)
    print("✓ 所有实验完成！")
    print("="*70)
    print("\n提示：")
    print("  - 查看results/目录下的图表和数据")
    print("  - 对比三种方案的位置误差和OSPA误差")
    print("  - 观察RANSAC统计信息（方案3）")

if __name__ == "__main__":
    run_comparison()
