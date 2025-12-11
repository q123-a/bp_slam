"""
快速测试脚本 - 图注意力 + 自适应RANSAC
"""

import numpy as np
from testbed import main

# ============================================================
# 方案1: 纯自监督（当前默认，作为baseline）
# ============================================================
print("\n" + "="*60)
print("方案1: 纯自监督（Baseline）")
print("="*60)
main(
    use_gnn=True,
    max_steps=100,  # 先测试100步
    num_particles=100000,
    gnn_warmup=50,
    gnn_load_checkpoint=None,
    gnn_save_checkpoint=True,
    gnn_inference_only=False,
    load_measurements='measurementbadf.mat',
    # 新增参数（在slam.py中配置）
    use_graph_attention=False,  # 不使用图注意力
    use_ransac=False            # 不使用RANSAC
)

# ============================================================
# 方案2: 图注意力增强
# ============================================================
print("\n" + "="*60)
print("方案2: 图注意力增强")
print("="*60)
main(
    use_gnn=True,
    max_steps=100,
    num_particles=100000,
    gnn_warmup=50,
    gnn_load_checkpoint=None,
    gnn_save_checkpoint=True,
    gnn_inference_only=False,
    load_measurements='measurementbadf.mat',
    # 新增参数
    use_graph_attention=True,   # 启用图注意力
    use_ransac=False            # 不使用RANSAC
)

# ============================================================
# 方案3: 混合方案（图注意力 + RANSAC）- 推荐用于论文
# ============================================================
print("\n" + "="*60)
print("方案3: 混合方案（图注意力 + RANSAC）")
print("="*60)
main(
    use_gnn=True,
    max_steps=100,
    num_particles=100000,
    gnn_warmup=50,
    gnn_load_checkpoint=None,
    gnn_save_checkpoint=True,
    gnn_inference_only=False,
    load_measurements='measurementbadf.mat',
    # 新增参数
    use_graph_attention=True,   # 启用图注意力
    use_ransac=True,            # 启用RANSAC
    ransac_threshold=2.0        # RANSAC距离阈值（米）
)
