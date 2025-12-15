#!/bin/bash
# 双头架构测试脚本

echo "=========================================="
echo "双头GNN架构测试"
echo "=========================================="

# 测试1: 双头GNN在失配模式下的性能
echo ""
echo "测试1: 双头GNN + 失配模式"
echo "------------------------------------------"
python testbed.py --mode gnn --steps 900 \
  --load-measurements measurementbadf.mat \
  --add-clutter --mismatch-mode

# 保存结果
mv results/results_gnn.npz results/results_gnn_dual_head_mismatch.npz

echo ""
echo "测试1完成，结果已保存到: results/results_gnn_dual_head_mismatch.npz"

# 测试2: BP在失配模式下的性能（对比基准）
echo ""
echo "测试2: BP + 失配模式（对比基准）"
echo "------------------------------------------"
python testbed.py --mode bp --steps 900 \
  --load-measurements measurementbadf.mat \
  --add-clutter --mismatch-mode

# 保存结果
mv results/results_bp.npz results/results_bp_mismatch.npz

echo ""
echo "测试2完成，结果已保存到: results/results_bp_mismatch.npz"

# 对比分析
echo ""
echo "=========================================="
echo "对比分析"
echo "=========================================="
python -c "
import numpy as np

# 加载结果
gnn_data = np.load('results/results_gnn_dual_head_mismatch.npz', allow_pickle=True)
bp_data = np.load('results/results_bp_mismatch.npz', allow_pickle=True)

# 提取误差
gnn_mean = gnn_data['mean_error']
gnn_max = gnn_data['max_error']
gnn_final = gnn_data['final_error']

bp_mean = bp_data['mean_error']
bp_max = bp_data['max_error']
bp_final = bp_data['final_error']

# 打印对比
print('\n位置误差对比 (失配模式):')
print('=' * 60)
print(f'指标                双头GNN          BP              改进')
print('-' * 60)
print(f'平均误差 (m)        {gnn_mean:.6f}        {bp_mean:.6f}        {(bp_mean-gnn_mean)/bp_mean*100:+.2f}%')
print(f'最大误差 (m)        {gnn_max:.6f}        {bp_max:.6f}        {(bp_max-gnn_max)/bp_max*100:+.2f}%')
print(f'最终误差 (m)        {gnn_final:.6f}        {bp_final:.6f}        {(bp_final-gnn_final)/bp_final*100:+.2f}%')
print('=' * 60)

# 判断结果
if gnn_mean < bp_mean:
    print('\n✓ 成功！双头GNN在失配模式下优于BP')
    print(f'  平均误差改进: {(bp_mean-gnn_mean)/bp_mean*100:.2f}%')
else:
    print('\n✗ 双头GNN仍未超越BP')
    print(f'  平均误差差距: {(gnn_mean-bp_mean)/bp_mean*100:.2f}%')
"

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
