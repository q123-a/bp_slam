#!/bin/bash
# 模型失配测试脚本
# 测试不同程度的参数失配对BP和GNN性能的影响
#
# BP假设参数（在testbed.py中配置）:
#   - 检测率: 0.95 (漏检率5%)
#   - 杂波数: 1个/帧

echo "=========================================="
echo "模型失配测试脚本"
echo "=========================================="
echo ""

# 测试参数
DATA_FILE="lwx.mat"
STEPS=900
PARTICLES=100000

# 创建结果目录
mkdir -p results/mismatch_tests
mkdir -p logs

# 测试函数
run_test() {
    local test_name=$1
    local mode=$2
    local gen_det_prob=$3
    local gen_clutter=$4
    local bp_det_prob=$5
    local bp_clutter=$6

    echo "=========================================="
    echo "测试: ${test_name} - ${mode} 模式"
    echo "=========================================="
    echo "真实环境: 检测率=${gen_det_prob}, 杂波=${gen_clutter}"
    echo "BP假设:   检测率=${bp_det_prob}, 杂波=${bp_clutter}"

    # 计算失配程度
    local det_diff=$(echo "${bp_det_prob} - ${gen_det_prob}" | bc)
    local clutter_diff=$(echo "${gen_clutter} - ${bp_clutter}" | bc)
    echo "失配程度: 检测率偏差=${det_diff}, 杂波数偏差=${clutter_diff}"
    echo ""

    # 运行测试
    python testbed.py \
        --mode ${mode} \
        --load-measurements ${DATA_FILE} \
        --add-clutter \
        --gen-detection-prob ${gen_det_prob} \
        --gen-clutter-num ${gen_clutter} \
        --steps ${STEPS} \
        --particles ${PARTICLES} \
        2>&1 | tee logs/${test_name}_${mode}.log

    # 重命名结果文件
    if [ -f "results/figure1_trajectory_anchors_${mode}.png" ]; then
        cp results/figure1_trajectory_anchors_${mode}.png results/mismatch_tests/${test_name}_trajectory_${mode}.png
        cp results/figure2_ospa_error_${mode}.png results/mismatch_tests/${test_name}_ospa_${mode}.png
        cp results/figure3_position_error_${mode}.png results/mismatch_tests/${test_name}_position_${mode}.png
        echo "✓ 结果已保存到 results/mismatch_tests/${test_name}_*_${mode}.png"
    fi

    echo ""
}

# BP假设参数（在testbed.py中配置）
BP_DET_PROB=0.95
BP_CLUTTER=1

echo "=========================================="
echo "BP假设参数: 检测率=${BP_DET_PROB}, 杂波=${BP_CLUTTER}"
echo "=========================================="
echo ""

# ==========================================
# 测试1: 模型匹配（基线）
# ==========================================
echo "==========================================
测试1: 模型匹配（基线）
  - 真实环境 = BP假设
  - 检测率: 0.95, 杂波: 1个
=========================================="
run_test "test1_matched" "bp" ${BP_DET_PROB} ${BP_CLUTTER} ${BP_DET_PROB} ${BP_CLUTTER}
run_test "test1_matched" "gnn" ${BP_DET_PROB} ${BP_CLUTTER} ${BP_DET_PROB} ${BP_CLUTTER}

# ==========================================
# 测试2: 极轻度失配（检测率-5%, 杂波+0.5）
# ==========================================
echo "==========================================
测试2: 极轻度失配
  - 检测率: 0.90 (BP假设0.95, 偏差5%)
  - 杂波数: 1.5个 (BP假设1个, 偏差0.5个)
=========================================="
run_test "test2_very_light" "bp" 0.90 1.5 ${BP_DET_PROB} ${BP_CLUTTER}
run_test "test2_very_light" "gnn" 0.90 1.5 ${BP_DET_PROB} ${BP_CLUTTER}

# ==========================================
# 测试3: 轻度失配（检测率-10%, 杂波+1）
# ==========================================
echo "==========================================
测试3: 轻度失配
  - 检测率: 0.85 (BP假设0.95, 偏差10%)
  - 杂波数: 2个 (BP假设1个, 偏差1个)
=========================================="
run_test "test3_light" "bp" 0.85 2 ${BP_DET_PROB} ${BP_CLUTTER}
run_test "test3_light" "gnn" 0.85 2 ${BP_DET_PROB} ${BP_CLUTTER}

# ==========================================
# 测试4: 中度失配（检测率-15%, 杂波+2）
# ==========================================
echo "==========================================
测试4: 中度失配
  - 检测率: 0.80 (BP假设0.95, 偏差15%)
  - 杂波数: 3个 (BP假设1个, 偏差2个)
=========================================="
run_test "test4_medium" "bp" 0.80 3 ${BP_DET_PROB} ${BP_CLUTTER}
run_test "test4_medium" "gnn" 0.80 3 ${BP_DET_PROB} ${BP_CLUTTER}

# ==========================================
# 测试5: 重度失配（检测率-25%, 杂波+4）
# ==========================================
echo "==========================================
测试5: 重度失配
  - 检测率: 0.70 (BP假设0.95, 偏差25%)
  - 杂波数: 5个 (BP假设1个, 偏差4个)
=========================================="
run_test "test5_heavy" "bp" 0.70 5 ${BP_DET_PROB} ${BP_CLUTTER}
run_test "test5_heavy" "gnn" 0.70 5 ${BP_DET_PROB} ${BP_CLUTTER}

# ==========================================
# 总结
# ==========================================
echo ""
echo "=========================================="
echo "所有测试完成！"
echo "=========================================="
echo ""
echo "结果文件位置:"
echo "  - 图片: results/mismatch_tests/"
echo "  - 日志: logs/"
echo ""
echo "对比方法:"
echo "  1. 查看 results/mismatch_tests/ 目录下的图片"
echo "  2. 对比同一测试的 BP vs GNN 性能"
echo "  3. 观察失配程度增加时的性能变化"
echo ""
