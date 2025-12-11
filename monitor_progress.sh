#!/bin/bash
# 实时监控1500步实验进度

echo "========================================"
echo "混合方案实验进度监控"
echo "========================================"
echo ""

# 检查日志文件是否存在
if [ ! -f "full_hybrid_output.log" ]; then
    echo "❌ 日志文件不存在，实验可能还未开始"
    exit 1
fi

# 获取最新进度
echo "📊 当前进度："
echo "----------------------------------------"
tail -20 full_hybrid_output.log | grep "Time instance:" | tail -1

# 获取最新误差
echo ""
echo "📉 最新位置误差："
echo "----------------------------------------"
tail -20 full_hybrid_output.log | grep "Position error agent:" | tail -1

# 获取RANSAC信息（如果有）
echo ""
echo "🔍 RANSAC统计（最近10步）："
echo "----------------------------------------"
grep "\[RANSAC\]" full_hybrid_output.log | tail -5 || echo "暂无RANSAC输出"

# 估算完成时间
current_step=$(tail -50 full_hybrid_output.log | grep "Time instance:" | tail -1 | awk '{print $3}')
if [ ! -z "$current_step" ]; then
    total_steps=1500
    progress=$(echo "scale=2; $current_step / $total_steps * 100" | bc)
    remaining=$((total_steps - current_step))

    echo ""
    echo "⏱️  进度统计："
    echo "----------------------------------------"
    echo "已完成: $current_step / $total_steps 步 ($progress%)"
    echo "剩余: $remaining 步"

    # 估算剩余时间（假设每步1秒）
    remaining_minutes=$((remaining / 60))
    remaining_hours=$((remaining_minutes / 60))
    remaining_minutes=$((remaining_minutes % 60))
    echo "预计剩余时间: ${remaining_hours}小时${remaining_minutes}分钟"
fi

echo ""
echo "========================================"
echo "💡 提示："
echo "  - 实时查看: tail -f full_hybrid_output.log"
echo "  - 再次检查: bash monitor_progress.sh"
echo "========================================"
