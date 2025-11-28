#!/bin/bash
# 监控完整测试进度

LOG_FILE="testbed_full_output.log"

echo "监控完整测试进度（900步，100000粒子）"
echo "预计运行时间：8-10分钟"
echo "=========================================="
echo ""

while true; do
    # 检查最新的时间步
    CURRENT_STEP=$(tail -100 "$LOG_FILE" 2>/dev/null | grep "Time instance:" | tail -1 | awk '{print $3}')

    if [ -n "$CURRENT_STEP" ]; then
        PROGRESS=$(echo "scale=1; $CURRENT_STEP * 100 / 900" | bc)
        echo -ne "\r当前进度: 第 $CURRENT_STEP / 900 步 (${PROGRESS}%)   "
    fi

    # 检查是否完成
    if grep -q "算法运行完成" "$LOG_FILE" 2>/dev/null; then
        echo ""
        echo ""
        echo "=========================================="
        echo "✓ 完整测试已完成！"
        echo "=========================================="
        echo ""

        # 显示最终结果
        tail -50 "$LOG_FILE" | grep -E "算法运行完成|最终|平均|OSPA|传感器"
        break
    fi

    sleep 5
done
