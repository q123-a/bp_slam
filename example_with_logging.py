#!/usr/bin/env python3
"""
带日志记录的训练示例
演示如何使用日志系统记录训练过程
"""

import numpy as np
import argparse
from bp_slam.utils.logger import create_logger, MetricsTracker

def example_training_with_logging():
    """示例：使用日志系统进行训练"""

    # 1. 创建日志记录器
    logger = create_logger(
        log_dir='logs',
        experiment_name='gnn_training_example',
        console_level='INFO'  # 可选: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    )

    # 2. 记录实验配置
    config = {
        'mode': 'gnn',
        'steps': 900,
        'particles': 100000,
        'hidden_dim': 128,
        'learning_rate': 1e-3,
        'data_file': 'measurement1500.mat',
        'use_amplitude': True
    }
    logger.log_config(config)

    # 3. 创建指标追踪器
    tracker = MetricsTracker()

    # 4. 模拟训练过程
    logger.info("开始训练...")

    num_steps = 100
    for step in range(1, num_steps + 1):
        # 模拟训练步骤
        loss = 2.0 * np.exp(-step / 50) + 0.5 + np.random.randn() * 0.1
        position_error = 1.5 * np.exp(-step / 40) + 0.3 + np.random.randn() * 0.05
        num_anchors = np.random.randint(10, 20)

        # 更新追踪器
        tracker.update(
            loss=loss,
            position_error=position_error,
            num_anchors=num_anchors
        )

        # 每 10 步记录一次
        if step % 10 == 0:
            logger.log_step(step, {
                'loss': loss,
                'pos_error': position_error,
                'anchors': num_anchors,
                'avg_loss_10': tracker.get_average('loss', last_n=10)
            })

        # 每 50 步记录 GNN 训练详情
        if step % 50 == 0:
            logger.log_gnn_training(
                step=step,
                loss=loss,
                num_measurements=15,
                num_anchors=num_anchors,
                positive_samples=10,
                negative_samples=5
            )

    # 5. 记录实验总结
    summary = tracker.summary()
    summary['total_steps'] = num_steps
    summary['final_loss'] = tracker.get_latest('loss')

    logger.log_summary(summary)

    # 6. 关闭日志记录器
    logger.close()

    print(f"\n✓ 训练完成！")
    print(f"  日志文件: {logger.log_file}")
    print(f"  指标文件: {logger.metrics_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='带日志记录的训练示例')
    parser.add_argument('--run-example', action='store_true',
                       help='运行示例训练')

    args = parser.parse_args()

    if args.run_example:
        example_training_with_logging()
    else:
        print("使用方法:")
        print("  python example_with_logging.py --run-example")
        print("\n这将创建一个示例训练日志，展示日志系统的使用方法。")

if __name__ == "__main__":
    main()
