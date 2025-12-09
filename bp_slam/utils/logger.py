"""
日志工具模块
提供训练过程的日志记录功能
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import json


class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, log_dir='logs', experiment_name=None, console_level=logging.INFO):
        """
        初始化日志记录器

        参数:
            log_dir: 日志保存目录
            experiment_name: 实验名称（如果为 None，使用时间戳）
            console_level: 控制台输出级别
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 生成实验名称
        if experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f'exp_{timestamp}'

        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f'{experiment_name}.log'
        self.metrics_file = self.log_dir / f'{experiment_name}_metrics.json'

        # 初始化指标存储
        self.metrics = {
            'train_loss': [],
            'position_error': [],
            'ospa_error': [],
            'num_anchors': [],
            'timestamps': []
        }

        # 配置日志记录器
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.DEBUG)

        # 清除已有的处理器
        self.logger.handlers.clear()

        # 文件处理器（记录所有级别）
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # 控制台处理器（可配置级别）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # 记录初始化信息
        self.info(f"日志系统初始化完成")
        self.info(f"实验名称: {experiment_name}")
        self.info(f"日志文件: {self.log_file}")
        self.info(f"指标文件: {self.metrics_file}")

    def info(self, message):
        """记录 INFO 级别日志"""
        self.logger.info(message)

    def debug(self, message):
        """记录 DEBUG 级别日志"""
        self.logger.debug(message)

    def warning(self, message):
        """记录 WARNING 级别日志"""
        self.logger.warning(message)

    def error(self, message):
        """记录 ERROR 级别日志"""
        self.logger.error(message)

    def log_config(self, config):
        """记录配置信息"""
        self.info("=" * 60)
        self.info("实验配置:")
        for key, value in config.items():
            self.info(f"  {key}: {value}")
        self.info("=" * 60)

    def log_step(self, step, metrics_dict):
        """
        记录训练步骤信息

        参数:
            step: 当前步数
            metrics_dict: 指标字典，例如 {'loss': 1.23, 'error': 0.45}
        """
        # 记录到日志文件
        message = f"Step {step:4d}"
        for key, value in metrics_dict.items():
            if isinstance(value, float):
                message += f" | {key}: {value:.4f}"
            else:
                message += f" | {key}: {value}"
        self.info(message)

        # 保存到指标存储
        timestamp = datetime.now().isoformat()
        self.metrics['timestamps'].append(timestamp)

        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def log_epoch(self, epoch, metrics_dict):
        """
        记录训练轮次信息

        参数:
            epoch: 当前轮次
            metrics_dict: 指标字典
        """
        self.info("=" * 60)
        message = f"Epoch {epoch}"
        for key, value in metrics_dict.items():
            if isinstance(value, float):
                message += f" | {key}: {value:.4f}"
            else:
                message += f" | {key}: {value}"
        self.info(message)
        self.info("=" * 60)

    def log_gnn_training(self, step, loss, num_measurements, num_anchors,
                         positive_samples=None, negative_samples=None):
        """
        记录 GNN 训练信息

        参数:
            step: 当前步数
            loss: 损失值
            num_measurements: 测量数量
            num_anchors: 锚点数量
            positive_samples: 正样本数量
            negative_samples: 负样本数量
        """
        message = f"GNN Step {step:4d} | Loss: {loss:.4f} | M: {num_measurements} | K: {num_anchors}"

        if positive_samples is not None and negative_samples is not None:
            total = positive_samples + negative_samples
            pos_ratio = positive_samples / total * 100 if total > 0 else 0
            message += f" | Pos: {positive_samples}/{total} ({pos_ratio:.1f}%)"

        self.debug(message)

    def save_metrics(self):
        """保存指标到 JSON 文件"""
        try:
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=2, ensure_ascii=False)
            self.debug(f"指标已保存到: {self.metrics_file}")
        except Exception as e:
            self.error(f"保存指标失败: {e}")

    def log_summary(self, summary_dict):
        """
        记录实验总结

        参数:
            summary_dict: 总结信息字典
        """
        self.info("")
        self.info("=" * 60)
        self.info("实验总结:")
        self.info("=" * 60)
        for key, value in summary_dict.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")
        self.info("=" * 60)

        # 保存指标
        self.save_metrics()

    def close(self):
        """关闭日志记录器"""
        self.save_metrics()
        self.info("日志记录器关闭")

        # 关闭所有处理器
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class MetricsTracker:
    """指标追踪器（轻量级版本，用于实时追踪）"""

    def __init__(self):
        self.metrics = {}
        self.step_count = 0

    def update(self, **kwargs):
        """更新指标"""
        self.step_count += 1
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def get_average(self, key, last_n=None):
        """获取指标的平均值"""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return None

        values = self.metrics[key]
        if last_n is not None:
            values = values[-last_n:]

        return sum(values) / len(values)

    def get_latest(self, key):
        """获取最新的指标值"""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return None
        return self.metrics[key][-1]

    def summary(self):
        """生成指标摘要"""
        summary = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                summary[f'{key}_mean'] = sum(values) / len(values)
                summary[f'{key}_min'] = min(values)
                summary[f'{key}_max'] = max(values)
                summary[f'{key}_latest'] = values[-1]
        return summary


def create_logger(log_dir='logs', experiment_name=None, console_level=logging.INFO):
    """
    创建日志记录器的便捷函数

    参数:
        log_dir: 日志保存目录
        experiment_name: 实验名称
        console_level: 控制台输出级别

    返回:
        TrainingLogger 实例
    """
    return TrainingLogger(log_dir=log_dir, experiment_name=experiment_name,
                         console_level=console_level)
