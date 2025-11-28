"""
统一的BP-SLAM可视化模块
Unified BP-SLAM Visualization Module

整合了 visualize_results.py 和 plot_matlab_style.py 的功能
支持在线（算法运行时）和离线（从文件加载）两种模式
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path
from .plotting import ospa_dist
from ..utils.distance import calc_distance


# MATLAB风格颜色方案
MATLAB_COLORS = np.array([
    [0.66, 0.00, 0.00],  # 深红色 - 传感器1
    [0.00, 0.30, 0.70],  # 深蓝色 - 传感器2
    [0.60, 0.90, 0.16],  # 绿色
    [0.54, 0.80, 0.99],  # 浅蓝色
    [0.99, 0.34, 0.00],  # 橙色
    [0.92, 0.75, 0.33],  # 黄色
    [0.00, 0.00, 0.00],  # 黑色
])


class BPSLAMVisualizer:
    """
    BP-SLAM统一可视化类

    支持两种使用模式：
    1. 在线模式：算法运行时直接传入数据
    2. 离线模式：从保存的.npz文件加载数据
    """

    def __init__(self, scene_file='scen_semroom_new.mat', output_dir='results'):
        """
        初始化可视化器

        参数:
            scene_file: 场景文件路径（包含房间平面图）
            output_dir: 输出目录
        """
        self.scene_file = scene_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 加载场景数据
        self.s_scen = self._load_scene_data()

    def _load_scene_data(self):
        """加载场景数据（房间布局）"""
        try:
            scene_data = sio.loadmat(self.scene_file)
            s_scen = scene_data.get('s_scen', None)
            if s_scen is not None:
                print(f"   ✓ 成功加载场景文件: {self.scene_file}")
                return s_scen
            else:
                print(f"   ⚠ 场景文件中未找到 's_scen' 字段")
                return None
        except FileNotFoundError:
            print(f"   ⚠ 场景文件不存在: {self.scene_file}")
            return None
        except Exception as e:
            print(f"   ⚠ 加载场景文件失败: {e}")
            return None

    def _plot_floor_plan(self, ax):
        """绘制房间平面图（墙壁）"""
        if self.s_scen is None:
            return

        try:
            # 提取平面图数据
            s_scen_struct = self.s_scen[0, 0]

            # 检查是否有 'fp' 字段（floor plan）
            if 'fp' in s_scen_struct.dtype.names:
                fp = s_scen_struct['fp'][0, 0]

                # 提取墙壁线段数据
                if 'segments' in fp.dtype.names:
                    segments = fp['segments']

                    # 绘制每个墙壁线段
                    for i in range(segments.shape[0]):
                        x1, y1, x2, y2 = segments[i, 0:4]
                        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5, alpha=0.6)

                    return segments.shape[0]
        except Exception as e:
            print(f"   ⚠ 绘制平面图失败: {e}")

        return 0

    def plot_trajectory_and_anchors(self, true_trajectory, estimated_trajectory,
                                   estimated_anchors, posterior_particles_anchors,
                                   data_va, parameters, step_idx=-1, show_particles=True):
        """
        绘制轨迹和锚点估计（图1）

        参数:
            true_trajectory: 真实轨迹 (n_dims, num_steps)
            estimated_trajectory: 估计轨迹 (4, num_steps)
            estimated_anchors: 估计的锚点 [sensor][step][anchor]
            posterior_particles_anchors: 锚点粒子 [sensor][anchor]
            data_va: 虚拟锚点数据
            parameters: 参数字典
            step_idx: 显示哪个时间步（-1表示最后一步）
            show_particles: 是否显示粒子云

        返回:
            fig: matplotlib图形对象
        """
        num_sensors = len(data_va)
        num_steps = true_trajectory.shape[1]

        if step_idx < 0:
            step_idx = num_steps + step_idx

        detection_threshold = parameters.get('detectionThreshold', 0.5)

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 10))

        # 绘制房间平面图
        num_walls = self._plot_floor_plan(ax)
        if num_walls > 0:
            print(f"   ✓ 绘制 {num_walls} 个墙壁线段")

        # 绘制真实轨迹
        ax.plot(true_trajectory[0, :], true_trajectory[1, :],
               '-', color=[0.5, 0.5, 0.5], linewidth=1.5, label='True Trajectory')

        # 绘制每个传感器的锚点
        for sensor in range(num_sensors):
            true_anchor_positions = data_va[sensor]['positions']

            # 绘制真实锚点（方框+叉）
            ax.plot(true_anchor_positions[0, :], true_anchor_positions[1, :],
                   linestyle='none', marker='s', markersize=8,
                   markeredgecolor=MATLAB_COLORS[sensor], markerfacecolor='none',
                   linewidth=1, label=f'True Anchors Sensor {sensor + 1}')
            ax.plot(true_anchor_positions[0, :], true_anchor_positions[1, :],
                   linestyle='none', marker='x', markersize=7.9,
                   markeredgecolor=MATLAB_COLORS[sensor], linewidth=1)

            # 绘制估计的锚点
            if estimated_anchors[sensor][step_idx] is not None:
                for anchor_idx, anchor in enumerate(estimated_anchors[sensor][step_idx]):
                    if anchor is None:
                        continue

                    anchor_pos = anchor['x']
                    anchor_existence = anchor['posteriorExistence']

                    if anchor_existence >= detection_threshold:
                        # 绘制粒子云（如果可用且需要）
                        if show_particles and posterior_particles_anchors is not None:
                            try:
                                if sensor < len(posterior_particles_anchors):
                                    if anchor_idx < len(posterior_particles_anchors[sensor]):
                                        particles = posterior_particles_anchors[sensor][anchor_idx]['x']
                                        ax.scatter(particles[0, :], particles[1, :],
                                                 c=[MATLAB_COLORS[sensor]], alpha=0.3, s=1, marker='.')
                            except:
                                pass

                        # 绘制估计位置（黑色+号）
                        ax.plot(anchor_pos[0], anchor_pos[1],
                               'k+', markersize=8, linewidth=1.5)

        # 绘制估计的当前位置
        ax.plot(estimated_trajectory[0, step_idx], estimated_trajectory[1, step_idx],
               color=[0, 0.5, 0], marker='+', markersize=8, linewidth=1.5,
               label='Estimated Position')

        ax.set_xlabel('x-axis [m]', fontsize=12)
        ax.set_ylabel('y-axis [m]', fontsize=12)
        ax.set_title('BP-SLAM: Trajectory and Anchor Estimation', fontsize=14)
        ax.set_xlim([-7, 15])
        ax.set_ylim([-8, 15.5])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')

        plt.tight_layout()
        return fig

    def plot_ospa_error(self, true_trajectory, estimated_trajectory,
                       estimated_anchors, data_va, parameters):
        """
        绘制OSPA地图误差（图2）

        返回:
            fig: matplotlib图形对象
            dist_ospa_map: OSPA误差矩阵 (num_sensors, num_steps)
        """
        num_sensors = len(data_va)
        num_steps = true_trajectory.shape[1]
        detection_threshold = parameters.get('detectionThreshold', 0.5)

        dist_ospa_map = np.zeros((num_sensors, num_steps))

        # 计算每个时间步的OSPA误差
        for sensor in range(num_sensors):
            true_anchor_positions = data_va[sensor]['positions']

            for step in range(num_steps):
                # 提取估计的锚点位置
                estimated_anchor_positions = []

                if estimated_anchors[sensor][step] is not None:
                    for anchor in estimated_anchors[sensor][step]:
                        if anchor is not None:
                            anchor_pos = anchor['x']
                            anchor_existence = anchor['posteriorExistence']

                            if anchor_existence >= detection_threshold:
                                estimated_anchor_positions.append(anchor_pos)

                if len(estimated_anchor_positions) > 0:
                    estimated_anchor_positions = np.array(estimated_anchor_positions).T
                else:
                    estimated_anchor_positions = np.zeros((2, 0))

                # 计算OSPA距离
                ospa, _, _ = ospa_dist(true_anchor_positions, estimated_anchor_positions, 10, 1)
                dist_ospa_map[sensor, step] = ospa

        # 绘制图形
        fig, ax = plt.subplots(figsize=(12, 6))

        for sensor in range(num_sensors):
            ax.plot(range(1, num_steps + 1), dist_ospa_map[sensor, :],
                   '-', color=MATLAB_COLORS[sensor], linewidth=1.5,
                   label=f'Sensor {sensor + 1}')

        ax.set_xlabel('Trajectory steps', fontsize=12)
        ax.set_ylabel('OSPA map error [m]', fontsize=12)
        ax.set_title('OSPA Distance for Anchor Estimation', fontsize=14)
        ax.set_xlim([0, num_steps])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        plt.tight_layout()
        return fig, dist_ospa_map

    def plot_position_error(self, true_trajectory, estimated_trajectory):
        """
        绘制位置误差（图3）

        返回:
            fig: matplotlib图形对象
            position_error: 位置误差数组 (num_steps,)
        """
        num_steps = true_trajectory.shape[1]
        position_error = np.zeros(num_steps)

        # 计算每个时间步的位置误差
        for step in range(num_steps):
            error = calc_distance(
                true_trajectory[0:2, step:step+1],
                estimated_trajectory[0:2, step:step+1]
            )
            if isinstance(error, np.ndarray):
                position_error[step] = error.item() if error.size == 1 else error[0]
            else:
                position_error[step] = error

        # 绘制图形
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(range(1, num_steps + 1), position_error, 'k-', linewidth=1.5)
        ax.set_xlabel('Trajectory steps', fontsize=12)
        ax.set_ylabel('Position error agent [m]', fontsize=12)
        ax.set_title('Agent Position Estimation Error', fontsize=14)
        ax.set_xlim([0, num_steps])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, position_error

    def visualize_all(self, true_trajectory, estimated_trajectory,
                     estimated_anchors, posterior_particles_anchors,
                     data_va, parameters, save=True, show=False):
        """
        生成所有可视化图表（在线模式）

        参数:
            save: 是否保存图表
            show: 是否显示图表

        返回:
            stats: 误差统计字典
        """
        print("=" * 70)
        print("生成MATLAB风格可视化图表")
        print("=" * 70)
        print()

        # 图1: 轨迹和锚点
        print("1. 生成图1: 轨迹和锚点...")
        fig1 = self.plot_trajectory_and_anchors(
            true_trajectory, estimated_trajectory, estimated_anchors,
            posterior_particles_anchors, data_va, parameters
        )
        if save:
            fig1_path = self.output_dir / 'figure1_trajectory_anchors.png'
            fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ 图1已保存: {fig1_path}")

        # 图2: OSPA误差
        print("2. 生成图2: OSPA地图误差...")
        fig2, dist_ospa_map = self.plot_ospa_error(
            true_trajectory, estimated_trajectory, estimated_anchors,
            data_va, parameters
        )
        if save:
            fig2_path = self.output_dir / 'figure2_ospa_error.png'
            fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ 图2已保存: {fig2_path}")

        # 图3: 位置误差
        print("3. 生成图3: 位置误差...")
        fig3, position_error = self.plot_position_error(
            true_trajectory, estimated_trajectory
        )
        if save:
            fig3_path = self.output_dir / 'figure3_position_error.png'
            fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ 图3已保存: {fig3_path}")

        # 打印统计信息
        print()
        print("=" * 70)
        print("误差统计 (Error Statistics)")
        print("=" * 70)
        print(f"平均位置误差 (Mean Position Error): {np.mean(position_error):.4f} m")
        print(f"最大位置误差 (Max Position Error):  {np.max(position_error):.4f} m")
        print(f"最终位置误差 (Final Position Error): {position_error[-1]:.4f} m")
        print()

        num_sensors = len(data_va)
        for sensor in range(num_sensors):
            print(f"传感器 {sensor + 1} 平均OSPA误差: {np.mean(dist_ospa_map[sensor, :]):.4f} m")
        print("=" * 70)
        print()

        if show:
            plt.show()
        else:
            plt.close('all')

        # 返回统计信息
        stats = {
            'position_error': position_error,
            'mean_position_error': np.mean(position_error),
            'max_position_error': np.max(position_error),
            'final_position_error': position_error[-1],
            'ospa_error': dist_ospa_map,
            'mean_ospa_error': [np.mean(dist_ospa_map[s, :]) for s in range(num_sensors)]
        }

        return stats

    def visualize_from_file(self, results_file='results/results.npz',
                           data_file='scenarioCleanM2_new.mat',
                           save=True, show=False):
        """
        从保存的文件加载并可视化（离线模式）

        参数:
            results_file: 结果文件路径
            data_file: 场景数据文件路径
            save: 是否保存图表
            show: 是否显示图表

        返回:
            stats: 误差统计字典
        """
        print("=" * 70)
        print("从文件加载并生成可视化")
        print("=" * 70)
        print()

        # 加载结果数据
        print(f"1. 加载结果数据: {results_file}...")
        results = np.load(results_file, allow_pickle=True)
        estimated_trajectory = results['estimated_trajectory']
        true_trajectory = results['true_trajectory']
        estimated_anchors = results['estimated_anchors']
        posterior_particles_anchors = results.get('posterior_particles_anchors', None)

        num_steps = estimated_trajectory.shape[1]
        print(f"   时间步数: {num_steps}")

        # 加载真实锚点数据
        print(f"2. 加载场景数据: {data_file}...")
        mat_data = sio.loadmat(data_file)
        data_va_raw = mat_data['dataVA'][:, 0]
        num_sensors = len(data_va_raw)

        data_va = []
        for sensor in range(num_sensors):
            sensor_data = {
                'positions': data_va_raw[sensor]['positions'][0, 0],
            }
            data_va.append(sensor_data)
        print(f"   传感器数量: {num_sensors}")

        # 加载参数
        parameters = results['parameters'].item()

        # 调用在线模式的可视化函数
        print()
        return self.visualize_all(
            true_trajectory, estimated_trajectory, estimated_anchors,
            posterior_particles_anchors, data_va, parameters,
            save=save, show=show
        )


def visualize_online(true_trajectory, estimated_trajectory, estimated_anchors,
                    posterior_particles_anchors, data_va, parameters,
                    scene_file='scen_semroom_new.mat', output_dir='results',
                    save=True, show=False):
    """
    在线可视化（算法运行时调用）

    这是一个便捷函数，用于在 testbed.py 中调用
    """
    visualizer = BPSLAMVisualizer(scene_file=scene_file, output_dir=output_dir)
    return visualizer.visualize_all(
        true_trajectory, estimated_trajectory, estimated_anchors,
        posterior_particles_anchors, data_va, parameters,
        save=save, show=show
    )


def visualize_offline(results_file='results/results.npz',
                     data_file='scenarioCleanM2_new.mat',
                     scene_file='scen_semroom_new.mat',
                     output_dir='results',
                     save=True, show=False):
    """
    离线可视化（从文件加载）

    这是一个便捷函数，用于独立运行可视化
    """
    visualizer = BPSLAMVisualizer(scene_file=scene_file, output_dir=output_dir)
    return visualizer.visualize_from_file(
        results_file=results_file,
        data_file=data_file,
        save=save, show=show
    )


if __name__ == '__main__':
    # 独立运行时，使用离线模式
    import sys
    import os

    # 确定结果文件路径
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        # 按优先级查找结果文件
        if os.path.exists('results/results.npz'):
            results_file = 'results/results.npz'
        elif os.path.exists('results/results_quick.npz'):
            results_file = 'results/results_quick.npz'
        elif os.path.exists('results.npz'):
            results_file = 'results.npz'
        else:
            results_file = 'results/results.npz'

    # 检查文件是否存在
    if not Path(results_file).exists():
        print(f"错误: 结果文件不存在: {results_file}")
        print("请先运行 testbed.py 生成结果文件")
        sys.exit(1)

    # 运行离线可视化
    stats = visualize_offline(results_file=results_file, show=True)
