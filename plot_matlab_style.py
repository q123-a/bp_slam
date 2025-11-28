"""
MATLAB风格的BP-SLAM结果可视化
完全复制MATLAB plotAll.m的输出效果

使用方法：
在testbed.py或testbed_quick.py的最后调用此函数
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from bp_slam.visualization.plotting import ospa_dist
from bp_slam.utils.distance import calc_distance


def plot_all_matlab_style(true_trajectory, estimated_trajectory, estimated_anchors,
                          posterior_particles_anchors, num_estimated_anchors,
                          data_va, parameters, mode=0, num_steps=None):
    """
    MATLAB风格的完整可视化，生成3个图表

    参数:
        true_trajectory: 真实轨迹 (n_dims, num_steps)
        estimated_trajectory: 估计轨迹 (4, num_steps)
        estimated_anchors: 估计的锚点
        posterior_particles_anchors: 锚点粒子
        num_estimated_anchors: 锚点数量 (num_sensors, num_steps)
        data_va: 虚拟锚点数据
        parameters: 参数字典
        mode: 绘图模式 (0=最终状态, 1=动画)
        num_steps: 时间步数
    """

    if num_steps is None:
        num_steps = true_trajectory.shape[1]

    num_sensors = len(data_va)

    # MATLAB颜色方案
    mycolors = np.array([
        [0.66, 0.00, 0.00],  # 红色 - 传感器1
        [0.00, 0.30, 0.70],  # 蓝色 - 传感器2
        [0.60, 0.90, 0.16],  # 绿色
        [0.54, 0.80, 0.99],  # 浅蓝
        [0.99, 0.34, 0.00],  # 橙色
        [0.92, 0.75, 0.33],  # 黄色
        [0.00, 0.00, 0.00],  # 黑色
    ])

    # ========================================
    # 图1: 轨迹和锚点可视化
    # ========================================
    fig1 = plt.figure(1, figsize=(12, 10))
    plt.clf()

    # 根据mode决定显示哪个时间步
    if mode == 0:
        tmp = num_steps - 1  # Python索引从0开始
    else:
        tmp = num_steps - 1

    # 绘制真实轨迹（灰色）
    plt.plot(true_trajectory[0, :], true_trajectory[1, :], '-',
            color=[0.5, 0.5, 0.5], linewidth=1.5, label='True Trajectory')

    # 遍历每个传感器
    for sensor in range(num_sensors):
        true_anchor_positions = data_va[sensor]['positions']
        num_positions = true_anchor_positions.shape[1]

        # 绘制真实锚点位置（方块+叉号）
        plt.plot(true_anchor_positions[0, :], true_anchor_positions[1, :],
                linestyle='none', marker='s', markersize=8,
                color=mycolors[sensor], markeredgecolor=mycolors[sensor])
        plt.plot(true_anchor_positions[0, :], true_anchor_positions[1, :],
                linestyle='none', marker='x', markersize=7.9,
                color=mycolors[sensor], markeredgecolor=mycolors[sensor])

        # 绘制估计的锚点
        if estimated_anchors[sensor][tmp] is not None:
            num_positions = len(estimated_anchors[sensor][tmp])

            for anchor in range(num_positions):
                if estimated_anchors[sensor][tmp][anchor] is None:
                    continue

                anchor_pos = estimated_anchors[sensor][tmp][anchor]['x']
                anchor_existence = estimated_anchors[sensor][tmp][anchor]['posteriorExistence']

                # 只绘制存在概率高于阈值的锚点
                if anchor_existence > parameters['detectionThreshold']:
                    # 绘制锚点粒子云（散点）
                    if (posterior_particles_anchors is not None and
                        sensor < len(posterior_particles_anchors) and
                        anchor < len(posterior_particles_anchors[sensor])):
                        particles = posterior_particles_anchors[sensor][anchor]['x']
                        plt.scatter(particles[0, :], particles[1, :],
                                  c=[mycolors[sensor]], marker='.',
                                  s=1, alpha=0.3)

                    # 绘制锚点估计位置（黑色+号）
                    plt.plot(anchor_pos[0], anchor_pos[1],
                            color='k', marker='+', markersize=8)

    # 绘制估计的当前位置（绿色+号）
    plt.plot(estimated_trajectory[0, tmp], estimated_trajectory[1, tmp],
            color=[0, 0.5, 0], marker='+', markersize=8, linewidth=1.5,
            label='Estimated Position')

    plt.xlabel('x-axis [m]', fontsize=12)
    plt.ylabel('y-axis [m]', fontsize=12)
    plt.title('BP-SLAM: Trajectory and Anchors', fontsize=14)
    plt.axis([-7, 15, -8, 15.5])  # MATLAB原始坐标范围
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # ========================================
    # 计算误差指标
    # ========================================
    position_error_agent = np.zeros(num_steps)
    dist_ospa_map = np.zeros((num_sensors, num_steps))

    for sensor in range(num_sensors):
        true_anchor_positions = data_va[sensor]['positions']

        for step in range(num_steps):
            # 提取估计的锚点位置
            if estimated_anchors[sensor][step] is not None:
                num_anchors_step = len(estimated_anchors[sensor][step])
                estimated_anchor_positions = []

                for anchor in range(num_anchors_step):
                    if estimated_anchors[sensor][step][anchor] is not None:
                        anchor_pos = estimated_anchors[sensor][step][anchor]['x']
                        anchor_existence = estimated_anchors[sensor][step][anchor]['posteriorExistence']

                        # 只包含存在概率高于阈值的锚点
                        if anchor_existence >= parameters['detectionThreshold']:
                            estimated_anchor_positions.append(anchor_pos)

                if len(estimated_anchor_positions) > 0:
                    estimated_anchor_positions = np.array(estimated_anchor_positions).T
                else:
                    estimated_anchor_positions = np.zeros((2, 0))
            else:
                estimated_anchor_positions = np.zeros((2, 0))

            # 计算位置误差
            error = calc_distance(
                true_trajectory[0:2, step:step+1],
                estimated_trajectory[0:2, step:step+1]
            )
            if isinstance(error, np.ndarray):
                position_error_agent[step] = error.item() if error.size == 1 else error[0]
            else:
                position_error_agent[step] = error

            # 计算OSPA距离
            ospa, _, _ = ospa_dist(true_anchor_positions, estimated_anchor_positions, 10, 1)
            dist_ospa_map[sensor, step] = ospa

    # ========================================
    # 图2: OSPA地图误差
    # ========================================
    fig2 = plt.figure(2, figsize=(12, 6))
    plt.clf()

    for sensor in range(num_sensors):
        plt.plot(range(1, num_steps + 1), dist_ospa_map[sensor, :],
                '-', color=mycolors[sensor], linewidth=1.5,
                label=f'Sensor {sensor + 1}')

    plt.xlabel('Trajectory steps', fontsize=12)
    plt.ylabel('OSPA map error [m]', fontsize=12)
    plt.title('OSPA Distance for Anchor Estimation', fontsize=14)
    plt.xlim([0, num_steps])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # ========================================
    # 图3: 位置误差
    # ========================================
    fig3 = plt.figure(3, figsize=(12, 6))
    plt.clf()

    plt.plot(range(1, num_steps + 1), position_error_agent, 'k-', linewidth=1.5)

    plt.xlabel('Trajectory steps', fontsize=12)
    plt.ylabel('Position error agent [m]', fontsize=12)
    plt.title('Agent Position Estimation Error', fontsize=14)
    plt.xlim([0, num_steps])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # ========================================
    # 打印统计信息
    # ========================================
    print("\n" + "="*60)
    print("误差统计 (Error Statistics)")
    print("="*60)
    print(f"平均位置误差 (Mean Position Error): {np.mean(position_error_agent):.4f} m")
    print(f"最大位置误差 (Max Position Error):  {np.max(position_error_agent):.4f} m")
    print(f"最终位置误差 (Final Position Error): {position_error_agent[-1]:.4f} m")
    print()
    for sensor in range(num_sensors):
        print(f"传感器 {sensor + 1} 平均OSPA误差: {np.mean(dist_ospa_map[sensor, :]):.4f} m")
    print("="*60 + "\n")

    # 保存图表
    fig1.savefig('figure1_trajectory_anchors.png', dpi=300, bbox_inches='tight')
    fig2.savefig('figure2_ospa_error.png', dpi=300, bbox_inches='tight')
    fig3.savefig('figure3_position_error.png', dpi=300, bbox_inches='tight')

    print("图表已保存:")
    print("  - figure1_trajectory_anchors.png")
    print("  - figure2_ospa_error.png")
    print("  - figure3_position_error.png")

    return fig1, fig2, fig3


def main():
    """独立运行：从保存的结果文件加载并可视化"""

    print("="*60)
    print("MATLAB风格可视化")
    print("="*60)

    # 加载结果
    result_files = ['results.npz', 'results_quick.npz']
    data = None

    for filename in result_files:
        try:
            print(f"\n尝试加载 {filename}...")
            data = np.load(filename, allow_pickle=True)
            print(f"✓ 成功加载 {filename}")
            break
        except FileNotFoundError:
            continue

    if data is None:
        print("\n错误：未找到结果文件！")
        print("请先运行 testbed.py 或 testbed_quick.py")
        return

    # 提取数据
    true_trajectory = data['true_trajectory']
    estimated_trajectory = data['estimated_trajectory']
    num_estimated_anchors = data['num_estimated_anchors']

    # 需要重新加载原始数据以获取锚点信息
    print("\n加载场景数据...")
    mat_data = sio.loadmat('scenarioCleanM2_new.mat')
    data_va_raw = mat_data['dataVA'][:, 0]  # 修复：获取所有传感器数据

    # 转换数据格式
    data_va = []
    for sensor in range(len(data_va_raw)):
        sensor_data = {
            'positions': data_va_raw[sensor]['positions'][0, 0],
            'visibility': np.ones((data_va_raw[sensor]['positions'][0, 0].shape[1],
                                  true_trajectory.shape[1]))
        }
        data_va.append(sensor_data)

    # 参数
    parameters = {
        'detectionThreshold': 0.5,
    }

    # 注意：完整可视化需要estimated_anchors和posterior_particles_anchors
    # 这些数据较大，通常不保存在npz文件中
    # 这里我们只绘制简化版本

    print("\n注意：完整的MATLAB风格可视化需要运行时的锚点数据")
    print("建议在testbed.py中直接调用plot_all_matlab_style()函数")
    print("\n当前仅显示误差图表...")

    # 简化版：只绘制误差图
    num_steps = true_trajectory.shape[1]
    num_sensors = len(data_va)

    # 计算位置误差
    position_error = np.sqrt(np.sum((true_trajectory[0:2, :] - estimated_trajectory[0:2, :])**2, axis=0))

    # 图3: 位置误差
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_steps + 1), position_error, 'k-', linewidth=1.5)
    plt.xlabel('Trajectory steps', fontsize=12)
    plt.ylabel('Position error agent [m]', fontsize=12)
    plt.title('Agent Position Estimation Error', fontsize=14)
    plt.xlim([0, num_steps])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure3_position_error.png', dpi=300, bbox_inches='tight')

    print(f"\n平均位置误差: {np.mean(position_error):.4f} m")
    print(f"最大位置误差: {np.max(position_error):.4f} m")
    print(f"最终位置误差: {position_error[-1]:.4f} m")

    plt.show()


if __name__ == '__main__':
    main()
