"""
生成与MATLAB相同的可视化图表
Generate MATLAB-style visualization plots
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from bp_slam.visualization.plotting import ospa_dist
from bp_slam.utils.distance import calc_distance

# 创建results文件夹
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

# MATLAB风格颜色
mycolors = np.array([
    [0.66, 0.00, 0.00],  # 深红色 - 传感器1
    [0.00, 0.30, 0.70],  # 深蓝色 - 传感器2
    [0.60, 0.90, 0.16],  # 绿色
    [0.54, 0.80, 0.99],  # 浅蓝色
    [0.99, 0.34, 0.00],  # 橙色
    [0.92, 0.75, 0.33],  # 黄色
    [0.00, 0.00, 0.00],  # 黑色
])

def load_scene_data(scene_file='scen_semroom_new.mat'):
    """加载场景数据（房间布局）"""
    try:
        scene_data = sio.loadmat(scene_file)
        s_scen = scene_data.get('s_scen', None)
        if s_scen is not None:
            print(f"   ✓ 成功加载场景文件: {scene_file}")
            return s_scen
        else:
            print(f"   ⚠ 场景文件中未找到 's_scen' 字段")
            return None
    except FileNotFoundError:
        print(f"   ⚠ 场景文件不存在: {scene_file}")
        return None
    except Exception as e:
        print(f"   ⚠ 加载场景文件失败: {e}")
        return None

def plot_floor_plan(s_scen, ax):
    """绘制房间平面图（墙壁）"""
    if s_scen is None:
        print("   ⚠ 未加载场景数据，跳过环境地图绘制")
        return

    try:
        # 提取平面图数据
        s_scen_struct = s_scen[0, 0]

        # 检查是否有 'fp' 字段（floor plan）
        if 'fp' in s_scen_struct.dtype.names:
            fp = s_scen_struct['fp'][0, 0]

            # 提取墙壁线段数据
            if 'segments' in fp.dtype.names:
                segments = fp['segments']
                print(f"   ✓ 绘制 {segments.shape[0]} 个墙壁线段")

                # 绘制每个墙壁线段
                # segments格式: [x1, y1, x2, y2, ?, ?]
                for i in range(segments.shape[0]):
                    x1, y1, x2, y2 = segments[i, 0:4]
                    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5, alpha=0.6)
            else:
                print("   ⚠ 场景数据中未找到 'segments' 字段")
        else:
            print("   ⚠ 场景数据中未找到 'fp' 字段")

    except Exception as e:
        print(f"   ⚠ 绘制平面图失败: {e}")
        import traceback
        traceback.print_exc()

def visualize_trajectory_and_anchors(results_file='results/results.npz',
                                    data_file='scenarioCleanM2_new.mat',
                                    scene_file='scen_semroom_new.mat'):
    """
    生成图1: 轨迹和锚点可视化（与MATLAB Figure 1相同）
    """
    print("=" * 70)
    print("生成MATLAB风格可视化图表")
    print("=" * 70)
    print()

    # 加载结果数据
    print("1. 加载结果数据...")
    results = np.load(results_file, allow_pickle=True)
    estimated_trajectory = results['estimated_trajectory']
    true_trajectory = results['true_trajectory']
    estimated_anchors = results['estimated_anchors']
    posterior_particles_anchors = results['posterior_particles_anchors']
    num_estimated_anchors = results['num_estimated_anchors']

    num_steps = estimated_trajectory.shape[1]
    print(f"   时间步数: {num_steps}")

    # 加载真实锚点数据
    print("2. 加载真实锚点数据...")
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
    detection_threshold = parameters.get('detectionThreshold', 0.5)

    # 加载场景数据
    print("3. 加载场景数据...")
    s_scen = load_scene_data(scene_file)

    # ========================================
    # 图1: 轨迹和锚点（完全按照MATLAB格式）
    # ========================================
    print("4. 生成图1: 轨迹和锚点...")
    fig1 = plt.figure(1, figsize=(12, 10))
    ax1 = fig1.add_subplot(111)

    # 绘制房间平面图（环境地图）
    plot_floor_plan(s_scen, ax1)

    # 绘制真实轨迹（灰色，linewidth=1.5）
    ax1.plot(true_trajectory[0, :], true_trajectory[1, :],
            '-', color=[0.5, 0.5, 0.5], linewidth=1.5, label='True Trajectory')

    # 绘制估计轨迹（绿色虚线，linewidth=1.5）
    ax1.plot(estimated_trajectory[0, :], estimated_trajectory[1, :],
            '--', color=[0, 0.5, 0], linewidth=1.5, label='Estimated Trajectory')

    # 最后一步
    final_step = num_steps - 1

    # 绘制每个传感器的锚点
    for sensor in range(num_sensors):
        # 真实锚点位置
        true_anchor_positions = data_va[sensor]['positions']

        # 绘制真实锚点（方框+叉，linewidth=1）
        ax1.plot(true_anchor_positions[0, :], true_anchor_positions[1, :],
                linestyle='none', linewidth=1, color=mycolors[sensor],
                marker='s', markersize=8, markeredgecolor=mycolors[sensor])
        ax1.plot(true_anchor_positions[0, :], true_anchor_positions[1, :],
                linestyle='none', linewidth=1, color=mycolors[sensor],
                marker='x', markersize=7.9, markeredgecolor=mycolors[sensor])

        # 绘制估计的锚点
        if estimated_anchors[sensor][final_step] is not None:
            for anchor_idx, anchor in enumerate(estimated_anchors[sensor][final_step]):
                if anchor is not None:
                    anchor_pos = anchor['x']
                    anchor_existence = anchor['posteriorExistence']

                    if anchor_existence >= detection_threshold:
                        # 绘制粒子云（MATLAB风格：plotScatter2d，alpha=0.3）
                        try:
                            if isinstance(posterior_particles_anchors[sensor], dict):
                                if anchor_idx in posterior_particles_anchors[sensor]:
                                    particles = posterior_particles_anchors[sensor][anchor_idx]['x']
                                    ax1.scatter(particles[0, :], particles[1, :],
                                              c=[mycolors[sensor]], alpha=0.3, s=1, marker='.')
                            elif anchor_idx < len(posterior_particles_anchors[sensor]):
                                particles = posterior_particles_anchors[sensor][anchor_idx]['x']
                                ax1.scatter(particles[0, :], particles[1, :],
                                          c=[mycolors[sensor]], alpha=0.3, s=1, marker='.')
                        except:
                            pass

                        # 绘制估计的锚点位置（黑色+号，markersize=8）
                        ax1.plot(anchor_pos[0], anchor_pos[1],
                                color='k', marker='+', markersize=8)

    # 绘制估计的当前位置（绿色+号，markersize=8, linewidth=1.5）
    ax1.plot(estimated_trajectory[0, final_step], estimated_trajectory[1, final_step],
            color=[0, 0.5, 0], marker='+', markersize=8, linewidth=1.5)

    ax1.set_xlabel('xaxis', fontsize=12)
    ax1.set_ylabel('yaxis', fontsize=12)
    ax1.set_xlim([-7, 15])
    ax1.set_ylim([-8, 15.5])
    ax1.set_aspect('equal')
    ax1.legend(loc='best', fontsize=10)
    ax1.set_title('BP-SLAM: Trajectory and Anchor Estimation', fontsize=14)

    # 保存图1
    fig1_path = results_dir / 'figure1_trajectory_anchors.png'
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ 图1已保存: {fig1_path}")

    # ========================================
    # 计算OSPA误差和位置误差
    # ========================================
    print("5. 计算误差指标...")
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

                        if anchor_existence >= detection_threshold:
                            estimated_anchor_positions.append(anchor_pos)

                if len(estimated_anchor_positions) > 0:
                    estimated_anchor_positions = np.array(estimated_anchor_positions).T
                else:
                    estimated_anchor_positions = np.zeros((2, 0))
            else:
                estimated_anchor_positions = np.zeros((2, 0))

            # 计算位置误差（只使用位置坐标，不包括速度）
            error = calc_distance(true_trajectory[:, step], estimated_trajectory[0:2, step])
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
    print("6. 生成图2: OSPA地图误差...")
    fig2 = plt.figure(2, figsize=(12, 6))
    ax2 = fig2.add_subplot(111)

    for sensor in range(num_sensors):
        ax2.plot(range(1, num_steps + 1), dist_ospa_map[sensor, :],
                '-', color=mycolors[sensor], linewidth=1.5,
                label=f'Sensor {sensor + 1}')

    ax2.set_xlabel('Trajectory steps', fontsize=12)
    ax2.set_ylabel('OSPA map error [m]', fontsize=12)
    ax2.set_title('OSPA Distance for Anchor Estimation', fontsize=14)
    ax2.set_xlim([0, num_steps])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # 保存图2
    fig2_path = results_dir / 'figure2_ospa_error.png'
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ 图2已保存: {fig2_path}")

    # ========================================
    # 图3: 位置误差
    # ========================================
    print("7. 生成图3: 位置误差...")
    fig3 = plt.figure(3, figsize=(12, 6))
    ax3 = fig3.add_subplot(111)

    ax3.plot(range(1, num_steps + 1), position_error_agent, 'k-', linewidth=1.5)
    ax3.set_xlabel('Trajectory steps', fontsize=12)
    ax3.set_ylabel('Position error agent [m]', fontsize=12)
    ax3.set_title('Agent Position Error', fontsize=14)
    ax3.set_xlim([0, num_steps])
    ax3.grid(True, alpha=0.3)

    # 保存图3
    fig3_path = results_dir / 'figure3_position_error.png'
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ 图3已保存: {fig3_path}")

    # ========================================
    # 打印统计信息
    # ========================================
    print()
    print("=" * 70)
    print("误差统计 (Error Statistics)")
    print("=" * 70)
    print(f"平均位置误差 (Mean Position Error): {np.mean(position_error_agent):.4f} m")
    print(f"最大位置误差 (Max Position Error):  {np.max(position_error_agent):.4f} m")
    print(f"最终位置误差 (Final Position Error): {position_error_agent[-1]:.4f} m")
    print()
    for sensor in range(num_sensors):
        print(f"传感器 {sensor + 1} 平均OSPA误差: {np.mean(dist_ospa_map[sensor, :]):.4f} m")
    print("=" * 70)
    print()

    print("所有图表已保存到 results/ 文件夹:")
    print(f"  - {fig1_path}")
    print(f"  - {fig2_path}")
    print(f"  - {fig3_path}")
    print()

    # 关闭所有图表（不显示）
    plt.close('all')

    return dist_ospa_map, position_error_agent

if __name__ == '__main__':
    # 使用快速测试结果（如果存在），否则使用完整测试结果
    import os
    if os.path.exists('results/results_quick.npz'):
        results_file = 'results/results_quick.npz'
    elif os.path.exists('results/results.npz'):
        results_file = 'results/results.npz'
    else:
        # 兼容旧的文件位置
        if os.path.exists('results_quick.npz'):
            results_file = 'results_quick.npz'
        else:
            results_file = 'results.npz'

    visualize_trajectory_and_anchors(
        results_file=results_file,
        data_file='scenarioCleanM2_new.mat',
        scene_file='scen_semroom_new.mat'
    )
