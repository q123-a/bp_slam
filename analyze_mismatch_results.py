#!/usr/bin/env python3
"""
模型失配测试结果分析脚本
从日志文件中提取关键指标并生成对比表格
"""

import re
import os
from pathlib import Path

def extract_metrics_from_log(log_file):
    """从日志文件中提取关键指标"""
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    metrics = {
        'mean_position_error': None,
        'max_position_error': None,
        'final_position_error': None,
        'mean_ospa_sensor1': None,
        'mean_ospa_sensor2': None,
        'num_anchors_sensor1': None,
        'num_anchors_sensor2': None,
    }
    
    # 提取位置误差
    match = re.search(r'平均位置误差.*?:\s*([\d.]+)\s*m', content)
    if match:
        metrics['mean_position_error'] = float(match.group(1))
    
    match = re.search(r'最大位置误差.*?:\s*([\d.]+)\s*m', content)
    if match:
        metrics['max_position_error'] = float(match.group(1))
    
    match = re.search(r'最终位置误差.*?:\s*([\d.]+)\s*m', content)
    if match:
        metrics['final_position_error'] = float(match.group(1))
    
    # 提取OSPA误差
    match = re.search(r'传感器 1 平均OSPA误差:\s*([\d.]+)\s*m', content)
    if match:
        metrics['mean_ospa_sensor1'] = float(match.group(1))
    
    match = re.search(r'传感器 2 平均OSPA误差:\s*([\d.]+)\s*m', content)
    if match:
        metrics['mean_ospa_sensor2'] = float(match.group(1))
    
    # 提取锚点数量（从最后一步）
    matches = re.findall(r'传感器 1.*?锚点数量:\s*(\d+)', content)
    if matches:
        metrics['num_anchors_sensor1'] = int(matches[-1])
    
    matches = re.findall(r'传感器 2.*?锚点数量:\s*(\d+)', content)
    if matches:
        metrics['num_anchors_sensor2'] = int(matches[-1])
    
    return metrics

def compare_tests():
    """对比所有测试结果"""
    
    tests = [
        ('test1_matched', '模型匹配', '0%', '0个'),
        ('test2_very_light', '极轻度失配', '5%', '0.5个'),
        ('test3_light', '轻度失配', '10%', '1个'),
        ('test4_medium', '中度失配', '15%', '2个'),
        ('test5_heavy', '重度失配', '25%', '4个'),
    ]
    
    print("=" * 120)
    print("模型失配测试结果对比分析")
    print("=" * 120)
    print()
    
    # 表头
    print(f"{'测试场景':<15} {'失配程度':<12} {'模式':<6} {'平均位置误差':<12} {'最大位置误差':<12} "
          f"{'OSPA误差(S1)':<12} {'OSPA误差(S2)':<12} {'锚点数(S1)':<10} {'锚点数(S2)':<10}")
    print("-" * 120)
    
    results_summary = []
    
    for test_name, test_desc, det_diff, clutter_diff in tests:
        print(f"\n{test_desc} (检测率偏差:{det_diff}, 杂波偏差:{clutter_diff})")
        print("-" * 120)
        
        # BP模式
        bp_log = f'logs/{test_name}_bp.log'
        bp_metrics = extract_metrics_from_log(bp_log)
        
        # GNN模式
        gnn_log = f'logs/{test_name}_gnn.log'
        gnn_metrics = extract_metrics_from_log(gnn_log)
        
        if bp_metrics:
            print(f"{'  BP':<15} {'':<12} {'BP':<6} "
                  f"{bp_metrics['mean_position_error']:>11.4f} "
                  f"{bp_metrics['max_position_error']:>11.4f} "
                  f"{bp_metrics['mean_ospa_sensor1']:>11.4f} "
                  f"{bp_metrics['mean_ospa_sensor2']:>11.4f} "
                  f"{bp_metrics['num_anchors_sensor1']:>9} "
                  f"{bp_metrics['num_anchors_sensor2']:>9}")
        else:
            print(f"{'  BP':<15} {'':<12} {'BP':<6} {'未找到日志':<50}")
        
        if gnn_metrics:
            print(f"{'  GNN':<15} {'':<12} {'GNN':<6} "
                  f"{gnn_metrics['mean_position_error']:>11.4f} "
                  f"{gnn_metrics['max_position_error']:>11.4f} "
                  f"{gnn_metrics['mean_ospa_sensor1']:>11.4f} "
                  f"{gnn_metrics['mean_ospa_sensor2']:>11.4f} "
                  f"{gnn_metrics['num_anchors_sensor1']:>9} "
                  f"{gnn_metrics['num_anchors_sensor2']:>9}")
        else:
            print(f"{'  GNN':<15} {'':<12} {'GNN':<6} {'未找到日志':<50}")
        
        # 计算改进百分比
        if bp_metrics and gnn_metrics:
            if bp_metrics['mean_position_error'] and gnn_metrics['mean_position_error']:
                improvement = ((bp_metrics['mean_position_error'] - gnn_metrics['mean_position_error']) 
                              / bp_metrics['mean_position_error'] * 100)
                print(f"{'  改进':<15} {'':<12} {'→':<6} "
                      f"{improvement:>10.1f}% "
                      f"{'(GNN优于BP)' if improvement > 0 else '(BP优于GNN)':<50}")
                
                results_summary.append({
                    'test': test_desc,
                    'det_diff': det_diff,
                    'clutter_diff': clutter_diff,
                    'bp_error': bp_metrics['mean_position_error'],
                    'gnn_error': gnn_metrics['mean_position_error'],
                    'improvement': improvement
                })
    
    # 总结
    print()
    print("=" * 120)
    print("总结：GNN相对BP的性能改进")
    print("=" * 120)
    print(f"{'测试场景':<20} {'检测率偏差':<12} {'杂波偏差':<10} {'BP误差(m)':<12} {'GNN误差(m)':<12} {'改进幅度':<15}")
    print("-" * 120)
    
    for result in results_summary:
        print(f"{result['test']:<20} {result['det_diff']:<12} {result['clutter_diff']:<10} "
              f"{result['bp_error']:>11.4f} {result['gnn_error']:>11.4f} "
              f"{result['improvement']:>13.1f}%")
    
    print("=" * 120)
    print()
    
    # 关键发现
    print("关键发现:")
    print("-" * 120)
    
    if results_summary:
        # 找出GNN优势最大的场景
        max_improvement = max(results_summary, key=lambda x: x['improvement'])
        print(f"✓ GNN优势最大: {max_improvement['test']} (改进 {max_improvement['improvement']:.1f}%)")
        
        # 找出GNN优势最小的场景
        min_improvement = min(results_summary, key=lambda x: x['improvement'])
        print(f"✓ GNN优势最小: {min_improvement['test']} (改进 {min_improvement['improvement']:.1f}%)")
        
        # 平均改进
        avg_improvement = sum(r['improvement'] for r in results_summary) / len(results_summary)
        print(f"✓ 平均改进幅度: {avg_improvement:.1f}%")
        
        # 失配程度与改进的关系
        print()
        print("趋势分析:")
        print("  随着模型失配程度增加，GNN相对BP的优势:")
        for result in results_summary:
            bar_length = int(result['improvement'] / 2)  # 缩放到合适长度
            bar = '█' * max(0, bar_length)
            print(f"  {result['test']:<20} {bar} {result['improvement']:>6.1f}%")
    
    print()

if __name__ == '__main__':
    compare_tests()
