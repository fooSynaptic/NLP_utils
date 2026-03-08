"""
可视化 benchmark 结果
"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np


def plot_speedup(results_file):
    """绘制速度对比图"""
    with open(results_file) as f:
        data = json.load(f)
    
    # 收集所有结果
    names = []
    speedups = []
    
    for r in data.get('coding', []):
        names.append(r['name'])
        speedups.append(r['speedup'])
    
    for r in data.get('ml', []):
        names.append(r['name'] + " (train)")
        speedups.append(r['train_speedup'])
        names.append(r['name'] + " (predict)")
        speedups.append(r['predict_speedup'])
    
    # 绘制
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['green' if s > 1 else 'red' for s in speedups]
    bars = ax.barh(names, speedups, color=colors, alpha=0.7)
    
    # 添加参考线
    ax.axvline(x=1, color='black', linestyle='--', linewidth=2, label='Equal Performance')
    
    ax.set_xlabel('Speedup (vs Competitor)')
    ax.set_title('NLP_utils Benchmark: Speedup Comparison')
    ax.legend()
    
    # 添加数值标签
    for bar, speedup in zip(bars, speedups):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{speedup:.2f}x', 
                ha='left' if width > 0 else 'right', va='center')
    
    plt.tight_layout()
    output_file = 'benchmark_speedup.png'
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    
    return output_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <results.json>")
        sys.exit(1)
    
    plot_speedup(sys.argv[1])
