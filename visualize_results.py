# -*- coding: utf-8 -*-
"""
RAG System Evaluation Results Visualization
生成实验结果的可视化图表
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 设置英文字体
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 设置图表样式
plt.style.use('seaborn-v0_8-darkgrid')

# 颜色方案
COLORS = {
    'qwen3_8b': '#FF6B6B',
    'deepseek': '#4ECDC4',
    'rag': '#45B7D1',
    'bruteforce': '#96CEB4',
    'no_context': '#FFEAA7'
}

def load_results():
    """加载实验结果"""
    with open('evaluation_results_exp1.json', 'r', encoding='utf-8') as f:
        exp1 = json.load(f)
    with open('evaluation_results_exp2.json', 'r', encoding='utf-8') as f:
        exp2 = json.load(f)
    return exp1, exp2

def plot_experiment1(exp1):
    """绘制实验1：模型对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 提取数据
    qwen_times = [item['response_time'] for item in exp1['qwen3_8b']]
    deepseek_times = [item['response_time'] for item in exp1['deepseek']]
    queries = [f'Q{i+1}' for i in range(len(exp1['qwen3_8b']))]

    x = np.arange(len(queries))
    width = 0.35

    # 响应时间对比
    bars1 = ax1.bar(x - width/2, qwen_times, width, label='qwen3:8b', color=COLORS['qwen3_8b'])
    bars2 = ax1.bar(x + width/2, deepseek_times, width, label='DeepSeek V3.2', color=COLORS['deepseek'])

    ax1.set_xlabel('Question', fontsize=12)
    ax1.set_ylabel('Response Time (s)', fontsize=12)
    ax1.set_title('Experiment 1: Model Response Time Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(queries, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=9)

    # 平均响应时间对比
    avg_qwen = np.mean(qwen_times)
    avg_deepseek = np.mean(deepseek_times)
    models = ['qwen3:8b', 'DeepSeek V3.2']
    avg_times = [avg_qwen, avg_deepseek]

    bars3 = ax2.bar(models, avg_times, color=[COLORS['qwen3_8b'], COLORS['deepseek']])
    ax2.set_ylabel('Avg Response Time (s)', fontsize=12)
    ax2.set_title('Average Response Time Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 添加数值标签和加速比
    for i, (bar, time) in enumerate(zip(bars3, avg_times)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 添加加速比标注
    speedup = (avg_qwen / avg_deepseek - 1) * 100
    speedup_text = f'DeepSeek is {speedup:.1f}% faster'
    ax2.text(0.5, max(avg_times) * 0.9,
             speedup_text,
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('experiment1_model_comparison.png', dpi=300, bbox_inches='tight')
    print("[OK] Experiment 1 chart saved: experiment1_model_comparison.png")
    plt.close()

def plot_experiment2(exp2):
    """绘制实验2：RAG vs Brute Force vs No Context"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 提取数据
    rag_times = [item['response_time'] for item in exp2['rag']]
    bf_times = [item['response_time'] for item in exp2['bruteforce']]
    nc_times = [item['response_time'] for item in exp2['no_context']]
    queries = [f'Q{i+1}' for i in range(len(exp2['rag']))]

    x = np.arange(len(queries))
    width = 0.25

    # 响应时间对比
    bars1 = ax1.bar(x - width, rag_times, width, label='RAG', color=COLORS['rag'])
    bars2 = ax1.bar(x, bf_times, width, label='Brute Force', color=COLORS['bruteforce'])
    bars3 = ax1.bar(x + width, nc_times, width, label='No Context', color=COLORS['no_context'])

    ax1.set_xlabel('Question', fontsize=12)
    ax1.set_ylabel('Response Time (s)', fontsize=12)
    ax1.set_title('Experiment 2: Strategy Response Time Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(queries, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 平均响应时间和上下文长度对比
    avg_rag_time = np.mean(rag_times)
    avg_bf_time = np.mean(bf_times)
    avg_nc_time = np.mean(nc_times)

    avg_rag_ctx = np.mean([item['context_length'] for item in exp2['rag']])
    avg_bf_ctx = np.mean([item['context_length'] for item in exp2['bruteforce']])
    avg_nc_ctx = np.mean([item['context_length'] for item in exp2['no_context']])

    strategies = ['RAG', 'Brute Force', 'No Context']
    avg_times = [avg_rag_time, avg_bf_time, avg_nc_time]
    avg_ctx = [avg_rag_ctx, avg_bf_ctx, avg_nc_ctx]

    x2 = np.arange(len(strategies))
    width2 = 0.35

    bars4 = ax2.bar(x2 - width2/2, avg_times, width2, label='Avg Response Time (s)', color='#3498db')
    ax2_twin = ax2.twinx()
    bars5 = ax2_twin.bar(x2 + width2/2, avg_ctx, width2, label='Avg Context Length', color='#e74c3c', alpha=0.7)

    ax2.set_xlabel('Strategy', fontsize=12)
    ax2.set_ylabel('Avg Response Time (s)', fontsize=12)
    ax2.set_title('Average Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(strategies)
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2_twin.set_ylabel('Avg Context Length', fontsize=12)

    # 添加数值标签
    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
    for bar in bars5:
        height = bar.get_height()
        ax2_twin.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('experiment2_strategy_comparison.png', dpi=300, bbox_inches='tight')
    print("[OK] Experiment 2 chart saved: experiment2_strategy_comparison.png")
    plt.close()

def plot_comprehensive(exp1, exp2):
    """绘制综合对比图"""
    fig = plt.figure(figsize=(18, 10))

    # 创建子图布局
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. 模型响应时间分布（箱线图）
    ax1 = fig.add_subplot(gs[0, 0])
    qwen_times = [item['response_time'] for item in exp1['qwen3_8b']]
    deepseek_times = [item['response_time'] for item in exp1['deepseek']]
    rag_times = [item['response_time'] for item in exp2['rag']]
    bf_times = [item['response_time'] for item in exp2['bruteforce']]
    nc_times = [item['response_time'] for item in exp2['no_context']]

    data_to_plot = [qwen_times, deepseek_times, rag_times, bf_times, nc_times]
    labels = ['qwen3:8b', 'DeepSeek', 'RAG', 'Brute Force', 'No Context']
    colors = [COLORS['qwen3_8b'], COLORS['deepseek'], COLORS['rag'],
              COLORS['bruteforce'], COLORS['no_context']]

    bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True,
                     medianprops={'linewidth': 2, 'color': 'black'},
                     boxprops={'linewidth': 1.5},
                     whiskerprops={'linewidth': 1.5},
                     capprops={'linewidth': 1.5})

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax1.set_ylabel('Response Time (s)', fontsize=12)
    ax1.set_title('Response Time Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. 上下文长度对比
    ax2 = fig.add_subplot(gs[0, 1])
    ctx_data = [
        [item['context_length'] for item in exp1['qwen3_8b']],
        [item['context_length'] for item in exp1['deepseek']],
        [item['context_length'] for item in exp2['rag']],
        [item['context_length'] for item in exp2['bruteforce']],
        [0] * len(exp2['no_context'])
    ]

    bp2 = ax2.boxplot(ctx_data, labels=labels, patch_artist=True,
                      medianprops={'linewidth': 2, 'color': 'black'},
                      boxprops={'linewidth': 1.5},
                      whiskerprops={'linewidth': 1.5},
                      capprops={'linewidth': 1.5})

    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_ylabel('Context Length', fontsize=12)
    ax2.set_title('Context Length Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 性能雷达图
    ax3 = fig.add_subplot(gs[1, 0], projection='polar')

    categories = ['Response Speed', 'Context Efficiency', 'Stability', 'Resource Usage']
    N = len(categories)

    # 计算各项指标（归一化到0-1）
    def normalize(values, inverse=False):
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return [0.5] * len(values)
        if inverse:
            return [(max_val - v) / (max_val - min_val) for v in values]
        return [(v - min_val) / (max_val - min_val) for v in values]

    # 响应速度（时间越短越好）
    times = [np.mean(qwen_times), np.mean(deepseek_times),
             np.mean(rag_times), np.mean(bf_times), np.mean(nc_times)]
    speed_scores = normalize(times, inverse=True)

    # 上下文效率（长度适中最好，太长太长都不好）
    ctx_lengths = [np.mean([item['context_length'] for item in exp1['qwen3_8b']]),
                   np.mean([item['context_length'] for item in exp1['deepseek']]),
                   np.mean([item['context_length'] for item in exp2['rag']]),
                   np.mean([item['context_length'] for item in exp2['bruteforce']]),
                   0]
    # 上下文效率：RAG最佳（有上下文但不太长），暴力塞文档次之，无上下文最差
    ctx_scores = [0.7, 0.7, 0.9, 0.3, 0.1]

    # 稳定性（方差越小越好）
    stds = [np.std(qwen_times), np.std(deepseek_times),
            np.std(rag_times), np.std(bf_times), np.std(nc_times)]
    stability_scores = normalize(stds, inverse=True)

    # 资源利用（综合评分）
    resource_scores = [0.6, 0.8, 0.7, 0.4, 0.9]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax3.set_theta_offset(np.pi / 2)
    ax3.set_theta_direction(-1)

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, fontsize=10)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax3.grid(True)

    # 绘制雷达图
    for i, (label, color) in enumerate(zip(labels, colors)):
        values = [speed_scores[i], ctx_scores[i], stability_scores[i], resource_scores[i]]
        values += values[:1]
        ax3.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
        ax3.fill(angles, values, alpha=0.15, color=color)

    ax3.set_title('Comprehensive Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    # 4. 关键指标汇总表
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # 计算关键指标
    metrics = {
        'Model/Strategy': labels,
        'Avg Response Time (s)': [f'{np.mean(t):.2f}' for t in [qwen_times, deepseek_times, rag_times, bf_times, nc_times]],
        'Std Dev (s)': [f'{np.std(t):.2f}' for t in [qwen_times, deepseek_times, rag_times, bf_times, nc_times]],
        'Avg Context Length': [f'{int(np.mean(c))}' for c in ctx_data],
        'Overall Score': [
            f'{(speed_scores[i] * 0.3 + ctx_scores[i] * 0.3 + stability_scores[i] * 0.2 + resource_scores[i] * 0.2):.2f}'
            for i in range(5)
        ]
    }

    table = ax4.table(cellText=list(metrics.values()),
                      colLabels=list(metrics.keys()),
                      cellLoc='center',
                      loc='center',
                      colColours=['#f0f0f0'] * 5)

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # 设置表头样式
    for i in range(5):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 设置行颜色
    for i in range(1, 6):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f9f9f9')

    ax4.set_title('Key Metrics Summary', fontsize=14, fontweight='bold', pad=20)

    plt.suptitle('RAG System Performance Evaluation', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("[OK] Comprehensive analysis chart saved: comprehensive_analysis.png")
    plt.close()

def main():
    """主函数"""
    print("=" * 80)
    print("RAG System Evaluation Results Visualization")
    print("=" * 80)
    print()

    # 加载数据
    print("Loading experimental results...")
    exp1, exp2 = load_results()
    print("[OK] Data loaded")
    print()

    # 生成图表
    print("Generating visualization charts...")
    plot_experiment1(exp1)
    plot_experiment2(exp2)
    plot_comprehensive(exp1, exp2)
    print()
    print("=" * 80)
    print("All charts generated successfully!")
    print("=" * 80)

if __name__ == '__main__':
    main()