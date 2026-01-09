# -*- coding: utf-8 -*-
"""
生成分开的独立图表（每个图表单独一张图片）
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
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

def plot_experiment1_separate(exp1):
    """绘制实验1：模型对比 - 分成两个独立图片"""
    # 提取数据
    qwen_times = [item['response_time'] for item in exp1['qwen3_8b']]
    deepseek_times = [item['response_time'] for item in exp1['deepseek']]
    queries = [f'Q{i+1}' for i in range(len(exp1['qwen3_8b']))]

    # ==================== 图1a: 响应时间对比 ====================
    fig1, ax1 = plt.subplots(figsize=(14, 7))

    x = np.arange(len(queries))
    width = 0.35

    bars1 = ax1.bar(x - width/2, qwen_times, width, label='qwen3:8b', color=COLORS['qwen3_8b'])
    bars2 = ax1.bar(x + width/2, deepseek_times, width, label='DeepSeek V3.2', color=COLORS['deepseek'])

    ax1.set_xlabel('Question', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Response Time (s)', fontsize=14, fontweight='bold')
    ax1.set_title('Experiment 1: Model Response Time Comparison', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(queries, rotation=45, ha='right', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=11)

    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('figure1a_model_response_time.png', dpi=300, bbox_inches='tight')
    print("[OK] Figure 1a saved: figure1a_model_response_time.png")
    plt.close()

    # ==================== 图1b: 平均响应时间对比 ====================
    fig2, ax2 = plt.subplots(figsize=(10, 7))

    avg_qwen = np.mean(qwen_times)
    avg_deepseek = np.mean(deepseek_times)

    avg_times = [avg_qwen, avg_deepseek]
    models = ['qwen3:8b', 'DeepSeek V3.2']
    x2 = np.arange(len(models))
    width2 = 0.6

    bars3 = ax2.bar(x2, avg_times, width2, 
                   color=[COLORS['qwen3_8b'], COLORS['deepseek']])

    ax2.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Avg Response Time (s)', fontsize=14, fontweight='bold')
    ax2.set_title('Average Response Time Comparison', fontsize=16, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(models, fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 添加数值标签和加速比
    for i, (bar, time) in enumerate(zip(bars3, avg_times)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}s', ha='center', va='bottom', fontsize=13, fontweight='bold')

    # 添加加速比标注
    speedup = (avg_qwen / avg_deepseek - 1) * 100
    speedup_text = f'DeepSeek is {speedup:.1f}% faster'
    ax2.text(0.5, max(avg_times) * 0.9,
             speedup_text,
             ha='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('figure1b_avg_response_time.png', dpi=300, bbox_inches='tight')
    print("[OK] Figure 1b saved: figure1b_avg_response_time.png")
    plt.close()

def plot_experiment2_separate(exp2):
    """绘制实验2：检索策略对比 - 分成两个独立图片"""
    # 提取数据
    rag_times = [item['response_time'] for item in exp2['rag']]
    bf_times = [item['response_time'] for item in exp2['bruteforce']]
    nc_times = [item['response_time'] for item in exp2['no_context']]
    queries = [f'Q{i+1}' for i in range(len(exp2['rag']))]

    # ==================== 图2a: 响应时间对比 ====================
    fig1, ax1 = plt.subplots(figsize=(14, 7))

    x = np.arange(len(queries))
    width = 0.25

    bars1 = ax1.bar(x - width, rag_times, width, label='RAG', color=COLORS['rag'])
    bars2 = ax1.bar(x, bf_times, width, label='Brute Force', color=COLORS['bruteforce'])
    bars3 = ax1.bar(x + width, nc_times, width, label='No Context', color=COLORS['no_context'])

    ax1.set_xlabel('Question', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Response Time (s)', fontsize=14, fontweight='bold')
    ax1.set_title('Experiment 2: Strategy Response Time Comparison', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(queries, rotation=45, ha='right', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=10)

    for bar in bars3:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('figure2a_strategy_response_time.png', dpi=300, bbox_inches='tight')
    print("[OK] Figure 2a saved: figure2a_strategy_response_time.png")
    plt.close()

    # ==================== 图2b: 平均性能对比 ====================
    fig2, ax2 = plt.subplots(figsize=(10, 7))

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

    ax2.set_xlabel('Strategy', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Avg Response Time (s)', fontsize=14, fontweight='bold')
    ax2.set_title('Average Performance Comparison', fontsize=16, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(strategies, fontsize=12)
    ax2.legend(loc='upper left', fontsize=11)
    ax2_twin.legend(loc='upper right', fontsize=11)
    ax2_twin.set_ylabel('Avg Context Length', fontsize=14, fontweight='bold')

    # 添加数值标签
    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=11)

    for bar in bars5:
        height = bar.get_height()
        ax2_twin.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('figure2b_avg_performance.png', dpi=300, bbox_inches='tight')
    print("[OK] Figure 2b saved: figure2b_avg_performance.png")
    plt.close()

def plot_radar_separate(exp1, exp2):
    """绘制综合性能雷达图 - 单独一张图"""
    fig = plt.figure(figsize=(12, 10))

    # 提取数据
    qwen_times = [item['response_time'] for item in exp1['qwen3_8b']]
    deepseek_times = [item['response_time'] for item in exp1['deepseek']]
    rag_times = [item['response_time'] for item in exp2['rag']]
    bf_times = [item['response_time'] for item in exp2['bruteforce']]
    nc_times = [item['response_time'] for item in exp2['no_context']]

    ctx_data = [
        [item['context_length'] for item in exp1['qwen3_8b']],
        [item['context_length'] for item in exp1['deepseek']],
        [item['context_length'] for item in exp2['rag']],
        [item['context_length'] for item in exp2['bruteforce']],
        [0] * len(exp2['no_context'])
    ]

    # 计算各项指标（归一化到0-1）
    def normalize(values, inverse=False):
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return [0.5] * len(values)
        if inverse:
            return [(max_val - v) / (max_val - min_val) for v in values]
        return [(v - min_val) / (max_val - min_val) for v in values]

    # 响应速度（时间越短越好）
    speed_scores = normalize([np.mean(qwen_times), np.mean(deepseek_times), 
                             np.mean(rag_times), np.mean(bf_times), np.mean(nc_times)], inverse=True)

    # 上下文效率（长度适中最好，太长太短都不好）
    avg_ctx = [np.mean(c) for c in ctx_data]
    # RAG最佳（有上下文但不太长），暴力塞文档次之，无上下文最差
    ideal_ctx = 3000
    ctx_scores = [1 - abs(c - ideal_ctx) / ideal_ctx for c in avg_ctx]
    ctx_scores = [max(0, min(1, s)) for s in ctx_scores]

    # 稳定性（方差越小越好）
    stability_scores = normalize([np.std(qwen_times), np.std(deepseek_times),
                                 np.std(rag_times), np.std(bf_times), np.std(nc_times)], inverse=True)

    # 资源利用（综合评分：速度和上下文效率的平衡）
    resource_scores = [(s + c) / 2 for s, c in zip(speed_scores, ctx_scores)]

    # 绘制雷达图
    ax = fig.add_subplot(111, projection='polar')

    categories = ['Response Speed', 'Context Efficiency', 'Stability', 'Resource Usage']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    labels = ['qwen3:8b', 'DeepSeek', 'RAG', 'Brute Force', 'No Context']
    colors = [COLORS['qwen3_8b'], COLORS['deepseek'], COLORS['rag'],
              COLORS['bruteforce'], COLORS['no_context']]

    for i, (label, color) in enumerate(zip(labels, colors)):
        values = [speed_scores[i], ctx_scores[i], stability_scores[i], resource_scores[i]]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True)
    ax.set_title('Comprehensive Performance Radar Chart', fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig('figure3_comprehensive_radar.png', dpi=300, bbox_inches='tight')
    print("[OK] Figure 3 saved: figure3_comprehensive_radar.png")
    plt.close()

def main():
    """主函数"""
    print("=" * 80)
    print("Generating Separate Charts")
    print("=" * 80)
    print()

    # 加载数据
    print("Loading experimental results...")
    exp1, exp2 = load_results()
    print("[OK] Data loaded")
    print()

    # 生成图表
    print("Generating visualization charts...")
    plot_experiment1_separate(exp1)
    plot_experiment2_separate(exp2)
    plot_radar_separate(exp1, exp2)
    print()
    print("=" * 80)
    print("All separate charts generated successfully!")
    print("=" * 80)

if __name__ == '__main__':
    main()