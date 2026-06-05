import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import numpy as np
import scipy.stats as stats
from scikit_posthocs import posthoc_conover

# ==================== 1. 配置部分 ====================
BASE_PATH = "/media/crystal/KINGSTON/qwe"

GROUPS_CONFIG = {
    "实验组_E": {
        "path_template": f"{BASE_PATH}/2/haru_conversation_logs/{{user_id}}/metadata.json",
        "condition_name": "Exp (Gaze)",
        "user_ids": [str(i) for i in range(1, 11)],
        "manual_data": {"6": 300, "7": 510, "8": 566, "9": 290, "10": 383}
    },
    "对照组_C2": {
        "path_template": f"{BASE_PATH}/no_gaze/haru_conversation_logs/{{user_id}}/metadata.json",
        "condition_name": "Ctrl (NoGaze)",
        "user_ids": [str(i) for i in range(1, 11)],
        "manual_data": {"6": 334, "7": 900, "8": 540, "9": 580, "10": 630}
    },
    "对照组_C1": {
        "path_template": f"{BASE_PATH}/speak/haru_conversation_logs/{{user_id}}/metadata.json",
        "condition_name": "Ctrl (Speak)",
        "user_ids": [str(i) for i in range(1, 11)]
    }
}

# ==================== 2. 数据加载函数 ====================
def load_metadata_data():
    all_data = []
    for group_name, config in GROUPS_CONFIG.items():
        user_ids = config.get("user_ids", [])
        for user_id in user_ids:
            file_path = config["path_template"].format(user_id=user_id)
            total_turns, completion_time = 0, 0
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    total_turns = metadata.get('conversation_count', 0)
                    completion_time = metadata.get('duration_seconds', 0)
                    if completion_time == 0:
                        start, end = metadata.get('start_time', 0), metadata.get('end_time', 0)
                        if start and end: completion_time = end - start
                except: pass

            manual_data = config.get("manual_data", {})
            if user_id in manual_data:
                completion_time = manual_data[user_id]
            
            all_data.append({
                'Condition': config["condition_name"],
                'Total_Turns': total_turns,
                'Completion_Time': completion_time
            })
    return pd.DataFrame(all_data)

# ==================== 3. 统计分析逻辑 ====================
def perform_stats(df, metric):
    print(f"\n--- {metric} 统计分析报告 ---")
    conditions = df['Condition'].unique()
    
    # 正态性检验
    for cond in conditions:
        p = stats.shapiro(df[df['Condition'] == cond][metric])[1]
        print(f"[{cond}] Shapiro p-value: {p:.4f}")
    
    # ANOVA 检验
    groups = [df[df['Condition'] == c][metric] for c in conditions]
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"ANOVA 结果: F={f_stat:.2f}, p={p_anova:.4e}")
    
    # 事后检验 (Conover-Iman)
    pc = posthoc_conover(df, val_col=metric, group_col='Condition', p_adjust='bonferroni')
    return p_anova, pc

# ==================== 4. 绘图与显著性标注 ====================
def add_significance_stars(ax, df, metric, posthoc_df):
    """在图表上自动添加显著性星号"""
    pairs = [("Exp (Gaze)", "Ctrl (Speak)"), ("Ctrl (NoGaze)", "Ctrl (Speak)")]
    y_max = df[metric].max()
    
    for i, (g1, g2) in enumerate(pairs):
        p_val = posthoc_df.loc[g1, g2]
        if p_val < 0.05:
            # 计算标注高度
            level = y_max * (1.05 + i * 0.1)
            # 获取 x 轴坐标 (Seaborn boxplot 默认顺序)
            order = [t.get_text() for t in ax.get_xticklabels()]
            x1, x2 = order.index(g1), order.index(g2)
            
            # 画横线
            ax.plot([x1, x1, x2, x2], [level-y_max*0.02, level, level, level-y_max*0.02], lw=1.5, c='k')
            # 确定星号数量
            stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
            ax.text((x1 + x2) * 0.5, level, stars, ha='center', va='bottom', color='k', fontsize=14, fontweight='bold')

# --- 主程序运行 ---
df = load_metadata_data()
sns.set_theme(style="whitegrid")

# 分析与绘制 Total Turns
p_turns, posthoc_turns = perform_stats(df, 'Total_Turns')
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.boxplot(x='Condition', y='Total_Turns', data=df, ax=ax1, palette='Set2', width=0.5, showmeans=True,
            meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})
sns.swarmplot(x='Condition', y='Total_Turns', data=df, ax=ax1, color=".25", alpha=0.6)
add_significance_stars(ax1, df, 'Total_Turns', posthoc_turns)
ax1.set_title('Total Conversation Turns', fontsize=15, fontweight='bold')
ax1.set_ylabel('Number of Turns')

# 分析与绘制 Completion Time
p_time, posthoc_time = perform_stats(df, 'Completion_Time')
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.boxplot(x='Condition', y='Completion_Time', data=df, ax=ax2, palette='Set2', width=0.5, showmeans=True,
            meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})
sns.swarmplot(x='Condition', y='Completion_Time', data=df, ax=ax2, color=".25", alpha=0.6)
add_significance_stars(ax2, df, 'Completion_Time', posthoc_time)
ax2.set_title('Total Task Completion Time', fontsize=15, fontweight='bold')
ax2.set_ylabel('Time (seconds)')

plt.tight_layout()
plt.show()