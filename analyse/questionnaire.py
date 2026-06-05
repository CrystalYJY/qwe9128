import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scikit_posthocs import posthoc_conover
from math import pi

# ==================== 1. 配置信息 ====================
likert_mapping = {
    "完全不同意": 1, "不同意": 2, "比较不同意": 3, 
    "中立": 4, "比较同意": 5, "同意": 6, "完全同意": 7
}

column_mapping = {
    "GS_Anthropomorphism": [3, 4, 5, 6, 7],
    "GS_Animacy":          [8, 9, 10, 11, 12, 13],
    "GS_Likeability":      [14, 15, 16, 17, 18],
    "GS_Intelligence":     [19, 20, 21, 22, 23],
    "GS_Safety":           [24, 25, 26],
    "Joint_Attention":     [27, 28, 29], 
    "Env_Understanding":   [30, 31, 32], 
    "Social_Appropriateness": [33, 34, 35]
}

custom_dimensions = ["Joint_Attention", "Env_Understanding", "Social_Appropriateness"]

files = {
    "Exp (Gaze)": "/media/crystal/KINGSTON/download/实验3_10_10.xlsx",
    "Ctrl (NoGaze)": "/media/crystal/KINGSTON/download/实验2_10_10.xlsx",
    "Ctrl (Speak)": "/media/crystal/KINGSTON/download/实验1_11_10.xlsx"
}

# ==================== 2. 数据加载与归一化 ====================
def load_and_merge_data():
    all_dfs = []
    pd.set_option('future.no_silent_downcasting', True)
    
    for condition, path in files.items():
        df_raw = pd.read_excel(path)
        df_temp = pd.DataFrame()
        df_temp['Condition'] = [condition] * len(df_raw)
        
        for dim_name, col_indices in column_mapping.items():
            if dim_name in custom_dimensions:
                # 1-7 分制归一化: (x-1)/6
                raw_scores = df_raw.iloc[:, col_indices].replace(likert_mapping).infer_objects(copy=False)
                norm_scores = (raw_scores - 1) / 6.0
                df_temp[dim_name] = norm_scores.mean(axis=1)
            else:
                # 1-5 分制归一化: (x-1)/4
                raw_scores = df_raw.iloc[:, col_indices]
                norm_scores = (raw_scores - 1) / 4.0
                df_temp[dim_name] = norm_scores.mean(axis=1)
        all_dfs.append(df_temp)
    return pd.concat(all_dfs, ignore_index=True)

# ==================== 3. 绘图子函数 ====================
def make_radar_chart(data_df, metrics, title):
    """绘制归一化雷达图"""
    categories = metrics
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    colors = {"Exp (Gaze)": "#4C72B0", "Ctrl (NoGaze)": "#DD8452", "Ctrl (Speak)": "#55A868"}
    
    for i, row in data_df.iterrows():
        values = row[metrics].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['Condition'], color=colors.get(row['Condition']))
        ax.fill(angles, values, color=colors.get(row['Condition']), alpha=0.15)
        
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, size=10)
    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title, size=15, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig("normalized_radar_chart.png", dpi=300)

# ==================== 4. 主分析逻辑 ====================
def run_advanced_analysis():
    df = load_and_merge_data()
    metrics = list(column_mapping.keys())
    conditions = list(files.keys())
    
    # --- A. 柱状图 ---
    plt.figure(figsize=(16, 8))
    sns.set_theme(style="whitegrid")
    
    df_melted = df.melt(id_vars=['Condition'], value_vars=metrics, var_name='Dimension', value_name='Score')
    ax = sns.barplot(x='Dimension', y='Score', hue='Condition', data=df_melted, 
                     palette='coolwarm', capsize=.05, errorbar='sd')
    
    print("\n" + "="*20 + " 详细统计分析报告 " + "="*20)
    
    for i, metric in enumerate(metrics):
        # 统计检验
        groups = [df[df['Condition'] == c][metric].values for c in conditions]
        stat, p_val = stats.friedmanchisquare(*groups)
        print(f"\n指标: {metric} | Friedman p = {p_val:.4e}")
        
        if p_val < 0.05:
            res = posthoc_conover(df, val_col=metric, group_col='Condition', p_adjust='bonferroni')
            max_val = df[metric].max() 
            
            # 标注星号逻辑
            p_es = res.loc['Exp (Gaze)', 'Ctrl (Speak)']
            if p_es < 0.05:
                stars = "***" if p_es < 0.001 else "**" if p_es < 0.01 else "*"
                ax.text(i - 0.25, max_val + 0.05, f"vs.S:{stars}", ha='center', va='bottom', color='red', fontsize=9, fontweight='bold')
            
            p_en = res.loc['Exp (Gaze)', 'Ctrl (NoGaze)']
            if p_en < 0.05:
                stars = "***" if p_en < 0.001 else "**" if p_en < 0.01 else "*"
                ax.text(i + 0.25, max_val + 0.15, f"vs.N:{stars}", ha='center', va='bottom', color='blue', fontsize=9, fontweight='bold')

    plt.title("Normalized Subjective Results (Scale 0.0 - 1.0)", fontsize=16, fontweight='bold')
    plt.ylim(0, 1.4) 
    plt.ylabel("Normalized Mean Score (0-1)")
    
    # 修改此处：rotation=0 确保横坐标标签不倾斜
    plt.xticks(rotation=0) 
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Conditions")
    plt.tight_layout()
    plt.savefig("normalized_bar_chart_with_stars.png", dpi=300)

    # --- B. 雷达图 ---
    radar_df = df.groupby('Condition')[metrics].mean().reset_index()
    make_radar_chart(radar_df, metrics, "Robot Perception Profile Comparison (Normalized)")
    
    plt.show()

if __name__ == "__main__":
    run_advanced_analysis()