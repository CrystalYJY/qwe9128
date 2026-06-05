import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scikit_posthocs import posthoc_conover
from math import pi

# ==================== 1. 配置列索引与映射 ====================
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

# ==================== 2. 数据处理 (归一化至 0-1) ====================
def load_and_merge_data():
    all_dfs = []
    # 修复 Future.no_silent_downcasting 警告
    pd.set_option('future.no_silent_downcasting', True)
    
    for condition, path in files.items():
        df_raw = pd.read_excel(path)
        df_temp = pd.DataFrame()
        df_temp['Condition'] = [condition] * len(df_raw)
        
        for dim_name, col_indices in column_mapping.items():
            if dim_name in custom_dimensions:
                # 1-7 分制归一化: (x-1)/(7-1)
                raw_scores = df_raw.iloc[:, col_indices].replace(likert_mapping).infer_objects(copy=False)
                norm_scores = (raw_scores - 1) / 6.0
                df_temp[dim_name] = norm_scores.mean(axis=1)
            else:
                # 1-5 分制归一化: (x-1)/(5-1)
                raw_scores = df_raw.iloc[:, col_indices]
                norm_scores = (raw_scores - 1) / 4.0
                df_temp[dim_name] = norm_scores.mean(axis=1)
        all_dfs.append(df_temp)
    return pd.concat(all_dfs, ignore_index=True)

# ==================== 3. 统计分析与高级绘图 ====================
def run_advanced_analysis():
    df = load_and_merge_data()
    metrics = list(column_mapping.keys())
    conditions = list(files.keys())
    
    # 修复 Future.no_silent_downcasting 警告
    pd.set_option('future.no_silent_downcasting', True)

    # --- A. 柱状图部分 ---
    plt.figure(figsize=(16, 8))
    sns.set_theme(style="whitegrid")
    
    df_melted = df.melt(id_vars=['Condition'], value_vars=metrics, var_name='Dimension', value_name='Score')
    ax = sns.barplot(x='Dimension', y='Score', hue='Condition', data=df_melted, 
                     palette='coolwarm', capsize=.05, errorbar='sd')
    
    print("\n" + "="*20 + " 详细统计分析报告 (归一化数据 0-1) " + "="*20)
    
    for i, metric in enumerate(metrics):
        # 提取各组数据用于 Friedman 检验
        g_exp = df[df['Condition'] == 'Exp (Gaze)'][metric].values
        g_no_gaze = df[df['Condition'] == 'Ctrl (NoGaze)'][metric].values
        g_speak = df[df['Condition'] == 'Ctrl (Speak)'][metric].values
        
        stat, p_val = stats.friedmanchisquare(g_exp, g_no_gaze, g_speak)
        
        print(f"\n指标: {metric} | Friedman p = {p_val:.4e}")
        
        if p_val < 0.05:
            # 事后检验
            res = posthoc_conover(df, val_col=metric, group_col='Condition', p_adjust='bonferroni')
            print("事后两两对比矩阵 (P-values):")
            print(res)

            # --- 自动标星逻辑修正 ---
            # 获取当前维度的最大值，作为标注的起始高度
            max_val = df[metric].max() 
            
            # 1. 标注 Exp vs Speak (红色)
            p_es = res.loc['Exp (Gaze)', 'Ctrl (Speak)']
            if p_es < 0.05:
                # 判定星号数量：根据真实 p 值自动分配
                stars = "***" if p_es < 0.001 else "**" if p_es < 0.01 else "*"
                ax.text(i - 0.25, max_val + 0.05, f"vs.S:{stars}", 
                        ha='center', va='bottom', color='red', fontsize=9, fontweight='bold')
            
            # 2. 标注 Exp vs NoGaze (蓝色)
            p_en = res.loc['Exp (Gaze)', 'Ctrl (NoGaze)']
            if p_en < 0.05:
                # 判定星号数量：根据真实 p 值自动分配
                stars = "***" if p_en < 0.001 else "**" if p_en < 0.01 else "*"
                ax.text(i + 0.25, max_val + 0.15, f"vs.N:{stars}", 
                        ha='center', va='bottom', color='blue', fontsize=9, fontweight='bold')

    plt.title("Normalized Subjective Results (Scale 0.0 - 1.0)", fontsize=16, fontweight='bold')
    plt.ylim(0, 1.4) # 给顶部的星号留足空间
    plt.ylabel("Normalized Mean Score (0-1)")
    plt.xticks(rotation=15)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Conditions")
    plt.tight_layout()
    plt.savefig("normalized_bar_chart_with_stars.png", dpi=300)

    # --- B. 雷达图部分 ---
    radar_df = df.groupby('Condition')[metrics].mean().reset_index()
    
    def make_radar_chart(data_df, title):
        categories = list(data_df)[1:]
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        colors = {"Exp (Gaze)": "#4C72B0", "Ctrl (NoGaze)": "#DD8452", "Ctrl (Speak)": "#55A868"}
        
        for i, row in data_df.iterrows():
            values = row.drop('Condition').values.flatten().tolist()
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

    make_radar_chart(radar_df, "Robot Perception Profile Comparison (Normalized)")
    plt.show()
    def make_radar_chart(data_df, title):
        categories = list(data_df)[1:]
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        colors = {"Exp (Gaze)": "#4C72B0", "Ctrl (NoGaze)": "#DD8452", "Ctrl (Speak)": "#55A868"}
        
        for i, row in data_df.iterrows():
            values = row.drop('Condition').values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['Condition'], color=colors.get(row['Condition']))
            ax.fill(angles, values, color=colors.get(row['Condition']), alpha=0.15)
            
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        # 优化坐标轴显示
        plt.xticks(angles[:-1], categories, size=10)
        ax.set_rlabel_position(30)
        # 刻度改为 0 到 1
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
        plt.ylim(0, 1.1)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title(title, size=15, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig("normalized_radar_chart.png", dpi=300)

    make_radar_chart(radar_df, "Robot Perception Profile Comparison (Normalized)")
    plt.show()

if __name__ == "__main__":
    run_advanced_analysis()