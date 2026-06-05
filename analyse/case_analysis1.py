import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 准备数据
data = {
    'User': ['U1', 'U1', 'U2', 'U2', 'U3', 'U3', 'U4', 'U4', 'U5', 'U5'],
    'Condition': ['Normal_Avg', 'Correction_Turn', 'Normal_Avg', 'Correction_Turn', 'Normal_Avg', 'Correction_Turn', 'Normal_Avg', 'Correction_Turn', 'Normal_Avg', 'Correction_Turn'],
    'Duration': [3.96, 4.5, 3.73, 4.22, 3.71, 5.2, 4.05, 6.0, 6.36, 7.25]
}
df = pd.DataFrame(data)

# 2. 设置科研风格
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))

# 3. 绘制带连线的点图 (Pointplot)
sns.pointplot(data=df, x='Condition', y='Duration', hue='User', 
              markers="o", linestyles="--", capsize=.05)

# 4. 图表美化
plt.title('Analysis 1: Individual Correction Burden', fontsize=14, pad=20)
plt.ylabel('Human Speaking Duration (s)', fontsize=12)
plt.xlabel('Interaction State', fontsize=12)
plt.ylim(0, df['Duration'].max() + 1)
sns.despine(left=True)

plt.tight_layout()
plt.show()