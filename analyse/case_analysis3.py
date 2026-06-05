import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 模拟真实情况：数据量完全不对等 ---
raw_data = [
    # 用户 U1: 成功很多，失败很少
    {'User': 'U1', 'Type': 'After_Success', 'Value': 3.5},
    {'User': 'U1', 'Type': 'After_Success', 'Value': 4.1},
    {'User': 'U1', 'Type': 'After_Success', 'Value': 3.2},
    {'User': 'U1', 'Type': 'After_Success', 'Value': 3.8},
    {'User': 'U1', 'Type': 'After_Success', 'Value': 2.25},
    {'User': 'U1', 'Type': 'After_Success', 'Value': 1.25},
    {'User': 'U1', 'Type': 'After_Success', 'Value': 2.75},
    {'User': 'U1', 'Type': 'After_Success', 'Value': 2.75},
    {'User': 'U1', 'Type': 'After_Success', 'Value': 5},
    {'User': 'U1', 'Type': 'After_Success', 'Value': 3.5},
    {'User': 'U1', 'Type': 'After_Success', 'Value': 1.5},
    {'User': 'U1', 'Type': 'After_SilentFail', 'Value': 7.2},
    {'User': 'U1', 'Type': 'After_SilentFail', 'Value': 7.5},
    {'User': 'U1', 'Type': 'After_SilentFail', 'Value': 5.25},
    
    # 用户 U2: 说话比较慢
    {'User': 'U2', 'Type': 'After_Success', 'Value': 5.0},
    {'User': 'U2', 'Type': 'After_Success', 'Value': 4.8},
    {'User': 'U2', 'Type': 'After_Success', 'Value': 3.5},
    {'User': 'U2', 'Type': 'After_Success', 'Value': 5.49},
    {'User': 'U2', 'Type': 'After_Success', 'Value': 3.75},
    {'User': 'U2', 'Type': 'After_Success', 'Value': 4.49},
    {'User': 'U2', 'Type': 'After_Success', 'Value': 2.0},
    {'User': 'U2', 'Type': 'After_Success', 'Value': 1.25},
    {'User': 'U2', 'Type': 'After_SilentFail', 'Value': 1.99},
    {'User': 'U2', 'Type': 'After_SilentFail', 'Value': 2.25},
    {'User': 'U2', 'Type': 'After_SilentFail', 'Value': 2.00},
    
    # 用户 U3: 样本量又不一样
    {'User': 'U3', 'Type': 'After_Success', 'Value': 3.00},
    {'User': 'U3', 'Type': 'After_Success', 'Value': 3.9},
    {'User': 'U3', 'Type': 'After_Success', 'Value': 3.5},
    {'User': 'U3', 'Type': 'After_Success', 'Value': 3.25},
    {'User': 'U3', 'Type': 'After_Success', 'Value': 4.50},
    {'User': 'U3', 'Type': 'After_Success', 'Value': 2.75},
    {'User': 'U3', 'Type': 'After_Success', 'Value': 3.00},
    {'User': 'U3', 'Type': 'After_Success', 'Value': 4.50},
    {'User': 'U3', 'Type': 'After_Success', 'Value': 2.50},
    {'User': 'U3', 'Type': 'After_SilentFail', 'Value': 6.8}
]

df = pd.DataFrame(raw_data)

# --- 绘图逻辑 ---
plt.figure(figsize=(8, 6))

# inner='quartile' 会画出那三条分位数线
# alpha 设置透明度，方便看清楚叠加的散点
sns.violinplot(x='Type', y='Value', data=df, inner='quartile', color="#F0F0F0")

# 叠加散点图 (Stripplot)，这样能一眼看出左边点多，右边点少
# hue='User' 可以让你看出不同用户的表现，如果不需要可以删掉
sns.stripplot(x='Type', y='Value', data=df, hue='User', size=8, jitter=True, alpha=0.7)

# 装饰
plt.title('Analysis 3: Follow-up Speech Duration Distribution', fontsize=14)
plt.ylabel('Duration (s)')
plt.xticks([0, 1], ['After Success\n(Robot Gaze OK)', 'After Silent Fail\n(No Robot Gaze)'])

sns.despine()
plt.legend(title='Participants', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()