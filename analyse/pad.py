import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 数据准备
groups = ['Exp (Gaze)', 'Exp (Chat)', 'Ctrl (NoGaze)', 'Ctrl (Chat)', 'Ctrl (Speak)']
data = {
    'Preprocessing (A)': [311, 310, 307, 304, 0],
    'VLM Inference (B)': [6597, 2792, 7487, 4603, 2080],
    'Execution (C)': [1116, 9, 7, 9, 12]
}

df = pd.DataFrame(data, index=groups)

# 绘图配置
fig, ax = plt.subplots(figsize=(12, 8))
y = np.arange(len(groups))  # 组别位置
height = 0.25  # 条形图高度

# 绘制三个阶段
ax.barh(y + height, df['Preprocessing (A)'], height, label='Preprocessing (A)', color='#FFD700')
ax.barh(y, df['VLM Inference (B)'], height, label='VLM Inference (B)', color='#87CEFA')
ax.barh(y - height, df['Execution (C)'], height, label='Execution (C)', color='#FF6347')

# 添加数值标注
for i, row in enumerate(df.itertuples()):
    ax.text(row[1]+50, i+height, f'{int(row[1])}ms', va='center', fontsize=9)
    ax.text(row[2]+50, i, f'{int(row[2])}ms', va='center', fontsize=9)
    ax.text(row[3]+50, i-height, f'{int(row[3])}ms', va='center', fontsize=9)

# 美化
ax.set_yticks(y)
ax.set_yticklabels(groups, fontweight='bold')
ax.set_xlabel('Latency (milliseconds)')
ax.set_title('Detailed Latency Breakdown by Stage across Conditions', fontsize=14, pad=20)
ax.legend()
ax.invert_yaxis()  # 让 Exp (Gaze) 放在最上面
ax.grid(axis='x', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()