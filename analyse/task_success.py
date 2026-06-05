import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set_theme(style="whitegrid")

# 任务成功率数据
conditions = ['Exp (Gaze)', 'Ctrl (NoGaze)', 'Ctrl (Speak)']
success_rates = [100, 100, 20]  # 百分比

# 创建柱状图
colors = sns.color_palette("Set2", len(conditions))
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(conditions, success_rates, color=colors, width=0.6)

# 添加数值标签
for bar, rate in zip(bars, success_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{rate}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 设置标题和标签
ax.set_title('Task Success Rate by Condition', fontsize=16, fontweight='bold')
ax.set_ylabel('Success Rate (%)', fontsize=14)
ax.set_xlabel('Condition', fontsize=14)
ax.set_ylim(0, 110)  # 设置y轴范围

# 显示网格
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()