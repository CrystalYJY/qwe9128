import matplotlib.pyplot as plt
import numpy as np

# --- 数据录入 ---
users = ['User 1', 'User 2', 'User 3', 'User 4', 'User 5', 'User 6', 'User 7', 'User 8', 'User 9', 'User 10']
c2_rates = [14.3, 50.0, 20.0, 10.0, 25.0, 20.0, 33.0, 18.0, 26.0, 15.0]  # C2 组跟随率
e_rates = [35.7, 57.3, 25.0, 22.0, 45.0, 33.0, 36.0, 57.0, 35.0, 28.0]    # E 组跟随率

x = np.arange(len(users))  # 标签位置
width = 0.35  # 柱状图宽度

# --- 开始绘图 ---
fig, ax = plt.subplots(figsize=(10, 7))

# 绘制柱子
rects1 = ax.bar(x - width/2, c2_rates, width, label='Control (C2: No Gaze)', color='#AED6F1', edgecolor='grey')
rects2 = ax.bar(x + width/2, e_rates, width, label='Experimental (E: VLM Gaze)', color='#F1948A', edgecolor='grey')

# 添加装饰
ax.set_ylabel('Human Gaze Rate (%)', fontsize=12, fontweight='bold')
ax.set_title("Evaluating the Robot's Ability to Lead Human Attention to Objects", fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(users, fontsize=11)
ax.set_ylim(0, 80) # 设置纵坐标上限
ax.legend(fontsize=10)

# 在柱子上标注具体数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3点纵向偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

# 增加网格线增强可读性
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True) # 网格线下沉

plt.tight_layout()
plt.show()