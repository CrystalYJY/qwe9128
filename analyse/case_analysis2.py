import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 模拟数据：记录每一次遇到 NotFound 情况时，最终解决问题花的轮数
raw_data = [
    {'Path': 'Direct Honesty', 'Turns': 1}, # 诚实路径通常都是1轮
    {'Path': 'Direct Honesty', 'Turns': 1},
    {'Path': 'Direct Honesty', 'Turns': 1},
    {'Path': 'Hallucination', 'Turns': 2},  # 幻觉路径：指错->纠正->认错 (2轮)
    {'Path': 'Hallucination', 'Turns': 3},  # 幻觉路径：指错->纠正->再错->再纠正 (3轮)
    {'Path': 'Hallucination', 'Turns': 2}
]

df2 = pd.DataFrame(raw_data)

# 计算均值用于绘图
df2_avg = df2.groupby('Path')['Turns'].mean().reset_index()

plt.figure(figsize=(6, 6))
ax = sns.barplot(x='Path', y='Turns', data=df2_avg, palette=['#45ad8b', '#c44e52'])

# 标注具体的数值
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.1f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 9), textcoords='offset points')

plt.title('Analysis 2: Efficiency (Turns to Resolve)', fontsize=14)
plt.ylabel('Average Conversation Turns')
plt.ylim(0, 4)
plt.tight_layout()
plt.show()