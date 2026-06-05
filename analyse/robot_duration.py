import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
# --- 第一步：加载数据 ---
try:
    # 实验组 E 的数据 (User 1, 2, 3)
    df_e_u1 = pd.read_csv('/media/crystal/KINGSTON/qwe/2/haru_conversation_logs/1/conversation_turns.csv') 
    df_e_u2 = pd.read_csv('/media/crystal/KINGSTON/qwe/2/haru_conversation_logs/2/conversation_turns.csv') 
    df_e_u3 = pd.read_csv('/media/crystal/KINGSTON/qwe/2/haru_conversation_logs/3/conversation_turns.csv') 
    df_e_u4 = pd.read_csv('/media/crystal/KINGSTON/qwe/2/haru_conversation_logs/4/conversation_turns.csv')
    df_e_u5 = pd.read_csv('/media/crystal/KINGSTON/qwe/2/haru_conversation_logs/5/conversation_turns.csv')

    # 对照组 C2 的数据
    df_c2_u1 = pd.read_csv('/media/crystal/KINGSTON/qwe/no_gaze/haru_conversation_logs/1/conversation_turns.csv')
    df_c2_u2 = pd.read_csv('/media/crystal/KINGSTON/qwe/no_gaze/haru_conversation_logs/2/conversation_turns.csv')
    df_c2_u3 = pd.read_csv('/media/crystal/KINGSTON/qwe/no_gaze/haru_conversation_logs/3/conversation_turns.csv')
    df_c2_u4 = pd.read_csv('/media/crystal/KINGSTON/qwe/no_gaze/haru_conversation_logs/4/conversation_turns.csv')
    df_c2_u5 = pd.read_csv('/media/crystal/KINGSTON/qwe/no_gaze/haru_conversation_logs/5/conversation_turns.csv')
    
    # 对照组 C1 的数据
    df_c1_u1 = pd.read_csv('/media/crystal/KINGSTON/qwe/speak/haru_conversation_logs/1/conversation_turns.csv')
    df_c1_u2 = pd.read_csv('/media/crystal/KINGSTON/qwe/speak/haru_conversation_logs/2/conversation_turns.csv')
    df_c1_u3 = pd.read_csv('/media/crystal/KINGSTON/qwe/speak/haru_conversation_logs/3/conversation_turns.csv')
    df_c1_u4 = pd.read_csv('/media/crystal/KINGSTON/qwe/speak/haru_conversation_logs/4/conversation_turns.csv')
    df_c1_u5 = pd.read_csv('/media/crystal/KINGSTON/qwe/speak/haru_conversation_logs/5/conversation_turns.csv')
except FileNotFoundError:
    print("请确保文件名正确！此处先用示例逻辑运行。")

# --- 第二步：提取目标列并打上组别标签 ---
def get_duration(df, label, user_id):
    # 提取 robot_duration (s) 这一列，并去掉可能的空值
    return pd.DataFrame({
        'Duration': df['robot_duration (s)'].dropna(),
        'Condition': label,
        'User': user_id
    })

# 合并所有数据
df_list = [
    get_duration(df_e_u1, 'Experimental', 'U1'),
    get_duration(df_e_u2, 'Experimental', 'U2'),
    get_duration(df_e_u3, 'Experimental', 'U3'),
    get_duration(df_e_u4, 'Experimental', 'U4'),
    get_duration(df_e_u5, 'Experimental', 'U5'),
    get_duration(df_c2_u1, 'Control_NoGaze', 'U1'),
    get_duration(df_c2_u2, 'Control_NoGaze', 'U2'),
    get_duration(df_c2_u3, 'Control_NoGaze', 'U3'),
    get_duration(df_c2_u4, 'Control_NoGaze', 'U4'),
    get_duration(df_c2_u5, 'Control_NoGaze', 'U5'),
    get_duration(df_c1_u1, 'Control_Speak', 'U1'),
    get_duration(df_c1_u2, 'Control_Speak', 'U2'),
    get_duration(df_c1_u3, 'Control_Speak', 'U3'),
    get_duration(df_c1_u4, 'Control_Speak', 'U4'),
    get_duration(df_c1_u5, 'Control_Speak', 'U5')
]

df = pd.concat(df_list, ignore_index=True)

# --- 计算详细统计量 ---
# 均值、标准差、样本量（对话轮数）
stats_summary = df.groupby('Condition')['Duration'].agg(['mean', 'std', 'count']).reset_index()
print("各组详细统计：")
print(stats_summary)

# --- 核心改进：计算 P 值 ---
# 证明这 3 秒的差距是否具有统计学意义
groups = [df[df['Condition'] == c]['Duration'] for c in df['Condition'].unique()]
f_stat, p_val = stats.f_oneway(*groups)

# --- 绘图优化 ---
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")

# 使用 palette='muted' 看起来更专业
ax = sns.barplot(x='Condition', y='Duration', hue='Condition', data=df, capsize=.1, palette='muted', err_kws={'linewidth': 1.5}, legend=False)

# 在图上直接标注均值数值
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}s', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=12)

#plt.title(f'Robot Speech Duration Consistency (p={p_val:.3f})', fontsize=14)
plt.title(f'Robot Speech Duration Consistency', fontsize=14)
plt.ylabel('Mean Robot Duration (s)')
plt.axhline(df['Duration'].mean(), color='red', linestyle='--', alpha=0.5, label='Total Mean')

plt.tight_layout()
plt.show()

if p_val > 0.05:
    print(f"P值为 {p_val:.3f} > 0.05：各组时长在统计学上没有显著差异，实验变量控制良好。")
else:
    print(f"P值为 {p_val:.3f} < 0.05：注意！各组时长存在显著差异，可能存在干扰变量。")

#为了保证实验靠谱，我们检查了三组机器人说话的时长。方差分析显示，三组之间没有显著差异：实验组（平均11.26秒）、不转头组（平均12.86秒）和盲人组
# （平均9.82秒）。统计数字显示 F(2,112)=1.45，由于 P 值 0.238 远大于 0.05，说明这点时间差距纯属随机波动。这有力地证明了：用户觉得实验组更好，
# 是因为机器人转头看向了物体，而不是因为机器人话多或话少。”
# P值为 0.238 > 0.05：各组时长在统计学上没有显著差异，实验变量控制良好。