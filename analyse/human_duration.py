import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==================== 配置部分 ====================
# 1. 设置你要计算平均值的列名
COLUMN_TO_AVERAGE = "human_duration (s)"  # 请改为你CSV文件中的实际列名

# 2. 配置实验组和对照组的路径
BASE_PATH = "/media/crystal/KINGSTON/qwe"

GROUPS_CONFIG = {
    "Experimental(E)": {
        "path_template": f"{BASE_PATH}/2/haru_conversation_logs/{{user_id}}/conversation_turns.csv",
        "user_ids": ["1", "2", "3", "5", "6", "7", "8", "9", "10"]
    },
    "Control(C2)": {
        "path_template": f"{BASE_PATH}/no_gaze/haru_conversation_logs/{{user_id}}/conversation_turns.csv",
        "user_ids": ["1", "2", "3", "5", "6", "7", "8", "9", "10"]
    },
    "Control(C1)": {
        "path_template": f"{BASE_PATH}/speak/haru_conversation_logs/{{user_id}}/conversation_turns.csv",
        "user_ids": ["1", "2", "3", "5", "6", "7", "8", "9", "10"]
    }
}

# --- 第一步：根据配置加载所有 CSV 文件 ---
def load_csv_files():
    """根据配置加载所有CSV文件"""
    all_dataframes = {}
    
    for group_name, config in GROUPS_CONFIG.items():
        path_template = config["path_template"]
        user_ids = config["user_ids"]
        
        for user_id in user_ids:
            file_path = path_template.format(user_id=user_id)
            key = f"{group_name}_U{user_id}"
            
            try:
                df = pd.read_csv(file_path)
                all_dataframes[key] = df
                print(f"✅ 成功加载: {key} -> {file_path}")
            except FileNotFoundError:
                print(f"❌ 文件不存在: {key} -> {file_path}")
                all_dataframes[key] = None
            except Exception as e:
                print(f"❌ 加载失败: {key} -> {file_path} (错误: {e})")
                all_dataframes[key] = None
    
    return all_dataframes

# 加载所有数据
print("🔍 开始加载CSV文件...")
all_dataframes = load_csv_files()
print(f"📊 共加载了 {len([df for df in all_dataframes.values() if df is not None])}/{len(all_dataframes)} 个文件\n")

# --- 第二步：提取目标列并打上组别标签 ---
def get_duration(df, label):
    """提取指定列的数据，并去掉可能的空值"""
    if df is None:
        return pd.DataFrame()
    
    return pd.DataFrame({
        'Duration': df[COLUMN_TO_AVERAGE].dropna(),
        'Condition': label
    })

# 合并所有数据
print("🔄 合并数据中...")
all_data = pd.DataFrame()

for key, df in all_dataframes.items():
    if df is not None:
        # 从key中提取组名 (如 "实验组_E_U1" -> "实验组_E")
        group_name = "_".join(key.split("_")[:-1])  # 去掉最后的 "_U{数字}"
        group_data = get_duration(df, group_name)
        all_data = pd.concat([all_data, group_data], ignore_index=True)
        print(f"✅ 添加了 {len(group_data)} 条记录到组 '{group_name}'")

print(f"\n📈 总共处理了 {len(all_data)} 条记录")
print(f"📊 数据分布:\n{all_data['Condition'].value_counts()}\n")

# --- 第三步：绘制箱线图 ---
plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")

# 绘图：x轴是组别，y轴是时长
ax = sns.boxplot(x='Condition', y='Duration', data=all_data,
                 palette="Set2", showmeans=True,
                 meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})

# 叠加散点，看到每一个对话轮次的具体分布
sns.stripplot(x='Condition', y='Duration', data=all_data,
              color=".3", size=4, alpha=0.4)

# 添加统计信息
plt.title(f'Distribution of Human Speech Duration per Turn', fontsize=16, pad=20)
plt.xlabel('Experimental Condition', fontsize=14)
plt.ylabel(f'Human Speech Duration (s)', fontsize=14)

# 旋转x轴标签，如果标签太长
#plt.xticks(rotation=45, ha='right')

# 添加网格
plt.grid(True, alpha=0.3)

# 显示统计摘要
print("\n📊 统计摘要:")
summary_stats = all_data.groupby('Condition')['Duration'].describe()
print(summary_stats)

plt.tight_layout()
plt.show()