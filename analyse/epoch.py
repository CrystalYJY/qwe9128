import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os

# ==================== 配置部分 ====================
# 配置实验组和对照组的路径
BASE_PATH = "/media/crystal/KINGSTON/qwe"

GROUPS_CONFIG = {
    "实验组_E": {
        "path_template": f"{BASE_PATH}/2/haru_conversation_logs/{{user_id}}/metadata.json",
        "condition_name": "Exp (Gaze)",
        "user_ids": [str(i) for i in range(1, 11)],  # 用户1-10
        "manual_data": {  # 手动补充用户6-10的时间数据（秒）
            "6": 300,  # 请填写用户6的完成时间
            "7": 510,  # 请填写用户7的完成时间
            "8": 566,  # 请填写用户8的完成时间
            "9": 290,  # 请填写用户9的完成时间
            "10": 383  # 请填写用户10的完成时间
        }
    },
    "对照组_C2": {
        "path_template": f"{BASE_PATH}/no_gaze/haru_conversation_logs/{{user_id}}/metadata.json",
        "condition_name": "Ctrl (NoGaze)",
        "user_ids": [str(i) for i in range(1, 11)],  # 用户1-10
        "manual_data": {  # 手动补充用户6-10的时间数据（秒）
            "6": 334,  # 请填写用户6的完成时间
            "7": 900,  # 请填写用户7的完成时间
            "8": 540,  # 请填写用户8的完成时间
            "9": 580,  # 请填写用户9的完成时间
            "10": 630  # 请填写用户10的完成时间
        }
    },

    "对照组_C1": {
        "path_template": f"{BASE_PATH}/speak/haru_conversation_logs/{{user_id}}/metadata.json",
        "condition_name": "Ctrl (Speak)",
        "user_ids": [str(i) for i in range(1, 11)]  # 用户1-10
    }
   
}

def load_metadata_data():
    """从 metadata.json 文件加载真实数据"""
    all_data = []

    for group_name, config in GROUPS_CONFIG.items():
        print(f"📂 正在读取 {group_name} 的数据...")

        user_ids = config.get("user_ids", [])
        if not user_ids:
            print(f"⚠️  {group_name} 没有指定用户ID列表")
            continue

        print(f"   需要处理 {len(user_ids)} 个用户: {user_ids}")

        for user_id in user_ids:
            file_path = config["path_template"].format(user_id=user_id)

            if not os.path.exists(file_path):
                print(f"   ⚠️  用户 {user_id} 的文件不存在: {file_path}")
                
                # 检查是否有手动补充的数据
                manual_data = config.get("manual_data", {})
                if user_id in manual_data:
                    completion_time = manual_data[user_id]
                    print(f"   📝 使用手动补充数据: 用户 {user_id}, 时间={completion_time}s")
                    all_data.append({
                        'Condition': config["condition_name"],
                        'User_ID': user_id,
                        'Total_Turns': 0,  # 手动数据只补充时间，轮数设为0
                        'Completion_Time': completion_time,
                        'Group': group_name,
                        'File_Path': file_path,
                        'Data_Status': 'Manual'
                    })
                else:
                    # 仍然添加一条记录，但数据为0，表示缺失
                    all_data.append({
                        'Condition': config["condition_name"],
                        'User_ID': user_id,
                        'Total_Turns': 0,
                        'Completion_Time': 0,
                        'Group': group_name,
                        'File_Path': file_path,
                        'Data_Status': 'Missing'
                    })
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                # 提取所需数据
                total_turns = metadata.get('conversation_count', 0)

                # 计算完成时间（优先使用 duration_seconds，否则计算 end_time - start_time）
                completion_time = metadata.get('duration_seconds', 0)
                if completion_time == 0:
                    start_time = metadata.get('start_time', 0)
                    end_time = metadata.get('end_time', 0)
                    if start_time and end_time:
                        completion_time = end_time - start_time

                # 检查是否有手动补充的数据，如果有则覆盖
                manual_data = config.get("manual_data", {})
                if user_id in manual_data and manual_data[user_id] > 0:
                    completion_time = manual_data[user_id]
                    print(f"   📝 使用手动补充数据覆盖: 用户 {user_id}, 时间={completion_time}s")

                all_data.append({
                    'Condition': config["condition_name"],
                    'User_ID': user_id,
                    'Total_Turns': total_turns,
                    'Completion_Time': completion_time,
                    'Group': group_name,
                    'File_Path': file_path,
                    'Data_Status': 'Available'
                })

                print(f"   ✅ 用户 {user_id}: 轮数={total_turns}, 时间={completion_time:.1f}s")

            except Exception as e:
                print(f"   ❌ 读取用户 {user_id} 的文件失败 {file_path}: {e}")
                # 添加错误记录
                all_data.append({
                    'Condition': config["condition_name"],
                    'User_ID': user_id,
                    'Total_Turns': 0,
                    'Completion_Time': 0,
                    'Group': group_name,
                    'File_Path': file_path,
                    'Data_Status': 'Error'
                })

    return pd.DataFrame(all_data)

# 读取真实数据
print("🔍 开始读取实验数据...")
df = load_metadata_data()

if df.empty:
    print("❌ 没有找到任何数据文件，请检查路径配置")
    exit(1)

print(f"\n📊 数据概览:")
print(f"总样本数: {len(df)}")
print(f"分组统计:")
print(df.groupby('Condition').size())
print(f"\n数据预览:")
print(df.head())

# 设置绘图风格
sns.set_theme(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- 图 1: Total Turns (对话轮数) ---
sns.boxplot(x='Condition', y='Total_Turns', data=df, ax=ax1, palette='Set2', width=0.5, showmeans=True,
            meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})
sns.swarmplot(x='Condition', y='Total_Turns', data=df, ax=ax1, color=".25", alpha=0.6)
ax1.set_title('Total Conversation Turns', fontsize=14, fontweight='bold')
ax1.set_ylabel('Number of Turns')

# --- 图 2: Total Completion Time (完成时间) ---
sns.boxplot(x='Condition', y='Completion_Time', data=df, ax=ax2, palette='Set2', width=0.5, showmeans=True,
            meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})
sns.swarmplot(x='Condition', y='Completion_Time', data=df, ax=ax2, color=".25", alpha=0.6)
ax2.set_title('Total Task Completion Time', fontsize=14, fontweight='bold')
ax2.set_ylabel('Time (seconds)')

plt.tight_layout()
plt.show()