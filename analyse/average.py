import pandas as pd

# ==================== 配置部分 ====================
# 1. 设置你要计算平均值的列名
COLUMN_TO_AVERAGE = "stage_B_vlm_inference (s)"  # 请改为你CSV文件中的实际列名

# 2. 配置实验组和对照组的路径
BASE_PATH = "/media/crystal/KINGSTON/qwe"

GROUPS_CONFIG = {
    "实验组_E": {
        "path_template": f"{BASE_PATH}/2/haru_conversation_logs/{{user_id}}/conversation_turns.csv",
        "user_ids": ["1", "2", "3", "4", "5"]
    },
    "对照组_C2": {
        "path_template": f"{BASE_PATH}/no_gaze/haru_conversation_logs/{{user_id}}/conversation_turns.csv",
        "user_ids": ["1", "2", "3", "4", "5"]
    },
    "对照组_C1": {
        "path_template": f"{BASE_PATH}/speak/haru_conversation_logs/{{user_id}}/conversation_turns.csv",
        "user_ids": ["1", "2", "3", "4", "5"]
    }
}

# ==================== 计算每个用户的平均值 ====================
print(f"开始计算列 '{COLUMN_TO_AVERAGE}' 的平均值")
print("=" * 50)  # 这行会打印50个等号，作为分隔线让输出更清晰

# 存储所有结果
all_results = {}

for group_name, config in GROUPS_CONFIG.items():
    print(f"\n【{group_name}】")
    group_results = {}
    
    for user_id in config["user_ids"]:
        # 构建文件路径
        file_path = config["path_template"].format(user_id=user_id)
        
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 检查列是否存在
            if COLUMN_TO_AVERAGE in df.columns:
                # 计算该用户的平均值
                average_value = df[COLUMN_TO_AVERAGE].mean()
                group_results[user_id] = average_value
                print(f"  用户 {user_id}: {average_value:.4f}")
            else:
                print(f"  用户 {user_id}: 列 '{COLUMN_TO_AVERAGE}' 不存在")
                group_results[user_id] = None
                
        except FileNotFoundError:
            print(f"  用户 {user_id}: 文件未找到 - {file_path}")
            group_results[user_id] = None
        except Exception as e:
            print(f"  用户 {user_id}: 读取错误 - {e}")
            group_results[user_id] = None
    
    # 保存该组的结果
    all_results[group_name] = group_results

# ==================== 可选：计算每组平均值 ====================
print("\n" + "=" * 50)  # 另一个分隔线
print("组平均（可选）:")

for group_name, results in all_results.items():
    # 过滤掉无效值（None）
    valid_values = [v for v in results.values() if v is not None]
    
    if valid_values:
        group_avg = sum(valid_values) / len(valid_values)
        print(f"{group_name}: {group_avg:.4f} (基于{len(valid_values)}个有效用户)")
    else:
        print(f"{group_name}: 无有效数据")

print("\n计算完成！")