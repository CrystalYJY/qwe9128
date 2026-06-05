import pandas as pd
import os

# ==================== 配置部分 ====================
BASE_PATH = "/media/crystal/KINGSTON/qwe"

GROUPS_CONFIG = {
    "实验组_E": {
        "path_template": f"{BASE_PATH}/2/haru_conversation_logs/{{user_id}}/conversation_turns.csv",
        "output_dir": f"{BASE_PATH}/2",
        "user_ids": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    },
    "对照组_C2": {
        "path_template": f"{BASE_PATH}/no_gaze/haru_conversation_logs/{{user_id}}/conversation_turns.csv",
        "output_dir": f"{BASE_PATH}/no_gaze",
        "user_ids": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    },
    "对照组_C1": {
        "path_template": f"{BASE_PATH}/speak/haru_conversation_logs/{{user_id}}/conversation_turns.csv",
        "output_dir": f"{BASE_PATH}/speak",
        "user_ids": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    }
}

# ==================== 拼接函数 ====================
def splice_csvs(group_name, config):
    print(f"\n开始拼接 {group_name} 的CSV文件...")
    all_dfs = []
    
    for user_id in config["user_ids"]:
        file_path = config["path_template"].format(user_id=user_id)
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # 添加用户ID列，便于区分
                df['user_id'] = user_id
                all_dfs.append(df)
                print(f"  ✓ 用户 {user_id}: 读取成功 ({len(df)} 行)")
            except Exception as e:
                print(f"  ✗ 用户 {user_id}: 读取失败 - {e}")
        else:
            print(f"  ✗ 用户 {user_id}: 文件不存在 - {file_path}")
    
    if all_dfs:
        # 拼接所有DataFrame
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # 保存到输出目录
        output_file = os.path.join(config["output_dir"], "combined_conversation_turns.csv")
        combined_df.to_csv(output_file, index=False)
        
        print(f"  ✓ 拼接完成！总行数: {len(combined_df)}")
        print(f"  ✓ 保存至: {output_file}")
    else:
        print(f"  ✗ {group_name} 没有有效文件，无法拼接")

# ==================== 主程序 ====================
print("开始CSV文件拼接任务")
print("=" * 50)

for group_name, config in GROUPS_CONFIG.items():
    splice_csvs(group_name, config)

print("\n" + "=" * 50)
print("所有拼接任务完成！")