import re
import pandas as pd
import os
from datetime import datetime

# ==================== 配置部分 ====================
BASE_PATH = "/media/crystal/KINGSTON/qwe"

GROUPS_CONFIG = {
    "实验组_E": {
        "path_template": f"{BASE_PATH}/2/haru_conversation_logs/{{user_id}}/full_terminal_output.log",
        "user_ids": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    },
    "对照组_C2": {
        "path_template": f"{BASE_PATH}/no_gaze/haru_conversation_logs/{{user_id}}/full_terminal_output.log",
        "user_ids": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    },
    "对照组_C1": {
        "path_template": f"{BASE_PATH}/speak/haru_conversation_logs/{{user_id}}/full_terminal_output.log",
        "user_ids": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    }
}

# ==================== 工具函数 ====================
def parse_timestamp(timestamp_str):
    """将 [HH:MM:SS.mmm] 转换为秒数"""
    # 移除方括号
    ts = timestamp_str.strip('[]')
    # 解析为datetime
    dt = datetime.strptime(ts, '%H:%M:%S.%f')
    # 转换为总秒数
    total_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    return total_seconds

def calculate_user_gap(log_file_path):
    """计算单个用户的User Gap"""
    gaps = []
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        robot_start = None
        robot_duration = None
        
        print(f"  调试: 检查文件 {log_file_path} 的前20行...")
        for i, line in enumerate(lines[:20]):
            print(f"    行{i+1}: {line.strip()}")
        
        for line in lines:
            # 匹配机器人开始说话
            robot_match = re.search(r'\[([^\]]+)\] \[机器人\]', line)
            if robot_match:
                robot_start = parse_timestamp(robot_match.group(1))
                print(f"    找到机器人开始: {robot_match.group(1)}")
                continue
            
            # 匹配机器人说话时长
            duration_match = re.search(r'🔊 机器人说话时长: ([0-9.]+)s', line)
            if duration_match:
                robot_duration = float(duration_match.group(1))
                print(f"    找到机器人时长: {robot_duration}s")
                continue
            
            # 匹配用户开始说话 (支持多种格式)
            user_match = re.search(r'\[([^\]]+)\] (?:\[用户\]|🎤 用户说:)', line)
            if user_match and robot_start is not None and robot_duration is not None:
                user_start = parse_timestamp(user_match.group(1))
                print(f"    找到用户开始: {user_match.group(1)}")
                # 计算Gap: 用户开始 - (机器人开始 + 机器人时长)
                robot_end = robot_start + robot_duration
                gap = user_start - robot_end
                print(f"    计算Gap: {gap:.2f}s")
                if gap > 0:  # 只记录正值
                    gaps.append(gap)
                # 重置
                robot_start = None
                robot_duration = None
    
    except Exception as e:
        print(f"  读取文件错误: {e}")
    
    print(f"  总共找到 {len(gaps)} 个有效Gap")
    return gaps

# ==================== 主程序 ====================
print("开始计算User Gap")
print("=" * 50)

all_results = []

for group_name, config in GROUPS_CONFIG.items():
    print(f"\n【{group_name}】")
    
    for user_id in config["user_ids"]:
        file_path = config["path_template"].format(user_id=user_id)
        
        if os.path.exists(file_path):
            gaps = calculate_user_gap(file_path)
            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                print(f"  用户 {user_id}: {len(gaps)} 个回合, 平均Gap: {avg_gap:.2f}s")
                
                # 保存每个Gap
                for i, gap in enumerate(gaps):
                    all_results.append({
                        'Group': group_name,
                        'User_ID': user_id,
                        'Turn': i + 1,
                        'User_Gap': gap
                    })
            else:
                print(f"  用户 {user_id}: 未找到有效Gap数据")
        else:
            print(f"  用户 {user_id}: 文件不存在 - {file_path}")

# ==================== 保存结果 ====================
if all_results:
    df = pd.DataFrame(all_results)
    output_file = "/home/crystal/音乐/qwe/qwe9119/analyse/user_gap_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ 结果已保存到: {output_file}")
    
    # 显示摘要
    print("\n" + "=" * 50)
    print("User Gap 摘要:")
    summary = df.groupby('Group')['User_Gap'].agg(['mean', 'std', 'count'])
    print(summary)
else:
    print("\n✗ 没有找到任何Gap数据")

print("\n计算完成！")