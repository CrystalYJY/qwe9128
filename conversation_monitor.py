#!/usr/bin/env python3
"""
对话监控终端 - 实时显示用户输入和机器人中文回复
使用方法：在新终端运行 python conversation_monitor.py
"""

import os
import time
import sys
from datetime import datetime
import re

class ConversationMonitor:
    def __init__(self):
        self.base_log_dirs = [
            "/media/crystal/KINGSTON/qwe/2/haru_conversation_logs",
            "/media/crystal/KINGSTON/qwe/no_gaze/haru_conversation_logs",
            "/media/crystal/KINGSTON/qwe/speak/haru_conversation_logs"
        ]
        self.current_session_dir = None
        self.terminal_log_path = None
        self.last_user_input = None
        self.last_chinese_reply = None
        
    def find_latest_session(self):
        """查找最新的会话目录（跨多个实验目录）"""
        all_sessions = []
        for base_dir in self.base_log_dirs:
            if not os.path.exists(base_dir):
                continue
            
            sessions = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                       if os.path.isdir(os.path.join(base_dir, d))]
            all_sessions.extend(sessions)
        
        if not all_sessions:
            return None
        
        # 按目录的最后修改时间排序，获取最新的
        all_sessions.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_session = all_sessions[0]
        
        # 查找 full_terminal_output.log 文件
        terminal_log = os.path.join(latest_session, "full_terminal_output.log")
        if os.path.exists(terminal_log):
            return terminal_log
        
        return None
    
    def parse_log_line(self, line):
        """解析日志行，提取用户输入和中文回复"""
        # 去除时间戳前缀（如果存在）
        line = re.sub(r'^\[\d{2}:\d{2}:\d{2}\.\d+\]\s*', '', line)
        
        # 匹配用户输入（多种可能的格式）
        user_patterns = [
            r"🎤\s*用户说[:：]\s*(.+?)(?:\s*\(说话时长|$)",
            r"👤\s*用户[:：]\s*(.+)",
            r"\[用户\]\s*[:：]?\s*(.+)",
            r"User\s*[:：]\s*(.+)",
        ]
        
        for pattern in user_patterns:
            match = re.search(pattern, line)
            if match:
                return "user", match.group(1).strip()
        
        # 匹配中文翻译（处理多个emoji的情况，包括emoji之间的空格）
        chinese_patterns = [
            r"🇨🇳[\s🇨🇳]*\[中文\]\s*(.+)",  # 匹配任意数量的🇨🇳和空格组合
            r"\[中文翻译\]\s*(.+)",
            r"Chinese[:：]\s*(.+)",
            r"^\d+\.\s*Chinese[:：]\s*(.+)", # 匹配 "2. Chinese: ..."
        ]
        
        for pattern in chinese_patterns:
            match = re.search(pattern, line)
            if match:
                return "chinese", match.group(1).strip()
        
        return None, None
    
    def monitor_loop(self):
        """主监控循环"""
        print("=" * 70)
        print("📺 对话监控终端")
        print("=" * 70)
        print("⏳ 等待会话日志...")
        print()
        
        file_handle = None
        
        while True:
            # 检查是否有新的会话目录
            new_log = self.find_latest_session()
            if new_log and new_log != self.terminal_log_path:
                # 关闭旧文件
                if file_handle:
                    file_handle.close()
                    file_handle = None
                
                self.terminal_log_path = new_log
                session_name = os.path.basename(os.path.dirname(new_log))
                print(f"📁 找到新会话: {session_name}")
                print("-" * 70)
                print()
                
                # 打开新文件，先从头读取已有内容
                try:
                    file_handle = open(self.terminal_log_path, 'r', encoding='utf-8', errors='ignore')
                    print(f"🎬 正在读取历史记录...")
                    
                    # 读取并处理已有的所有行
                    lines = file_handle.readlines()
                    for line in lines:
                        self.process_line(line)
                    
                    print(f"✨ 已同步，正在实时监控新消息...")
                    print()
                except Exception as e:
                    print(f"❌ 无法打开日志文件: {e}")
                    file_handle = None
            
            # 读取日志文件的新内容
            if file_handle:
                try:
                    # 持续读取新行
                    line = file_handle.readline()
                    if line:
                        self.process_line(line)
                    else:
                        # 没有新行，短时间休眠
                        time.sleep(0.1)
                except Exception as e:
                    # 文件可能被关闭
                    file_handle.close()
                    file_handle = None
            else:
                # 还没有找到会话，短时间休眠
                time.sleep(1.0)

    def process_line(self, line):
        """解析并显示单行日志"""
        msg_type, content = self.parse_log_line(line)
        
        if msg_type == "user":
            # 只有当内容变化时才打印，或者这是第一条
            if content != self.last_user_input:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] 👤 用户: {content}", flush=True)
                self.last_user_input = content
                
        elif msg_type == "chinese":
            # 只有当内容变化时才打印
            if content != self.last_chinese_reply:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] 🤖 机器人: {content}", flush=True)
                print("-" * 40, flush=True) # 分隔符
                self.last_chinese_reply = content

def main():
    try:
        monitor = ConversationMonitor()
        monitor.monitor_loop()
    except KeyboardInterrupt:
        print("\n\n👋 监控已停止")
        sys.exit(0)

if __name__ == "__main__":
    main()
