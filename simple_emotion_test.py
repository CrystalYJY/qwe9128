#!/usr/bin/env python3
"""
简单的情感表情测试脚本
专注于基础状态机流程：IDLE → LISTENING → SPEAKING → WAITING → IDLE
"""

import rclpy
from rclpy.node import Node
from idmind_tabletop_msgs.srv import Routine
import time
import threading

class SimpleEmotionTest(Node):
    def __init__(self):
        super().__init__('simple_emotion_test')
        
        # 创建ROS2客户端
        self.routine_client = self.create_client(Routine, '/idmind_tabletop/execute_routine')
        
        # 检查服务是否可用（最多等待3秒）
        self.service_available = False
        wait_count = 0
        while not self.routine_client.wait_for_service(timeout_sec=1.0) and wait_count < 3:
            self.get_logger().info('等待 routine 服务...')
            wait_count += 1
        
        # 检查最终的服务状态
        if self.routine_client.service_is_ready():
            self.service_available = True
            print("✅ Routine服务已连接")
        else:
            self.service_available = False
            print("⚠️ Routine服务未找到，将使用模拟模式")
        
        # 状态机
        self.state = "IDLE"
        
        # 调试配置 - VLM最终只需要5种intent：find_object, yes, no, bow, chat
        self.debug_intent = "chat"  # find_object/yes/no/bow/chat
        self.debug_emotion = "happy"  # happy/sad/angry/surprised/shy/worried/playful
        self.emotion_intensity = "medium"  # low/medium/high
        
        # 8种情感 × 3种强度 = 24种表情组合（基于实际routine数据）
        self.emotion_routines = {
            "happy": {
                "low": [2032, 2068],        # giggle(轻笑), friendly_smile(礼貌微笑)
                "medium": [2020, 2061, 2087], # amused_A(觉得有趣), harumoji_wide_smile(咧嘴笑), harumoji_tongue_out(吐舌头)
                "high": [2006, 2076]        # harumoji_laughing(开怀大笑), laughter(大笑)
            },
            "sad": {
                "low": [2015],              # sadMild(轻度伤心)
                "medium": [2037, 2084],     # sad(普通悲伤), sympathetic(共情)
                "high": [2037]              # sad作为高强度
            },
            "angry": {
                "low": [2033],              # grumpy_A(脾气差、轻微不满)
                "medium": [2009],           # complaint(抱怨、不满)
                "high": [2021]              # harumoji_angry(愤怒、争论)
            },
            "surprised": {
                "low": [2124],              # question_mark_green(表示疑问)
                "medium": [2059, 2124],     # surprsed606(普通惊讶), question_mark_green
                "high": [2038]              # harumoji_shocked(突然的震惊)
            },
            "curious": {
                "low": [2124],              # question_mark_green(疑问、好奇)
                "medium": [2124, 2008],     # question_mark_green, shortPeek(好奇地观察)
                "high": [2059]              # surprsed606(强烈好奇)
            },
            "shy": {
                "low": [2063, 2050],             # harumoji_blushing(脸红)
                "medium": [2081, 2082],     # shy(害羞), harumoji_shy(专属害羞动作)
                "high": [2082]              # harumoji_shy(强度版本)
            },
            "worried": {
                "low": [2088],              # unsure(犹豫、拿不准)
                "medium": [2017, 2088],     # worriedA(担心、焦虑), unsure
                "high": [2017]              # worriedA(强度版本)
            },
            "playful": {
                "low": [2092],              # harumoji_wink(眨眼、调皮)
                "medium": [2087],           # harumoji_tongue_out(吐舌头、调皮动作)
                "high": [2087]              # harumoji_tongue_out(强度版本)
            },
        
        }
        
        # 社交动作(基于实际routine数据) 
        self.social_routines = {
            "yes": [2093],              # yes(明确肯定)
            "no": [2080],               # no(明确否定) 
            "bow": [2052]               # 可用作礼貌动作
        }
        
        # FINDING状态专用routine(基于实际数据)
        self.finding_routines = {
            "lookaround": [2002],    # lookaround - 环顾、观察
            "peek": [2008]           # shortPeek - 偷看、短时观察
        }
        
        # 倾听和等待动作(由ASR/TTS信号控制，暂时保留备用)
        self.listening_routines = {
            "listening": [2005, 2077, 2078, 2079],  # shortListening, listening_b/c/d
            "waiting": [2018]                       # awaitingReply(等待对方回应)
        }
        
        # 当前强度设置
        self.emotion_intensity = "medium"
        
        print("🤖 简单情感测试系统启动")
        print(f"📋 当前配置: 意图={self.debug_intent}, 情感={self.debug_emotion}")
        
        #print("💡 在终端输入文字开始测试...")
        
    def transition_state(self, new_state, reason=""):
        """状态转换"""
        old_state = self.state
        self.state = new_state
        print(f"🔄 状态转换: {old_state} → {new_state} | {reason}")
    
    def call_routine(self, routine_id):
        """调用表情routine（同步等待完成）"""
        try:
            if not self.service_available:
                print(f"🎭 [模拟] 执行表情routine: {routine_id}")
                import time
                time.sleep(1)  # 模拟模式下简单延时
                return True
                
            if not self.routine_client.service_is_ready():
                print("❌ Routine服务未准备就绪")
                return False
            
            request = Routine.Request()
            # 正确的属性名是routine
            request.routine = routine_id
            
            print(f"🎭 同步执行表情routine: {routine_id}")
            
            # 同步调用并等待完成
            future = self.routine_client.call_async(request)
            
            # 等待结果，使用rclpy.spin_until_future_complete
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            
            if future.done():
                result = future.result()
                print(f"✅ Routine {routine_id} 执行完成")
                return True
            else:
                print(f"⏰ Routine {routine_id} 执行超时")
                return False
            
        except Exception as e:
            print(f"❌ 调用routine失败: {e}")
            return False
    
    def execute_emotion(self, emotion, intensity=None):
        """执行情感表情（支持强度选择）"""
        if intensity is None:
            intensity = self.emotion_intensity
            
        print(f"🎭 表达情感: {emotion} (强度: {intensity})")
        
        # 检查情感是否存在
        if emotion not in self.emotion_routines:
            print(f"❌ 未知情感: {emotion}")
            return False
        
        # 检查强度是否存在
        if intensity not in self.emotion_routines[emotion]:
            print(f"❌ 未知强度: {intensity}，使用medium")
            intensity = "medium"
        
        # 随机选择一个routine
        import random
        available_routines = self.emotion_routines[emotion][intensity]
        routine_id = random.choice(available_routines)
        
        return self.call_routine(routine_id)
    
    def execute_social_action(self, action):
        """执行社交动作"""
        print(f"🤝 社交动作: {action}")
        
        if action not in self.social_routines:
            print(f"❌ 未知社交动作: {action}")
            return False
        
        # 随机选择一个routine
        import random
        available_routines = self.social_routines[action]
        routine_id = random.choice(available_routines)
        
        return self.call_routine(routine_id)
    
    def handle_user_input(self, text):
        """处理用户输入的完整流程"""
        print(f"\n👤 用户输入: {text}")
        
        # 1. IDLE → LISTENING
        self.transition_state("LISTENING", "检测到用户说话")
        time.sleep(0.5)  # 模拟听取过程
        
        # 2. 根据intent分流到不同状态
        if self.debug_intent == "find_object":
            # FINDING流程
            self.transition_state("FINDING", f"执行寻找物体，表达{self.debug_emotion}情感")
            self._handle_finding_flow()
        elif self.debug_intent in ["yes", "no", "bow", "chat"]:
            # SPEAKING流程  
            self.transition_state("SPEAKING", f"执行{self.debug_intent}动作")
            self._handle_speaking_flow(text)
        else:
            print(f"❌ 未知intent: {self.debug_intent}")
            return
        
        # 最终转到WAITING状态
        self.transition_state("WAITING", "等待用户回应")
        
        # 设置定时器，3秒后回到IDLE
        def back_to_idle():
            time.sleep(3)
            if self.state == "WAITING":
                self.transition_state("IDLE", "超时回到空闲状态")
        
        threading.Thread(target=back_to_idle, daemon=True).start()
    
    def _handle_finding_flow(self):
        """处理FINDING状态的流程"""
        import random
        
        # 1. 先表达情感（通常是curious）
        self.execute_emotion(self.debug_emotion, self.emotion_intensity)
        
        # 2. 执行寻找动作
        finding_actions = ["lookaround", "peek"]
        action = random.choice(finding_actions)
        
        print(f"🔍 执行寻找动作: {action}")
        routine_id = random.choice(self.finding_routines[action])
        self.call_routine(routine_id)
    
    def _handle_speaking_flow(self):
        """处理SPEAKING状态的流程"""
        # 直接表达情感，因为语音回复由主程序处理
        if self.debug_emotion and self.emotion_intensity:
            self.execute_emotion(self.debug_emotion, self.emotion_intensity)
    
    def change_config(self, intent=None, emotion=None, intensity=None):
        """修改配置"""
        if intent:
            self.debug_intent = intent
            print(f"✅ 意图已更新为: {intent}")
        if emotion:
            self.debug_emotion = emotion
            print(f"✅ 情感已更新为: {emotion}")
        if intensity:
            self.emotion_intensity = intensity
            print(f"✅ 强度已更新为: {intensity}")
    
    def run_interactive(self):
        """交互式运行"""
        print("\n" + "="*50)
        print("🎮 交互模式启动")
        print("输入指令 (模拟VLM输出格式):")
        print("  <intent> <emotion> [intensity]")
        print("  示例:")
        print("    chat happy medium     - 聊天，开心情感，中等强度")
        print("    find_object surprised low  - 寻找物体，惊讶情感，低强度")
        print("    yes none none         - 点头，无情感")
        print("    no none none          - 摇头，无情感")
        print("    bow none none         - 鞠躬，无情感")
        print("  其他指令:")
        print("    exit - 退出")
        print("    idle/listening/speaking/waiting - 手动状态切换")
        print(f"可选情感: {list(self.emotion_routines.keys())}")
        print(f"可选强度: low, medium, high")
        print("="*50)
        
        while True:
            try:
                user_input = input(f"\n[{self.state}] > ").strip()
                
                if user_input.lower() == "exit":
                    print("👋 退出程序")
                    break
                
                elif user_input.lower() in ["idle", "listening", "speaking", "waiting"]:
                    self.transition_state(user_input.upper(), "手动状态切换")
                
                elif user_input:
                    # 解析VLM格式: <intent> <emotion> [intensity]
                    parts = user_input.split()
                    if len(parts) >= 1:
                        intent = parts[0]
                        emotion = parts[1] if len(parts) > 1 and parts[1] != "none" else None
                        intensity = parts[2] if len(parts) > 2 and parts[2] != "none" else "medium"
                        
                        # 验证intent
                        if intent not in ["find_object", "yes", "no", "bow", "chat"]:
                            print(f"❌ 未知intent: {intent}")
                            print("   可用intent: find_object, yes, no, bow, chat")
                            continue
                        
                        # 验证emotion（如果提供的话）
                        if emotion and emotion not in self.emotion_routines:
                            print(f"❌ 未知情感: {emotion}")
                            print(f"   可选情感: {list(self.emotion_routines.keys())}")
                            continue
                        
                        # 更新配置
                        self.debug_intent = intent
                        if emotion:
                            self.debug_emotion = emotion
                        if intensity:
                            self.emotion_intensity = intensity
                        
                        print(f"✅ 解析VLM输出: intent={intent}, emotion={emotion}, intensity={intensity}")
                        
                        # 只有在IDLE状态才能开始新的对话流程
                        if self.state == "IDLE":
                            self.handle_user_input(f"执行{intent}动作")
                        else:
                            print(f"⚠️ 当前状态为{self.state}，请等待回到IDLE状态")
                    else:
                        print("❌ 请使用格式: <intent> <emotion> [intensity]")
                
            except KeyboardInterrupt:
                print("\n👋 程序中断退出")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")

def main():
    rclpy.init()
    
    try:
        node = SimpleEmotionTest()
        
        # 启动ROS2节点处理（后台线程）
        def spin_node():
            rclpy.spin(node)
        
        spin_thread = threading.Thread(target=spin_node, daemon=True)
        spin_thread.start()
        
        # 等待服务可用（如果可用的话）
        if node.service_available:
            print("⏳ 等待routine服务...")
            while not node.routine_client.service_is_ready():
                time.sleep(0.1)
            print("✅ Routine服务已连接")
        else:
            print("🔧 运行在模拟模式下")
        
        # 运行交互式测试
        node.run_interactive()
        
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
