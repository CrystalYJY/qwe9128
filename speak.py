import threading
import os
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from xunfei_asr_adapter import XunfeiASRAdapter as ASRModule
from image_save_module import ImageSaveModule
from doubao_api_module import DoubaoAPIModule
from robot_env import RobotEnv
# ===== 已禁用：表情模块 =====
# from simple_emotion_test import SimpleEmotionTest
from conversation_logger import ConversationSessionLogger
from strawberry_ros_msgs.msg import Skeletons
from types import SimpleNamespace
import time
import random
from typing import Optional


class MainNodeWithGUI(Node):
    def __init__(self):
        super().__init__('main_node_gui')
        self.asr = ASRModule(mic_device="plughw:3,0")  # 使用 USB 麦克风 (card 2, device 0)
        self.image_saver = ImageSaveModule()
        # speak.py 作为盲人对照组，使用 blind prompt（不涉及视觉）
        self.doubao_api = DoubaoAPIModule(prompt_profile="blind")
        self.env = RobotEnv()
        from look_controller import LookController
        self.look_controller = LookController(self.env.motor_node, self)
        # ===== 已禁用：表情模块 =====
        # self.emotion_module = SimpleEmotionTest()
        self.emotion_module = None
        self.conversation_state = "IDLE"
        # 默认关闭刷屏调试；需要排障时再临时打开。
        self.DEBUG_MODE = False
        # 盲人模式不需要 VLM 详细调试输出
        self.VLM_VERBOSE = False
        self.data_ready = False
        self.resume_tracking_timer = None
        self.prev_text = ""
        self.tts_gaze_mode = None  # 记录TTS期间的注视模式: "object", "human", None
        self.tts_gaze_target = None  # 记录TTS期间的注视目标

        # 对照组（盲人视角）：不上传图像，使用 blind prompt
        self.TEXT_ONLY_MODE = True
        
        # 初始化日志系统
        self.session_logger = ConversationSessionLogger(
            base_log_dir="/media/crystal/KINGSTON/qwe/speak/haru_conversation_logs"
        )
        # 捕获 stdout/stderr，保证"完整终端输出"可找回
        self.session_logger.enable_full_terminal_capture()

        # 对话轮数统计（1轮 = 用户一句 + 机器人一句）
        self.dialogue_rounds = 0
        self._dialogue_rounds_lock = threading.Lock()

        # 用户需求：不再落盘 ROS_LOG_DIR 到会话目录（避免生成 ros_logs 文件夹）。
        # 若仍需要把 ROS2 logger 输出并入 full_terminal_output.log，建议在运行环境层面配置。
        
        # 初始化 Joint Attention 评估器(简化为30度固定阈值)
        self.ja_evaluator = JointAttentionEvaluator(
            # person-frame / nose+eyes 现场噪声下，15° 过严，基本不可能判成功。
            # 这里先放宽到 55° 便于回归验证（后续再根据现场收紧）。
            angle_threshold_deg=55.0,
            reaction_window=3.0,
            robot_motion_delay=0.8,
            min_shared_duration=0.5,
            logger=self.session_logger
        )

        # -------------------------------
        # JA 低频坐标诊断（默认不刷屏）
        # -------------------------------
        # 每 1s 打一次，用于确认 gaze_dir / target_dir 是否处于同一坐标口径。
        # 如需关闭：改回 False。
        self.ja_evaluator._debug_coord_diag = False

        # person-frame 调试：每秒输出一次是否成功使用人体坐标系（需要定位问题时再打开）
        self.ja_evaluator._debug_person_frame = False
        # 可选对照（先别开，除非需要验证镜像/坐标置换）：
        # self.ja_evaluator._debug_mirror_target_y = True
        # self.ja_evaluator._debug_opencv_object_for_compare = True
        
        # 订阅人体检测话题
        self.skeleton_sub_for_ja = self.create_subscription(
            Skeletons,
            '/strawberry/azure/skeletons',
            self.skeleton_callback_for_ja,
            10
        )
        
        # 不再需要图像缓存（每轮都发新图片）
        # self.cached_image = None
        # self.cached_image_base64 = None
        # self.last_image_upload_time = None
        
        # 移除GUI相关代码
        threading.Thread(target=self.init_sensor_check, daemon=True).start()
        self.look_controller.start_head_tracking()
        if self.DEBUG_MODE:
            print("👁️ 人体追踪已启动")
        
        # 订阅TTS状态用于注视控制
        from idmind_tabletop_msgs.msg import TTSStatus
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.tts_status_sub = self.create_subscription(
            TTSStatus,
            "/idmind_tabletop/tts_status",
            self.tts_status_callback,
            qos_profile,
        )
        print("📡 已订阅 TTS 状态话题: /idmind_tabletop/tts_status")

        # 订阅rosout用于快速TTS时长计算
        from rcl_interfaces.msg import Log as RosoutLog
        self._rosout_sub = self.create_subscription(
            RosoutLog,
            "/rosout",
            self.rosout_callback_for_tts,
            qos_profile,
        )
        print("📡 已订阅 rosout 话题用于TTS时长计算")

        # TTS 播放时长追踪
        # 运行时属性：不用行内类型批注，避免部分解析器报"此语句不支持类型批注"
        self._tts_play_start = None
        self._tts_log_send_ts = None  # rosout记录的发送时间
        self._tts_log_done_ts = None  # rosout记录的完成时间
        self._pending_tts_start = None
        self._last_tts_duration = None
        self._current_conversation_index = None  # 当前对话索引，用于更新TTS时长
        
        threading.Thread(target=self.voice_input_loop, daemon=True).start()
        if self.DEBUG_MODE:
            print("🤖 机器人对话系统已启动")
            print("🎤 语音监听已启动，请说话...")

    def init_sensor_check(self):
        if self.DEBUG_MODE:
            print("🔍 检查传感器...")
        for i in range(15):
            status = self.image_saver.check_sensor_status()
            if status["image_ready"] and status["depth_ready"] and status["camera_info_ready"]:
                self.data_ready = True
                if self.DEBUG_MODE:
                    print("✅ 传感器就绪")
                return
            time.sleep(1)
        if self.DEBUG_MODE:
            print("⚠️ 传感器检测超时")
            print("❌ 传感器检测超时，请检查：")
            print("   1. Azure Kinect 设备连接")
            print("   2. ROS2 驱动程序运行状态")
            print("   3. 相关话题发布状态")

    def skeleton_callback_for_ja(self, msg: Skeletons):
        """接收骨骼数据,用于 Joint Attention 评估"""
        timestamp = time.time()
        pseudo_people = []
        for idx, skeleton in enumerate(msg.skeletons):
            # 使用 tracking_id 作为人员ID,没有则使用索引
            person_id = getattr(skeleton, 'tracking_id', idx)
            pseudo_people.append(SimpleNamespace(id=person_id, skeleton=skeleton))

        # 默认不打印逐帧骨骼调试信息（太刷屏）。
        # 需要时可在运行过程中临时设置：self.JA_DEBUG_SKELETON = True
        if getattr(self, "JA_DEBUG_SKELETON", False) and self.DEBUG_MODE:
            if not hasattr(self, "_ja_debug_parts_logged"):
                first = msg.skeletons[0] if msg.skeletons else None
                if first:
                    part_names = [bp.name for bp in first.body_parts]
                    print(f"🧩 [JA] 首次骨骼部位: {part_names}")
                self._ja_debug_parts_logged = True

            # 低频节流：每 0.5s 最多打印一次
            if not hasattr(self, "_ja_dbg_last_print"):
                self._ja_dbg_last_print = 0.0
            if timestamp - self._ja_dbg_last_print >= 0.5:
                self._ja_dbg_last_print = timestamp
                first = msg.skeletons[0] if msg.skeletons else None
                coords = None
                if first:
                    for part in first.body_parts:
                        if part.name.lower() == "nose":
                            coords = (
                                part.pose.position.x,
                                part.pose.position.y,
                                part.pose.position.z,
                            )
                            break
                print(
                    "🦴 [JA] skeleton_callback: count={}, active={}, nose={}".format(
                        len(msg.skeletons), self.ja_evaluator.robot_gaze_active, coords
                    )
                )
        
        self.ja_evaluator.on_people_update(timestamp, pseudo_people)

    def tts_status_callback(self, msg):
        """TTS状态回调 - 控制说话时的注视行为"""
        # PLAYING = 2, STOPPED = 1 (不是 0！)
        is_playing = (msg.status == 2)

        now = time.time()

        if is_playing:
            # TTS 开始播放：记录真实播放开始时间（优先使用 pending 开始时间）
            if self._tts_play_start is None:
                self._tts_play_start = self._pending_tts_start or now
                self._pending_tts_start = None
                # 通知ASR抑制识别，传递估算时长用于自动解除抑制
                estimated_duration = getattr(self, '_estimated_tts_duration', None)
                self.asr.set_tts_playing(True, estimated_duration)
        else:
            # TTS 停止播放
            # 清理播放状态
            self._tts_play_start = None
            self._pending_tts_start = None
            
            # 通知ASR恢复识别
            self.asr.set_tts_playing(False)
            
            # 调用恢复追踪逻辑
            self._process_tts_status_stopped()

            # 原有:TTS 停止播放,恢复人体追踪(仅在原来的条件下)

    def rosout_callback_for_tts(self, msg):
        """处理rosout日志用于快速TTS时长计算"""
        if msg.name != "tts_command_node":
            return
            
        text = msg.msg
        if not text:
            return
        
        # ✅ 提取日志时间戳
        ts = float(msg.stamp.sec) + float(msg.stamp.nanosec) * 1e-9
            
        if "已发送语音消息" in text:
            # 提取TTS文本内容并估算时长
            import re
            match = re.search(r'🗣️ 已发送语音消息: "(.*?)"', text)
            if match:
                tts_text = match.group(1)
                char_count = len(tts_text)
                # 英文约15字符/秒，增加1秒缓冲避免提前解除抑制
                estimated_duration = char_count / 15.0 + 1.0
                self._estimated_tts_duration = estimated_duration
            else:
                self._estimated_tts_duration = None
            self._tts_log_send_ts = ts

    def _process_tts_status_stopped(self):
        """处理TTS停止状态的逻辑"""
        if self.tts_gaze_mode == "object" and self.tts_gaze_target:
            # 如果之前在看物体，保持注视一段时间再恢复人体追踪
            self._cancel_resume_tracking_timer()

            grace = getattr(self.ja_evaluator, "post_gaze_grace", 0.0)
            if grace > 0:
                if self.DEBUG_MODE:
                    print(
                        f"⏳ TTS结束，继续看向物体 {grace:.1f}s 后再恢复人体追踪"
                    )
            else:
                if self.DEBUG_MODE:
                    print("🔄 TTS结束，准备恢复人体追踪")

            # 先通知 JA 评估器，开始进入补偿窗口（仅当 episode 已启动）
            if getattr(self.ja_evaluator, "robot_gaze_active", False):
                self.ja_evaluator.on_robot_gaze_end(time.time())
            else:
                # 常见原因：chat 模式仅 look_at_object 但未启动 on_robot_gaze_start。
                # 这里静默跳过，避免误导性告警刷屏。
                pass

                def _resume_tracking_after_grace():
                    try:
                        now2 = time.time()
                        self.ja_evaluator.on_grace_period_timeout(now2)
                        if self.DEBUG_MODE:
                            print("👀 Grace 结束，恢复人体追踪")
                        self.look_controller.start_head_tracking()
                    finally:
                        self.resume_tracking_timer = None

                if grace > 0:
                    self.resume_tracking_timer = threading.Timer(
                        grace, _resume_tracking_after_grace
                    )
                    self.resume_tracking_timer.start()
                else:
                    _resume_tracking_after_grace()

                self.tts_gaze_mode = None
                self.tts_gaze_target = None

    def _cancel_resume_tracking_timer(self):
        if self.resume_tracking_timer:
            self.resume_tracking_timer.cancel()
            self.resume_tracking_timer = None

    def _setup_tts_gaze_control(self, intent):
        """设置TTS期间的注视控制
        Args:
            intent: 当前意图（gaze_object, chat等）
        """
        # 如果当前在看物体，记录状态
        if self.look_controller.current_mode == "object" and self.look_controller.object_target:
            self.tts_gaze_mode = "object"
            self.tts_gaze_target = self.look_controller.object_target
            if self.DEBUG_MODE:
                print("💬 TTS期间保持注视物体")

    def voice_input_loop(self):
        while True:
            # 不再调用 rclpy.spin_once
            if not self.data_ready:
                time.sleep(1)
                continue
                
            text = self.asr.get_text()
            
            if text and text != self.prev_text:
                asr_timestamp = time.time()
                # 尝试获取该次识别对应的人声时长
                try:
                    speech_len = self.asr.get_last_speech_duration()
                except Exception:
                    speech_len = None

                if speech_len is not None:
                    # 说话时长的校正已在 ASRModule.get_last_speech_duration() 源头完成
                    msg = f"\n{'='*60}\n🎤 用户说: {text} (说话时长: {speech_len:.2f}s)\n{'='*60}"
                else:
                    msg = f"\n{'='*60}\n🎤 用户说: {text}\n{'='*60}"
                print(msg)
                self.session_logger.log_terminal_output(msg)
                
                self.prev_text = text
                # 重要：清空 ASR 文本避免重复处理
                self.asr.text = ""
                # 立即开始处理，不等待（传入说话时长）
                threading.Thread(target=self.process_user_input, args=(text, asr_timestamp, speech_len), daemon=True).start()
            time.sleep(0.1)  # 优化到0.1秒，提高响应速度

    # 其余逻辑和原 MainNodeWithGUI 完全一致
    def transition_state(self, new_state, reason=""):
        old_state = self.conversation_state
        self.conversation_state = new_state
        if self.DEBUG_MODE:
            print(f"🔄 状态转换: {old_state} → {new_state} | {reason}")
    
    # 已废弃：不再使用相似度计算（每轮都发新图片）
    # def compare_image_similarity(self, img1, img2):
    #     """
    #     使用 SSIM (结构相似性) 算法对比两张图像
    #     SSIM 能更好地检测结构变化（如人离开/进入画面）
    #     返回值: 0-1之间的相似度分数（1表示完全相同）
    #     """
    #     import cv2
    #     from skimage.metrics import structural_similarity as ssim
    #     
    #     # 降采样到 256x256 以平衡速度和精度
    #     # SSIM 需要一定分辨率才能准确检测结构变化
    #     small1 = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_AREA)
    #     small2 = cv2.resize(img2, (256, 256), interpolation=cv2.INTER_AREA)
    #     
    #     # 转换为灰度图（SSIM 在灰度图上计算更快）
    #     gray1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
    #     gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
    #     
    #     # 计算 SSIM
    #     # data_range 设为 255（8位图像的最大值）
    #     similarity = ssim(gray1, gray2, data_range=255)
    #     
    #     return similarity

    def handle_non_object_intent(self, intent, emotion, intensity):
        try:
            # 对于chat意图，如果没有情绪动作，就不停止追踪
            if intent == "chat" and not (emotion and intensity):
                return False
            
            self.transition_state("SPEAKING", f"处理{intent}意图")
            self._cancel_resume_tracking_timer()
            # 对照组：保持 human tracking，不进入 object 模式
            if not getattr(self, "TEXT_ONLY_MODE", False):
                self.look_controller.stop_head_tracking()
            # ===== 已禁用：表情功能 =====
            # if intent == "yes":
            #     self.emotion_module.call_routine(2093)
            # elif intent == "no":
            #     self.emotion_module.call_routine(2080)
            # elif intent == "bow":
            #     self.emotion_module.call_routine(2052)
            # elif intent == "chat" and emotion and intensity:
            #     success = self.emotion_module.execute_emotion(emotion, intensity)
            # 不在这里恢复追踪，由物体延时逻辑控制
            return True
        except Exception as e:
            self.look_controller.start_head_tracking()
            print("⚠️ 表情执行出错")
            return False

    def process_user_input(self, text, start_timestamp, user_speech_duration=None):
        try:
            # ====== 阶段A开始: 预处理 ======
            stage_A_start = time.time()
            
            image_start = time.time()
            self.transition_state("LISTENING", f"开始处理用户输入: {text[:20]}...")
            
            if not self.data_ready:
                print("❌ 传感器数据未就绪")
                self.transition_state("IDLE", "传感器未就绪，返回空闲状态")
                return
                
            if getattr(self, "TEXT_ONLY_MODE", False):
                print("🔄 正在处理...（盲人对照组：不发送图像）")
                image_to_send = None
                include_image = False
            else:
                print("🔄 正在处理...")

                # 不再使用相似度计算，每轮对话都发送新图片
                current_image = self.image_saver.latest_image
                if current_image is None:
                    print("❌ 无法获取图像数据")
                    return
                
                # 每轮都重新编码并发送新图片
                image_to_send = self.image_saver.get_latest_image_base64()
                include_image = True
                print(f"📸 发送新图片（每轮对话都使用最新图像）")
            
            # ====== 阶段A结束 ======
            stage_A_end = time.time()
            stage_A_preprocessing = stage_A_end - stage_A_start
            
            # ====== 阶段B开始: VLM推理 ======
            stage_B_start = time.time()
                    
            result = self.doubao_api.process_user_input(
                text,
                image_to_send,
                debug=self.VLM_VERBOSE,
                include_image=include_image,
                session_logger=self.session_logger,
            )
            
            # ====== 阶段B结束 ======
            stage_B_end = time.time()
            stage_B_vlm_inference = stage_B_end - stage_B_start
            
            vlm_done = time.time()
            print(f"⏱️  [AI理解完成] 耗时: {stage_B_vlm_inference:.2f}秒")
            
            # ====== speak.py特殊: 无3D定位阶段,直接进入动作调度 ======
            # ====== 阶段C开始: 动作调度 (从AI理解完成到机器人说话) ======
            stage_C_start = stage_B_end

            # ---------
            # 100% 记录：VLM 原始输出 + 解析摘要
            # ---------
            try:
                raw = result.get('raw_response')
                if isinstance(raw, str) and raw.strip():
                    raw_path = self.session_logger.save_vlm_raw(raw, tag="doubao_raw")
                    # 不刷屏：只把路径写入结构化日志
                    self.session_logger.log_terminal_output(f"[VLM] raw_response saved: {raw_path}")
            except Exception as e:
                self.session_logger.log_terminal_output(f"[VLM] raw_response save failed: {e}")
            
            intent = result.get('intent')
            emotion = result.get('emotion')
            intensity = result.get('intensity')

            # -------------------------------
            # 对照组：硬约束（找物=不允许）
            # -------------------------------
            if getattr(self, "TEXT_ONLY_MODE", False):
                # 1) 永远不接受 objects（即使模型返回了 JSON，也当作无效，避免出现“坐标/框”）
                if isinstance(result.get("objects"), list) and result["objects"]:
                    result["objects"] = []
                # 2) 禁止 gaze_object：降级为 chat
                if intent == "gaze_object":
                    intent = "chat"
                    result["intent"] = "chat"
                    # 强制把回复改成“不具备视觉 + 给通用建议”的风格
                    result["speech_response"] = (
                        "I can’t see the room right now. Where did you last see the umbrella? "
                        "You can check common spots like near the door, on a chair, by the entryway, "
                        "or where you usually leave rainy-day items."
                    )

            try:
                obj_names = None
                if isinstance(result.get('objects'), list):
                    obj_names = [o.get('name') for o in result.get('objects') if isinstance(o, dict) and o.get('name')]
                self.session_logger.log_terminal_output(
                    f"[VLM] parsed intent={intent} emotion={emotion} intensity={intensity} objects={obj_names}"
                )
            except Exception:
                pass
            if intent in ["yes", "no", "bow", "chat"]:
                # 对照组：不做任何 objects/注视/JA
                emotion_completed = self.handle_non_object_intent(intent, emotion, intensity)
                result["emotion_completed"] = emotion_completed
            
            if result:
                # ====== 阶段C结束 (在打印机器人响应前) ======
                stage_C_end = time.time()
                stage_C_action_dispatch = stage_C_end - stage_C_start
                
                robot_response = result['speech_response']
                print(f"\n[机器人] 🤖 {robot_response}\n")
                
                # 记录对话到日志
                detected_objects_for_log = None
                
                # 计算总时间（到 TTS 发起）
                total_turn_time = time.time() - start_timestamp if start_timestamp else 0
                
                self.session_logger.log_conversation_turn(
                    user_text=text,
                    robot_response=robot_response,
                    intent=result.get('intent'),
                    emotion=result.get('emotion'),
                    intensity=result.get('intensity'),
                    detected_objects=detected_objects_for_log,
                    user_speech_duration=user_speech_duration,
                    total_turn_time=total_turn_time,  # 到 TTS 发起的总时间
                    human_followed=False,  # 盲人模式没有 JA
                    robot_gaze=False,  # 盲人模式不看物体
                    stage_A_preprocessing=stage_A_preprocessing,
                    stage_B_vlm_inference=stage_B_vlm_inference,
                    stage_C_action_dispatch=stage_C_action_dispatch,
                    stage_C_grounding=None  # speak.py没有grounding阶段, 兼容接口
                )
                # 保存对话索引，用于后续更新TTS时长
                self._current_conversation_index = self.session_logger.conversation_index
                # ✅ 为 tts_status_callback 保存一个独立副本,避免被其他逻辑清空后无法更新
                self._tts_conversation_index = self.session_logger.conversation_index
                
                # 记录对话轮数（线程安全） -- 保持在记录会话时增加
                try:
                    with self._dialogue_rounds_lock:
                        self.dialogue_rounds += 1
                except Exception:
                    self.dialogue_rounds += 1
                
                intent = result.get('intent')
                has_emotion_action = result.get('has_emotion_action', False)
                emotion_completed = result.get('emotion_completed', False)
                
                # 提前计算估算时长（必须在speak之前，因为tts_status会立即触发）
                char_count = len(robot_response)
                self._estimated_tts_duration = char_count / 15.0 + 1.0  # 增加1秒缓冲避免提前解除抑制
                
                # 触发 TTS：在 tts_status_callback 中基于 TTSStatus 事件计算精确时长
                self._pending_tts_start = time.time()
                # TTS开始时设置注视状态
                if not getattr(self, "TEXT_ONLY_MODE", False):
                    self._setup_tts_gaze_control(intent)
                self.env.speak(robot_response)
                
                # 立即写入估算时长到CSV，不等tts_status回调
                tts_index = self.session_logger.conversation_index
                if tts_index is not None and self._estimated_tts_duration is not None:
                    self.session_logger.update_tts_duration(tts_index, self._estimated_tts_duration)

                # 对照组：强制保持/恢复 human tracking，避免被其它逻辑切到 object
                if getattr(self, "TEXT_ONLY_MODE", False):
                    try:
                        self.look_controller.start_head_tracking()
                    except Exception:
                        pass

                # 记录总耗时
                print(f"⏱️  [完成] 总耗时(到TTS发起): {total_turn_time:.2f}秒")
                self.session_logger.log_terminal_output(f"处理完成，总耗时(到TTS发起): {total_turn_time:.2f}秒")
            else:
                print("❌ 处理请求时出现问题")
                self.session_logger.log_terminal_output("❌ 处理请求时出现问题")
            if result and result.get('intent') not in ["gaze_object", "yes", "no", "bow", "chat"]:
                self.transition_state("IDLE", "用户输入处理完成，回到空闲状态")
        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            print(f"❌ {error_msg}")
            self.session_logger.log_terminal_output(error_msg)
            self.transition_state("IDLE", f"处理异常，返回空闲状态")

def main():
    rclpy.init()
    main_node = MainNodeWithGUI()
    executor = MultiThreadedExecutor()
    executor.add_node(main_node)
    # executor.add_node(main_node.asr)  # 讯飞ASR不是ROS节点，无需添加到executor
    executor.add_node(main_node.image_saver)
    executor.add_node(main_node.env.tts_node)
    executor.add_node(main_node.env.motor_node)
    # ===== 已禁用：表情模块 =====
    # executor.add_node(main_node.emotion_module)
    
    def run_executor():
        try:
            executor.spin()
        except Exception as e:
            print(f"执行器错误: {e}")
    executor_thread = threading.Thread(target=run_executor, daemon=True)
    executor_thread.start()
    
    # 添加强制spin线程确保rosout回调被及时处理
    spin_stop_flag = threading.Event()
    def force_spin_main_node():
        while not spin_stop_flag.is_set():
            try:
                rclpy.spin_once(main_node, timeout_sec=0.01)
            except Exception:
                break  # 节点已销毁，退出循环
            time.sleep(0.01)  # 100Hz
    spin_thread = threading.Thread(target=force_spin_main_node, daemon=True)
    spin_thread.start()
    
    print("🤖 机器人系统已启动，等待语音输入...")
    print("💡 提示：请直接对机器人说话，系统会自动识别并处理")
    print(f"📁 会话日志保存至: {main_node.session_logger.session_dir}")
    
    try:
        # 保持主线程运行，等待用户中断
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 正在关闭系统...")
        
        # 打印最终统计
        metrics = main_node.ja_evaluator.get_current_metrics()
        print("\n" + "="*60)
        print("📊 Joint Attention 最终统计:")
        print(f"   总交互次数: {metrics.get('total_episodes', 0)}")
        print(f"   成功跟随次数: {metrics.get('successful_episodes', 0)}")
        print(f"   视线跟随率: {metrics.get('gaze_following_rate', 0.0):.1%}")
        print("="*60)

        # 打印并记录对话轮数（终端会被 full_terminal_output 捕获）
        try:
            rounds = main_node.dialogue_rounds
        except Exception:
            rounds = 0
        print(f"   对话轮数 (user↔robot 轮): {rounds}")
        
        # 停止spin线程
        spin_stop_flag.set()
        time.sleep(0.1)  # 等待线程退出
        
    finally:
        # 确保无论如何都保存完整的 metadata
        try:
            rounds = main_node.dialogue_rounds
            # 把轮数写入结构化 terminal log
            main_node.session_logger.log_terminal_output(f"Total dialogue rounds: {rounds}")
            # 并存入 metadata，便于会话总结
            main_node.session_logger.metadata["dialogue_rounds"] = rounds
        except Exception as e:
            print(f"⚠️ 记录对话轮数失败: {e}")

        # 关闭前：检查是否有未记录的TTS时长
        try:
            tts_index = getattr(main_node, "_tts_conversation_index", None)
            if tts_index is not None and hasattr(main_node, '_estimated_tts_duration'):
                if main_node._estimated_tts_duration is not None:
                    # 有未记录的TTS时长，补充写入
                    main_node.session_logger.update_tts_duration(tts_index, main_node._estimated_tts_duration)
                    print(f"📝 关闭前补充写入 TTS 时长: turn {tts_index}")
        except Exception as e:
            print(f"⚠️ 关闭时写入TTS时长失败: {e}")

        # 关闭会话日志（写入 end_time 和 duration_seconds）
        try:
            main_node.session_logger.close_session()
        except Exception as e:
            print(f"⚠️ 关闭会话日志失败: {e}")
        
        # 清理资源
        main_node.destroy_node()
        main_node.asr.destroy_node()  # 销毁 ASR 节点
        main_node.image_saver.destroy_node()
        # ===== 已禁用：表情模块 =====
        # main_node.emotion_module.destroy_node()
        main_node.env.shutdown()
        executor.shutdown()
        # 避免 Ctrl+C/外部退出时重复 shutdown 导致：
        # rclpy._rclpy_pybind11.RCLError: rcl_shutdown already called
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            # 这里主要是容错：当上下文已经 shutdown 时，再调用会抛 RCLError。
            pass

if __name__ == "__main__":
    main()
