import threading
import os
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from xunfei_asr_adapter import XunfeiASRAdapter as ASRModule
from image_save_module import ImageSaveModule
from doubao_api_module import DoubaoAPIModule
from robot_env import RobotEnv
from simple_emotion_test import SimpleEmotionTest
from conversation_logger import ConversationSessionLogger
from strawberry_ros_msgs.msg import Skeletons
from types import SimpleNamespace
import time
import random
from typing import Optional


class MainNodeWithGUI(Node):
    def __init__(self):
        super().__init__('main_node_gui')
        self.asr = ASRModule(mic_device="plughw:2,0")
        self.image_saver = ImageSaveModule()
        self.doubao_api = DoubaoAPIModule()
        self.env = RobotEnv()
        from look_controller import LookController
        self.look_controller = LookController(self.env.motor_node, self)
        self.emotion_module = SimpleEmotionTest()
        self.conversation_state = "IDLE"
        # 默认关闭刷屏调试；需要排障时再临时打开。
        self.DEBUG_MODE = False
        # 仅控制 VLM 解析/JSON 等详细输出（像素坐标、置信度、原始 JSON）。
        # 这个开关不会影响其它模块的调试打印。
        self.VLM_VERBOSE = True
        self.data_ready = False
        self.resume_tracking_timer = None
        self.prev_text = ""
        self.tts_gaze_mode = None  # 记录TTS期间的注视模式: "object", "human", None
        self.tts_gaze_target = None  # 记录TTS期间的注视目标
        
        # 初始化日志系统
        self.session_logger = ConversationSessionLogger(
            base_log_dir="/media/crystal/KINGSTON/qwe/2/haru_conversation_logs"
        )
        # 捕获 stdout/stderr，保证"完整终端输出"可找回
        self.session_logger.enable_full_terminal_capture()

        # 对话轮数统计（1轮 = 用户一句 + 机器人一句）
        self.dialogue_rounds = 0
        self._dialogue_rounds_lock = threading.Lock()

        # 用户需求：不再落盘 ROS_LOG_DIR 到会话目录（避免生成 ros_logs 文件夹）。
        # 若仍需要把 ROS2 logger 输出并入 full_terminal_output.log，建议在运行环境层面配置。
        
        # -------------------------------
        # JA 检测已禁用（改为人工记录）
        # -------------------------------
        # 设置为 None 避免 AttributeError
        self.ja_evaluator = None
        
        # 初始化 Joint Attention 评估器(简化为30度固定阈值)
        # self.ja_evaluator = JointAttentionEvaluator(
        #     # 使用 60° 作为平衡点：之前的日志显示用户真正看向时能达到 60.7° 左右
        #     # 这个值比 55° 宽松一点，但比 70° 严格，避免误判
        #     angle_threshold_deg=60.0,
        #     reaction_window=3.0,
        #     robot_motion_delay=0.8,
        #     min_shared_duration=0.5,
        #     logger=self.session_logger
        # )

        # # JA 临时诊断配置（检测为何无法跟随）
        # self.ja_evaluator._debug_coord_diag = True
        # self.ja_evaluator._debug_person_frame = True
        # self.ja_evaluator._quiet = False
        # self.ja_evaluator._person_frame_enabled = False
        # self.ja_evaluator._enable_angle_smoothing = False
        
        # 骨骼订阅也注释掉（不再需要）
        # self.skeleton_sub_for_ja = self.create_subscription(
        #     Skeletons,
        #     '/strawberry/azure/skeletons',
        #     self.skeleton_callback_for_ja,
        #     10
        # )
        
        # 图像缓存机制
        # 不再需要图像缓存（每轮都发新图片）
        # self.cached_image = None
        # self.cached_image_base64 = None
        self.last_image_upload_time = None  # 上次上传图像的时间
        
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

        # TTS 播放时长追踪
        self._tts_play_start: Optional[float] = None
        self._pending_tts_start: Optional[float] = None
        self._last_tts_duration: Optional[float] = None
        self._current_conversation_index: Optional[int] = None  # 当前对话索引，用于更新TTS时长
        
        threading.Thread(target=self.voice_input_loop, daemon=True).start()
        if self.DEBUG_MODE:
            print("🤖 机器人对话系统已启动")
            print("🎤 语音监听已启动，请说话...")

    def init_sensor_check(self):
        print("🔍 开始检查传感器状态...")
        for i in range(15):
            status = self.image_saver.check_sensor_status()
            print(f"🔍 传感器状态检查 {i+1}/15: 图像={status['image_ready']}, 深度={status['depth_ready']}, 相机信息={status['camera_info_ready']}")
            if status["image_ready"] and status["depth_ready"] and status["camera_info_ready"]:
                self.data_ready = True
                print("✅ 传感器就绪，语音输入处理已启用")
                return
            time.sleep(1)
        print("⚠️ 传感器检测超时（15秒），但仍启用语音输入处理")
        print("❌ 传感器检测超时，请检查：")
        print("   1. Azure Kinect 设备连接")
        print("   2. ROS 2 节点是否正常启动")
        print("   3. 网络连接")
        # 即使传感器检测失败，也设置 data_ready 为 True，让语音输入能工作
        self.data_ready = True
        print("🔧 已强制启用语音输入处理（传感器数据可能不可用）")

    # JA 检测已禁用 - skeleton_callback_for_ja 不再使用
    # def skeleton_callback_for_ja(self, msg: Skeletons):
    #     """接收骨骼数据,用于 Joint Attention 评估"""
    #     timestamp = time.time()
    #     
    #     # 检测 JA episode finalize（robot_gaze_active 从 True 变为 False）
    #     if not hasattr(self, "_prev_robot_gaze_active"):
    #         self._prev_robot_gaze_active = False
    #     
    #     current_active = getattr(self.ja_evaluator, "robot_gaze_active", False)
    #     
    #     if self._prev_robot_gaze_active and not current_active:
    #         # JA episode 刚刚结束，立即更新CSV的 human_followed
    #         print("🔔 [JA] Episode finalized, 更新CSV记录...", flush=True)
    #         try:
    #             success = bool(getattr(self.ja_evaluator, "last_episode_success", False))
    #             if getattr(self, "_current_conversation_index", None) is not None:
    #                 self.session_logger.update_human_followed(
    #                     self._current_conversation_index, success
    #                 )
    #                 print(f"✅ [JA] 更新 human_followed={success}", flush=True)
    #                 # 清空 index，避免重复更新
    #                 self._current_conversation_index = None
    #         except Exception as e:
    #             print(f"⚠️ [JA] 更新CSV失败: {e}", flush=True)
    #     
    #     self._prev_robot_gaze_active = current_active
    #     
    #     pseudo_people = []
    #     for idx, skeleton in enumerate(msg.skeletons):
    #         # 使用 tracking_id 作为人员ID,没有则使用索引
    #         person_id = getattr(skeleton, 'tracking_id', idx)
    #         pseudo_people.append(SimpleNamespace(id=person_id, skeleton=skeleton))
    # 
    #     # 默认不打印逐帧骨骼调试信息（太刷屏）。
    #     # 需要时可在运行过程中临时设置：self.JA_DEBUG_SKELETON = True
    #     if getattr(self, "JA_DEBUG_SKELETON", False) and self.DEBUG_MODE:
    #         if not hasattr(self, "_ja_debug_parts_logged"):
    #             first = msg.skeletons[0] if msg.skeletons else None
    #             if first:
    #                 part_names = [bp.name for bp in first.body_parts]
    #                 print(f"🧩 [JA] 首次骨骼部位: {part_names}")
    #             self._ja_debug_parts_logged = True
    # 
    #         # 低频节流：每 0.5s 最多打印一次
    #         if not hasattr(self, "_ja_dbg_last_print"):
    #             self._ja_dbg_last_print = 0.0
    #         if timestamp - self._ja_dbg_last_print >= 0.5:
    #             self._ja_dbg_last_print = timestamp
    #             first = msg.skeletons[0] if msg.skeletons else None
    #             coords = None
    #             if first:
    #                 for part in first.body_parts:
    #                     if part.name.lower() == "nose":
    #                         coords = (
    #                             part.pose.position.x,
    #                             part.pose.position.y,
    #                             part.pose.position.z,
    #                         )
    #                         break
    #             print(
    #                 "🦴 [JA] skeleton_callback: count={}, active={}, nose={}".format(
    #                     len(msg.skeletons), self.ja_evaluator.robot_gaze_active, coords
    #                 )
    #             )
    #     
    #     self.ja_evaluator.on_people_update(timestamp, pseudo_people)

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

            # 原有：TTS 停止播放，恢复人体追踪（仅在原来的条件下）
            if self.tts_gaze_mode == "object" and self.tts_gaze_target:
                # 如果之前在看物体，保持注视一段时间再恢复人体追踪
                self._cancel_resume_tracking_timer()

                # JA 检测已禁用 - 直接恢复追踪，不再等待 grace period
                # grace = getattr(self.ja_evaluator, "post_gaze_grace", 0.0)
                grace = 0.0 if self.ja_evaluator is None else getattr(self.ja_evaluator, "post_gaze_grace", 0.0)
                
                if grace > 0:
                    if self.DEBUG_MODE:
                        print(
                            f"⏳ TTS结束，继续看向物体 {grace:.1f}s 后再恢复人体追踪"
                        )
                else:
                    if self.DEBUG_MODE:
                        print("🔄 TTS结束，准备恢复人体追踪")

                # JA 检测已禁用 - 注释掉相关调用
                # # 先通知 JA 评估器，开始进入补偿窗口（仅当 episode 已启动）
                # if getattr(self.ja_evaluator, "robot_gaze_active", False):
                #     self.ja_evaluator.on_robot_gaze_end(time.time())
                # else:
                #     # 常见原因：chat 模式仅 look_at_object 但未启动 on_robot_gaze_start。
                #     # 这里静默跳过，避免误导性告警刷屏。
                #     pass

                # def _resume_tracking_after_grace():
                #     try:
                #         now2 = time.time()
                #         self.ja_evaluator.on_grace_period_timeout(now2)
                #         if self.DEBUG_MODE:
                #             print("👀 Grace 结束，恢复人体追踪")
                #         self.look_controller.start_head_tracking()
                #     finally:
                #         self.resume_tracking_timer = None

                # if grace > 0:
                #     self.resume_tracking_timer = threading.Timer(
                #         grace, _resume_tracking_after_grace
                #     )
                #     self.resume_tracking_timer.start()
                # else:
                #     _resume_tracking_after_grace()

                # 简化：直接恢复人体追踪（不再等待 grace period）
                if self.DEBUG_MODE:
                    print("👀 恢复人体追踪")
                self.look_controller.start_head_tracking()

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
                # 添加调试信息
                if not hasattr(self, '_data_ready_logged'):
                    print("⏳ 等待传感器初始化完成...")
                    self._data_ready_logged = True
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
            self.look_controller.stop_head_tracking()
            
            # 机器人表情功能已禁用
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
                
            print("🔄 正在准备图像并请求 AI 理解...")
            
            # ========== 智能图像缓存机制 ==========
            import cv2
            import base64
            import numpy as np
            
            # ✅ 关键修复：循环重置缓存并等待新帧
            max_retries = 5
            retry_delay = 0.3  # 300ms，给ROS回调更多时间
            current_image_b64 = None
            current_image = None
            
            for retry in range(max_retries):
                # 每次重试都重置缓存
                self.image_saver.reset_capture()
                time.sleep(retry_delay)  # 等待ROS回调接收新帧
                
                current_image_b64 = self.image_saver.get_latest_image_base64()
                if current_image_b64:
                    try:
                        img_bytes = base64.b64decode(current_image_b64)
                        img_np = np.frombuffer(img_bytes, np.uint8)
                        current_image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                        
                        # 如果成功解码且不是第一次，检查是否有变化
                        if current_image is not None:
                            # 首次或成功获取，跳出
                            if retry == 0 or self.cached_image is None:
                                break
                            
                            # 快速检查与缓存图像是否相同（避免卡帧）
                            gray_new = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
                            gray_cached = cv2.cvtColor(self.cached_image, cv2.COLOR_BGR2GRAY)
                            tiny_new = cv2.resize(gray_new, (16, 16), interpolation=cv2.INTER_AREA)
                            tiny_cached = cv2.resize(gray_cached, (16, 16), interpolation=cv2.INTER_AREA)
                            fp_new = int(tiny_new.mean() * 1000)
                            fp_cached = int(tiny_cached.mean() * 1000)
                            
                            if fp_new != fp_cached:
                                # 画面更新了
                                break
                        else:
                            if retry < max_retries - 1:
                                pass
                    except Exception:
                        current_image = None
            
            # 容错回退
            if current_image is None:
                current_image = self.image_saver.latest_image
            
            # 不再使用相似度计算，每轮对话都发送新图片
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
            
            # ====== 阶段C开始: 动作调度 (从AI理解完成到机器人说话) ======
            stage_C_start = stage_B_end
            
            vlm_done = time.time()
            print(f"⏱️  [AI理解完成] 耗时: {stage_B_vlm_inference:.2f}秒")

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
            
            # ========== 保存 VLM 原始 bbox（在 update_object_positions 修改之前）==========
            original_bboxes = {}
            if self.doubao_api.last_detected_objects:
                for obj in self.doubao_api.last_detected_objects:
                    obj_name = obj.get('name')
                    obj_bbox = obj.get('bboxes')
                    if obj_name and obj_bbox:
                        original_bboxes[obj_name] = list(obj_bbox)  # 深拷贝
            
            # 用于记录机器人实际看向的物体（会在下面的逻辑中设置）
            gazed_object_name = None
            gazed_object_coord = None
            
            # ========== 每轮对话都保存图像（无论是否检测到物体）==========
            def save_images_async():
                if result.get("objects"):
                    # 有物体时保存标注图像
                    paths = self.image_saver.save_annotated_image(result['objects'])
                else:
                    # 无物体时保存原始图像
                    paths = self.image_saver.save_annotated_image([])
                
                overlay_path = self.image_saver.save_overlay_image()
                
                self.session_logger.save_images(
                    marked_image_path=paths.get('marked'),
                    rgb_image_path=paths.get('rgb'),
                    depth_image_path=paths.get('depth'),
                    overlay_image_path=overlay_path,
                )
            
            # 启动异步保存（不阻塞主流程）
            threading.Thread(target=save_images_async, daemon=True).start()
            
            # ========== 物体检测与注视控制（简化版，不依赖intent）==========
            if result.get("objects"):
                print(f"📦 检测到物体: {[obj['name'] for obj in result['objects']]}")
                
                # 更新物体坐标（2D bbox -> 3D世界坐标）
                self.doubao_api.update_object_positions(self.image_saver, debug=self.DEBUG_MODE)
                
                # 打印物体坐标字典
                print(f"🔍 物体坐标字典: {self.doubao_api.object_coordinates}")
                
                # 记录检测到的物体到日志（使用原始 VLM bbox）
                detected_objects_with_coords = []
                for obj in self.doubao_api.last_detected_objects:
                    obj_name = obj['name']
                    obj_coord = self.doubao_api.object_coordinates.get(obj_name)
                    detected_objects_with_coords.append({
                        'name': obj_name,
                        'bboxes': original_bboxes.get(obj_name, []),  # 使用 VLM 原始 bbox
                        'coordinates': obj_coord
                    })
                self.session_logger.log_detected_objects(detected_objects_with_coords, time.time())
                
                # 看向第一个检测到的物体
                target_object_name = list(self.doubao_api.object_coordinates.keys())[0] if self.doubao_api.object_coordinates else None
                if target_object_name:
                    target_coord = self.doubao_api.object_coordinates[target_object_name]
                    if target_coord:
                        print(f"👀 看向物体: {target_object_name}")
                        
                        self._cancel_resume_tracking_timer()
                        self.look_controller.stop_head_tracking()
                        self.look_controller.look_at_object(target_coord)
                        
                        # 记录真正看向的物体（用于CSV记录）
                        gazed_object_name = target_object_name
                        gazed_object_coord = target_coord
                        
                        # 说话时保持注视，说完后由TTS回调恢复追踪
                        print(f"💬 将在TTS结束后自动恢复人体追踪")
                    else:
                        print(f"⚠️ 物体 {target_object_name} 没有3D坐标信息")
                else:
                    print(f"⚠️ 未找到物体坐标信息")
            else:
                print(f"ℹ️ 本轮对话未检测到物体")
            
            # ========== 兼容旧代码：保留 intent/emotion 变量（避免后续代码报错）==========
            intent = result.get('intent')  # 新版本中为 None
            emotion = result.get('emotion')  # 新版本中为 None
            intensity = result.get('intensity')  # 新版本中为 None
            
            if result:
                # ====== 阶段C结束 (在打印机器人响应前) ======
                stage_C_end = time.time()
                stage_C_action_dispatch = stage_C_end - stage_C_start
                
                robot_response = result['speech_response']
                print(f"\n[机器人] 🤖 {robot_response}\n")
                
                # 记录对话到日志
                detected_objects_for_log = None
                
                # 优先记录机器人实际看向的物体（如果有坐标数据）
                if gazed_object_name:
                    detected_objects_for_log = [{
                        'name': gazed_object_name,
                        'bboxes': original_bboxes.get(gazed_object_name, []),  # 使用 VLM 原始 bbox
                        'confidence': 0.9,  # 默认置信度
                        'coordinates': gazed_object_coord
                    }]
                # 如果没在看（如深度缺失导致无法look_at），但 VLM 确实检出了物体，则记录第一个物体的 2D 信息
                elif result.get("objects"):
                    first_obj = result["objects"][0]
                    obj_name = first_obj.get("name")
                    detected_objects_for_log = [{
                        'name': obj_name,
                        'bboxes': original_bboxes.get(obj_name, []),
                        'confidence': first_obj.get("confidence", 0.9),
                        'coordinates': None
                    }]
                
                # 判断是否正在追踪人体和是否正在注视物体
                # JA 检测已禁用 - human_followed 默认为 False（需要人工记录）
                # human_followed: 使用上一轮JA episode的最终结果（True=人类跟随了，False=没跟随）
                # 如果还没有结果（首轮对话），默认为False
                # 关键口径：reason=window_expired success=✅ 必须记录 True
                # human_followed = bool(getattr(self.ja_evaluator, 'last_episode_success', False))
                human_followed = False  # 改为人工记录，默认 False
                
                # robot_gaze: 如果机器人正在看向物体（有物体坐标）
                is_gazing = bool(self.doubao_api.object_coordinates)
                
                self.session_logger.log_conversation_turn(
                    user_text=text,
                    robot_response=robot_response,
                    intent=result.get('intent'),
                    emotion=result.get('emotion'),
                    intensity=result.get('intensity'),
                    detected_objects=detected_objects_for_log,
                    image_size=str(result.get('image_size', '')),
                    user_speech_duration=user_speech_duration,
                    total_turn_time=0,  # 临时占位,稍后在TTS发起后更新
                    human_followed=human_followed,
                    robot_gaze=is_gazing,
                    stage_A_preprocessing=stage_A_preprocessing,
                    stage_B_vlm_inference=stage_B_vlm_inference,
                    stage_C_action_dispatch=stage_C_action_dispatch
                )
                # 保存对话索引，用于后续更新TTS时长和total_turn_time
                self._current_conversation_index = self.session_logger.conversation_index
                
                # 记录对话轮数（线程安全） -- 保持在记录会话时增加
                try:
                    with self._dialogue_rounds_lock:
                        self.dialogue_rounds += 1
                except Exception:
                    self.dialogue_rounds += 1
                
                intent = result.get('intent')
                has_emotion_action = result.get('has_emotion_action', False)
                emotion_completed = result.get('emotion_completed', False)
                
                # 触发 TTS：不在此处测量播放时长（speak 可能是异步），
                # 改为在 tts_status_callback 中基于 TTSStatus 事件计算精确时长。
                self._pending_tts_start = time.time()
                
                # 估算TTS时长（用于ASR抑制）
                char_count = len(robot_response)
                self._estimated_tts_duration = char_count / 15.0 + 1.0  # 增加1秒缓冲
                
                # TTS开始时设置注视状态
                self._setup_tts_gaze_control(intent)
                self.env.speak(robot_response)
                
                # 立即写入估算时长到CSV，不等tts_status回调
                tts_index = self._current_conversation_index
                if tts_index is not None and self._estimated_tts_duration is not None:
                    self.session_logger.update_tts_duration(tts_index, self._estimated_tts_duration)

                # 计算并更新 total_turn_time（从用户开始说话到TTS发起）
                total_turn_time = time.time() - start_timestamp if start_timestamp else 0
                if self._current_conversation_index is not None:
                    self.session_logger.update_total_turn_time(self._current_conversation_index, total_turn_time)

                # 记录总耗时
                print(f"⏱️  [完成] 总耗时(到TTS发起): {total_turn_time:.2f}秒")
                self.session_logger.log_terminal_output(f"处理完成，总耗时(到TTS发起): {total_turn_time:.2f}秒")
            else:
                print("❌ 处理请求时出现问题")
                self.session_logger.log_terminal_output("❌ 处理请求时出现问题")
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
    executor.add_node(main_node.emotion_module)
    def run_executor():
        try:
            executor.spin()
        except Exception as e:
            print(f"执行器错误: {e}")
    executor_thread = threading.Thread(target=run_executor, daemon=True)
    executor_thread.start()
    
    print("🤖 机器人系统已启动，等待语音输入...")
    print("💡 提示：请直接对机器人说话，系统会自动识别并处理")
    print(f"📁 会话日志保存至: {main_node.session_logger.session_dir}")
    
    try:
        # 保持主线程运行，等待用户中断
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 正在关闭系统...")
        
        # 打印最终统计（JA 检测已禁用，跳过统计）
        # metrics = main_node.ja_evaluator.get_current_metrics()
        # print("\n" + "="*60)
        # print("📊 Joint Attention 最终统计:")
        # print(f"   总交互次数: {metrics.get('total_episodes', 0)}")
        # print(f"   成功跟随次数: {metrics.get('successful_episodes', 0)}")
        # print(f"   视线跟随率: {metrics.get('gaze_following_rate', 0.0):.1%}")
        # print("="*60)

        # 打印并记录对话轮数（终端会被 full_terminal_output 捕获）
        try:
            rounds = main_node.dialogue_rounds
        except Exception:
            rounds = 0
        print(f"   对话轮数 (user↔robot 轮): {rounds}")
        
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
        main_node.emotion_module.destroy_node()
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
