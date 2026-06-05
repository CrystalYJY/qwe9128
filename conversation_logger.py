"""
对话会话日志系统
每次对话创建独立文件夹，保存所有相关数据
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
import shutil
import sys
import threading
import csv


class ConversationSessionLogger:
    def __init__(self, base_log_dir="/media/crystal/KINGSTON/qwe/haru_conversation_logs"):
        self.base_log_dir = Path(base_log_dir)
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建本次会话文件夹
        session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_log_dir / session_time
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # 子目录
        self.images_dir = self.session_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        # VLM 原始输出（避免“终端没打印就丢”）
        self.vlm_dir = self.session_dir / "vlm_raw"
        self.vlm_dir.mkdir(exist_ok=True)
        
        # 文件路径
        self.metadata_file = self.session_dir / "metadata.json"
        self.conversation_file = self.session_dir / "conversation.txt"
        self.terminal_log_file = self.session_dir / "terminal_output.log"
        # 完整终端输出（捕获 stdout/stderr，避免只记录部分事件）
        self.full_terminal_log_file = self.session_dir / "full_terminal_output.log"

        # 用户需求：不再保存 conversation.txt / terminal_output.log
        self.enable_conversation_txt = False
        self.enable_terminal_output_log = False

        self.objects_file = self.session_dir / "detected_objects.json"
        self.ja_metrics_file = self.session_dir / "joint_attention.csv"
        
        # 初始化元数据
        self.metadata = {
            "session_id": session_time,
            "start_time": time.time(),
            "start_time_str": datetime.now().isoformat(),
            "conversation_count": 0,
            "image_count": 0,
            "object_detection_count": 0
        }
        self._save_metadata()
        
        # 对话序号
        self.conversation_index = 0
        self.image_index = 0
        self.vlm_index = 0
        
        # 初始化日志文件（按开关决定是否创建）
        if self.enable_conversation_txt:
            with open(self.conversation_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Conversation Session: {session_time} ===\n\n")
        
        if self.enable_terminal_output_log:
            with open(self.terminal_log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Terminal Output Log: {session_time} ===\n\n")

        with open(self.full_terminal_log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Full Terminal Output (stdout/stderr): {session_time} ===\n\n")
        
        # 初始化 JA 指标 CSV
        with open(self.ja_metrics_file, 'w', encoding='utf-8') as f:
            f.write("timestamp,event_type,person_id,gaze_angle_deg,distance_m,object_id,object_name,robot_gaze_active,human_looking_object,shared_gaze_duration,reaction_latency\n")
        
        print(f"📁 会话日志目录: {self.session_dir}")

        self._stream_tee_installed = False
        self._orig_stdout = None
        self._orig_stderr = None
        self._tee_lock = threading.Lock()

        # 初始化对话轮次的CSV日志文件
        self.csv_log_file = self.session_dir / "conversation_turns.csv"
        with open(self.csv_log_file, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "turn_id", "user_utterance", "robot_utterance", "intent", "emotion", "intensity",
                "total_turn_time (s)", "detected_object_name",
                "detection_confidence", "object_bbox", "image_size", "object_3d_position",
                "human_followed", "robot_gaze", "human_duration (s)", "robot_duration (s)",
                "stage_A_preprocessing (s)", "stage_B_vlm_inference (s)", "stage_C_action_dispatch (s)"
            ])

    def enable_full_terminal_capture(self):
        """捕获 stdout/stderr 到 full_terminal_output.log。

        说明
        - 这是“终端完整输出可找回”的核心：print、异常 traceback 等都会写入该文件。
        - 仍保留现有 terminal_output.log（结构化事件）；两者用途不同。
        - 若你还希望把 ROS2 的 logger（rclpy logging）也完全接入同一文件，
          需要额外配置 ROS_LOG_DIR/RCUTILS_LOGGING_*（这部分在运行环境层面更稳）。
        """
        if self._stream_tee_installed:
            return

        # 先确保文件存在（会话初始化已创建，这里防御）
        self.full_terminal_log_file.parent.mkdir(parents=True, exist_ok=True)
        self.full_terminal_log_file.touch(exist_ok=True)

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

        logger = self

        class _TeeTextIO:
            def __init__(self, stream_name: str, orig_stream):
                self._name = stream_name
                self._orig = orig_stream
                self._buf = ""

            def _now_prefix(self) -> str:
                # 统一毫秒级时间戳，便于回放时对齐
                return "[{}] ".format(datetime.now().strftime("%H:%M:%S.%f")[:-3])

            def write(self, s):
                # 先写回终端（尽量不影响交互）
                try:
                    self._orig.write(s)
                except Exception:
                    pass

                # 再写入 full log（跨线程尽量串行）
                try:
                    if not s:
                        return

                    # 逐行加时间戳：按换行拆分，给每一行前加 [HH:MM:SS.mmm]
                    self._buf += s
                    lines = self._buf.split("\n")
                    # 最后一个可能是不完整行，留到下次
                    self._buf = lines.pop() if lines else ""

                    if not lines:
                        return

                    with logger._tee_lock:
                        with open(logger.full_terminal_log_file, 'a', encoding='utf-8') as f:
                            for line in lines:
                                f.write(self._now_prefix() + line + "\n")
                except Exception:
                    # 不能在这里再 print，否则递归
                    pass

            def flush(self):
                try:
                    self._orig.flush()
                except Exception:
                    pass

                # flush 时也把缓冲区落盘（即使没有换行）
                try:
                    if self._buf:
                        with logger._tee_lock:
                            with open(logger.full_terminal_log_file, 'a', encoding='utf-8') as f:
                                f.write(self._now_prefix() + self._buf + "\n")
                        self._buf = ""
                except Exception:
                    pass

            # 兼容某些库对 isatty 的检测
            def isatty(self):
                try:
                    return bool(getattr(self._orig, "isatty")())
                except Exception:
                    return False

        sys.stdout = _TeeTextIO("stdout", self._orig_stdout)
        sys.stderr = _TeeTextIO("stderr", self._orig_stderr)
        self._stream_tee_installed = True

    def disable_full_terminal_capture(self):
        """恢复 stdout/stderr（通常不需要调用）。"""
        if not self._stream_tee_installed:
            return
        try:
            if self._orig_stdout is not None:
                sys.stdout = self._orig_stdout
            if self._orig_stderr is not None:
                sys.stderr = self._orig_stderr
        finally:
            self._stream_tee_installed = False
    
    def _save_metadata(self):
        """保存元数据"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def log_terminal_output(self, message: str):
        """记录终端输出"""
        if not getattr(self, "enable_terminal_output_log", True):
            return
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with open(self.terminal_log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")

    def save_vlm_raw(self, raw_text: str, tag: str = "vlm") -> str:
        """保存 VLM 原始文本到会话目录。

        目的：让“终端没打印的内容”也能 100% 找回。
        """
        self.vlm_index += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        safe_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (tag or "vlm"))
        path = self.vlm_dir / f"{self.vlm_index:03d}_{safe_tag}_{ts}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(raw_text or "")
        return str(path)
    
    def log_conversation_turn(self, user_text: str, robot_response: str, 
                             intent: str = None, emotion: str = None,
                             detected_objects: list = None,
                             user_speech_duration: float = None,
                             tts_duration: float = None,
                             **kwargs):
        """记录一轮对话并保存到CSV"""
        self.conversation_index += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if getattr(self, "enable_conversation_txt", True):
            with open(self.conversation_file, 'a', encoding='utf-8') as f:
                f.write(f"{'='*60}\n")
                f.write(f"[Turn {self.conversation_index}] {timestamp}\n")
                f.write(f"{'='*60}\n\n")
                
                # 用户输入
                f.write(f"👤 User: {user_text}\n")
                if user_speech_duration is not None:
                    f.write(f"   ⏱️ Speech Duration: {user_speech_duration:.2f}s\n")
                f.write("\n")
                
                # 机器人响应
                f.write(f"🤖 Robot: {robot_response}\n")
                if tts_duration is not None:
                    f.write(f"   ⏱️ TTS Duration: {tts_duration:.2f}s\n")
                f.write("\n")
                
                if intent or emotion:
                    f.write(f"📊 Analysis:\n")
                    f.write(f"   Intent: {intent or 'N/A'}\n")
                    f.write(f"   Emotion: {emotion or 'N/A'}\n\n")
                
                if detected_objects:
                    f.write(f"📦 Detected Objects:\n")
                    for obj in detected_objects:
                        f.write(f"   - {obj.get('name', 'Unknown')}: ")
                        f.write(f"bbox={obj.get('bboxes', 'N/A')}, ")
                        f.write(f"3D={obj.get('coordinates', 'N/A')}\n")
                    f.write("\n")
        
        # 写入CSV - 清理数据，将 <none> 或 None 转为空字符串
        def clean_value(val):
            """清理CSV值，将None、<none>、none转为空字符串"""
            if val is None or val == "<none>" or val == "none":
                return ""
            return val
        
        try:
            with open(self.csv_log_file, 'a', encoding='utf-8', newline='') as csvfile:
                writer = csv.writer(csvfile)
                row_data = [
                    self.conversation_index, 
                    user_text, 
                    robot_response, 
                    clean_value(intent), 
                    clean_value(emotion), 
                    clean_value(kwargs.get("intensity")),
                    kwargs.get("total_turn_time", ""),
                    detected_objects[0].get("name", "") if detected_objects else "",
                    detected_objects[0].get("confidence", "") if detected_objects else "",
                    str(detected_objects[0].get("bboxes", "")) if detected_objects else "",
                    kwargs.get("image_size", ""),
                    str(detected_objects[0].get("coordinates", "")) if detected_objects else "",
                    kwargs.get("human_followed", ""), 
                    kwargs.get("robot_gaze", ""), 
                    user_speech_duration if user_speech_duration else "", 
                    tts_duration if tts_duration else "",
                    kwargs.get("stage_A_preprocessing", ""),
                    kwargs.get("stage_B_vlm_inference", ""),
                    kwargs.get("stage_C_action_dispatch", "")
                ]
                writer.writerow(row_data)
                print(f"✅ CSV写入成功: turn {self.conversation_index}")
        except Exception as e:
            print(f"❌ CSV写入失败: {e}")
            import traceback
            traceback.print_exc()
        
        self.metadata["conversation_count"] += 1
        self._save_metadata()
        
        self.log_terminal_output(f"Conversation turn {self.conversation_index}: {user_text}")
        
        # 返回当前对话索引，用于后续更新TTS时长
        return self.conversation_index
    
    def update_tts_duration(self, conversation_index: int, tts_duration: float):
        """更新指定对话轮次的TTS时长
        
        参数:
            conversation_index: 对话索引（从1开始）
            tts_duration: TTS播报时长（秒）
        """
        # 更新 CSV 文件中的 TTS 时长
        try:
            import csv
            # 读取现有 CSV
            rows = []
            with open(self.csv_log_file, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            # conversation_index从1开始，CSV行索引 = conversation_index（因为第0行是表头）
            row_index = conversation_index  # 第1轮对话在CSV第1行（索引1）
            
            if len(rows) > row_index:  # 确保行存在
                row = rows[row_index]
                if len(row) >= 16:  # 确保有足够的列
                    row[15] = f"{tts_duration:.2f}"  # robot_duration (s) 是第16列（索引15）
                    rows[row_index] = row
                    print(f"🔊 机器人说话时长: {tts_duration:.2f}s")
            
            # 写回 CSV
            with open(self.csv_log_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
                
        except Exception as e:
            print(f"⚠️ 更新CSV中TTS时长失败: {e}")
    
    def update_total_turn_time(self, conversation_index: int, total_turn_time: float):
        """更新指定对话轮次的总时间（到TTS发起）
        
        参数:
            conversation_index: 对话索引
            total_turn_time: 总时间（秒）
        """
        # 更新 CSV 文件中的 total_turn_time
        try:
            import csv
            # 读取现有 CSV
            rows = []
            with open(self.csv_log_file, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            # 更新对应行的 total_turn_time（第7列，索引6）
            if len(rows) > conversation_index:  # 跳过表头
                row = rows[conversation_index]
                if len(row) >= 7:  # 确保有足够的列
                    row[6] = str(total_turn_time)  # total_turn_time (s) 是第7列（索引6）
                    rows[conversation_index] = row
            
            # 写回 CSV
            with open(self.csv_log_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
        except Exception as e:
            print(f"⚠️ 更新CSV中total_turn_time失败: {e}")
        
        # total_turn_time 目前只回填 CSV（TXT 不做回填，避免和 TTS duration 文本更新逻辑耦合）。
    
    def update_human_followed(self, conversation_index: int, human_followed: bool):
        """更新指定对话轮次的 human_followed 字段（JA评估结果）
        
        参数:
            conversation_index: 对话索引
            human_followed: 人类是否跟随机器人注视
        """
        try:
            import csv
            # 读取现有 CSV
            rows = []
            with open(self.csv_log_file, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            # 更新对应行的 human_followed（第13列，索引12）
            if len(rows) > conversation_index:  # 跳过表头
                row = rows[conversation_index]
                if len(row) >= 13:  # 确保有足够的列
                    row[12] = str(human_followed)  # human_followed 是第13列（索引12）
                    rows[conversation_index] = row
            
            # 写回 CSV
            with open(self.csv_log_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
                
            self.log_terminal_output(f"Updated human_followed for turn {conversation_index}: {human_followed}")
        except Exception as e:
            print(f"⚠️ 更新CSV中human_followed失败: {e}")
    
    def save_images(
        self,
        marked_image_path: str = None,
        rgb_image_path: str = None,
        depth_image_path: str = None,
        overlay_image_path: str = None,
    ):
        """保存图像文件"""
        self.image_index += 1
        saved_paths = {}
        
        if marked_image_path and os.path.exists(marked_image_path):
            dest = self.images_dir / f"marked_image_{self.image_index:03d}.jpg"
            shutil.copy2(marked_image_path, dest)
            saved_paths['marked'] = str(dest)
        
        if rgb_image_path and os.path.exists(rgb_image_path):
            dest = self.images_dir / f"rgb_image_{self.image_index:03d}.jpg"
            shutil.copy2(rgb_image_path, dest)
            saved_paths['rgb'] = str(dest)

        if depth_image_path and os.path.exists(depth_image_path):
            dest = self.images_dir / f"depth_image_{self.image_index:03d}.jpg"
            shutil.copy2(depth_image_path, dest)
            saved_paths['depth'] = str(dest)
        
        if overlay_image_path and os.path.exists(overlay_image_path):
            dest = self.images_dir / f"overlay_image_{self.image_index:03d}.jpg"
            shutil.copy2(overlay_image_path, dest)
            saved_paths['overlay'] = str(dest)
        
        if saved_paths:
            self.metadata["image_count"] += 1
            self._save_metadata()
            self.log_terminal_output(f"Saved images set {self.image_index}")
        
        return saved_paths
    
    def log_detected_objects(self, objects: list, timestamp: float = None):
        """记录检测到的物体信息"""
        if timestamp is None:
            timestamp = time.time()
        
        detection_record = {
            "timestamp": timestamp,
            "timestamp_str": datetime.fromtimestamp(timestamp).isoformat(),
            "objects": objects
        }
        
        # 追加到 JSON 文件
        if os.path.exists(self.objects_file):
            with open(self.objects_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"detections": []}
        
        data["detections"].append(detection_record)
        
        with open(self.objects_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.metadata["object_detection_count"] += 1
        self._save_metadata()
    
    def log_ja_metric_event(self, timestamp: float, event_type: str,
                           person_id: int = None, gaze_angle_deg: float = None,
                           distance_m: float = None, object_id: str = None,
                           object_name: str = None, robot_gaze_active: bool = None,
                           human_looking_object: bool = None,
                           shared_gaze_duration: float = None,
                           reaction_latency: float = None):
        """记录 Joint Attention 指标事件到 CSV"""
        with open(self.ja_metrics_file, 'a', encoding='utf-8') as f:
            line = f"{timestamp},{event_type},"
            line += f"{person_id if person_id is not None else ''},"
            line += f"{gaze_angle_deg if gaze_angle_deg is not None else ''},"
            line += f"{distance_m if distance_m is not None else ''},"
            line += f"{object_id if object_id else ''},"
            line += f"{object_name if object_name else ''},"
            line += f"{robot_gaze_active if robot_gaze_active is not None else ''},"
            line += f"{human_looking_object if human_looking_object is not None else ''},"
            line += f"{shared_gaze_duration if shared_gaze_duration is not None else ''},"
            line += f"{reaction_latency if reaction_latency is not None else ''}\n"
            f.write(line)
    
    def close_session(self):
        """关闭会话，保存最终统计"""
        self.metadata["end_time"] = time.time()
        self.metadata["end_time_str"] = datetime.now().isoformat()
        self.metadata["duration_seconds"] = self.metadata["end_time"] - self.metadata["start_time"]
        self._save_metadata()
        
        if getattr(self, "enable_conversation_txt", True):
            with open(self.conversation_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Session ended at {self.metadata['end_time_str']}\n")
                f.write(f"Total conversations: {self.metadata['conversation_count']}\n")
                f.write(f"Total images: {self.metadata['image_count']}\n")
                f.write(f"Total detections: {self.metadata['object_detection_count']}\n")
        
        print(f"✅ 会话日志已关闭: {self.session_dir}")
