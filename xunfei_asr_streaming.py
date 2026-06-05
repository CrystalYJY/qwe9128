#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
讯飞语音听写 - 持续监听版本
自动检测语音开始/结束，实时返回识别结果（流式）
"""
import websocket
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
import subprocess
import struct
import threading


STATUS_FIRST_FRAME = 0
STATUS_CONTINUE_FRAME = 1
STATUS_LAST_FRAME = 2


class XunfeiASRStreaming:
    """讯飞语音听写 - 持续监听流式版本"""
    
    def __init__(self, appid, api_key, api_secret, mic_device="plughw:2,0", 
                 on_result_callback=None, verbose=False):
        """
        初始化
        
        参数:
            appid: 讯飞 APPID
            api_key: 讯飞 API Key
            api_secret: 讯飞 API Secret
            mic_device: 麦克风设备
            on_result_callback: 回调函数 callback(partial_text, is_final)
            verbose: 是否打印调试信息
        """
        self.appid = appid
        self.api_key = api_key
        self.api_secret = api_secret
        self.mic_device = mic_device
        self.on_result_callback = on_result_callback
        self.verbose = verbose
        
        # 识别结果
        self.current_text = ""
        self.last_final_text = ""
        self.last_speech_duration = 0.0  # 最后一次说话时长(秒)
        
        # 控制标志
        self.is_running = False
        self.is_recognizing = False
        
        # 时间记录
        self.speech_start_time = None  # 语音开始时间
        self.speech_end_time = None  # 语音结束时间(检测到静音时)
        self.recognition_start_time = None  # 识别开始时间
        
        # 音频进程
        self.arecord_process = None
        
        # WebSocket
        self.ws = None
        self.ws_connected = False
        self.ws_ready_event = threading.Event()  # 用于等待 WebSocket 连接
        self.frame_status = STATUS_FIRST_FRAME
        
        # 音频缓冲区 - 保存最近的音频帧,防止丢失开头
        self.audio_buffer = []
        self.max_buffer_frames = 15  # 缓冲 15 帧(0.6秒)
        
        # 线程
        self.monitor_thread = None
        
        # 业务参数
        self.common_args = {"app_id": self.appid}
        self.business_args = {
            "domain": "iat",
            "language": "zh_cn",
            "accent": "mandarin",
            "vinfo": 1,
            "vad_eos": 60000  # 60 秒静音才停止
        }
    
    def _log(self, msg):
        if self.verbose:
            print(msg)
    
    def _create_url(self):
        """生成鉴权 URL"""
        url = 'wss://iat-api.xfyun.cn/v2/iat'
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        
        signature_origin = "host: " + "iat-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"
        
        signature_sha = hmac.new(
            self.api_secret.encode('utf-8'),
            signature_origin.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
        
        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.api_key, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        
        v = {
            "authorization": authorization,
            "date": date,
            "host": "iat-api.xfyun.cn"
        }
        
        return url + '?' + urlencode(v)
    
    def start_listening(self):
        """开始持续监听"""
        if self.is_running:
            self._log("⚠️ 已在监听中")
            return
        
        self.is_running = True
        self._log("🎤 开始持续监听...")
        
        # 启动音频采集线程
        threading.Thread(target=self._audio_monitor_loop, daemon=True).start()
    
    def stop_listening(self):
        """停止监听"""
        self.is_running = False
        if self.arecord_process:
            self.arecord_process.terminate()
        self._log("🛑 停止监听")
    
    def _audio_monitor_loop(self):
        """音频监听主循环"""
        frame_size = 8000
        interval = 0.04
        energy_threshold = 500
        
        # 启动 arecord（持续运行）
        arecord_cmd = [
            "arecord",
            "-D", self.mic_device,
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            "-t", "raw"
        ]
        
        self.arecord_process = subprocess.Popen(
            arecord_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        
        time.sleep(0.5)
        
        silence_frames = 0
        speech_frames = 0
        min_speech_frames = 3  # 降低到 3 帧(0.12秒)更容易触发
        max_silence_frames = 13  # 0.5 秒静音结束(从 25 帧改为 13 帧)
        energy_threshold = 300  # 降低能量阈值,让检测更敏感
        frame_count = 0  # 用于调试输出
        non_speech_tolerance = 5  # 允许 5 帧(0.2秒)的低能量波动
        non_speech_count = 0  # 低能量帧计数器
        
        while self.is_running:
            try:
                # 读取音频
                buf = b""
                while len(buf) < frame_size:
                    chunk = self.arecord_process.stdout.read(frame_size - len(buf))
                    if not chunk:
                        break
                    buf += chunk
                
                if len(buf) < frame_size:
                    break
                
                frame_count += 1
                
                # 计算能量
                samples = struct.unpack(f"<{len(buf)//2}h", buf)
                energy = sum(abs(s) for s in samples) / len(samples)
                has_speech = energy > energy_threshold
                
                if has_speech:
                    speech_frames += 1
                    silence_frames = 0
                    non_speech_count = 0  # 重置低能量计数器
                    
                    # 记录语音开始时间
                    if self.speech_start_time is None:
                        self.speech_start_time = time.time()
                    
                    # 检测到足够长的语音，开始识别
                    if not self.is_recognizing and speech_frames >= min_speech_frames:
                        self._log(f"🗣️ 检测到语音")
                        self.recognition_start_time = time.time()
                        self._start_recognition()
                        
                        # 发送缓冲区中的音频(开头部分,不包括当前帧)
                        for buffered_frame in self.audio_buffer:
                            self._send_audio_frame(buffered_frame)
                        self.audio_buffer.clear()
                        
                        # 发送当前帧
                        self._send_audio_frame(buf)
                    else:
                        # 添加到缓冲区(仅在未开始识别时)
                        if not self.is_recognizing:
                            self.audio_buffer.append(buf)
                            if len(self.audio_buffer) > self.max_buffer_frames:
                                self.audio_buffer.pop(0)
                        
                        if not self.is_recognizing and self.verbose:
                            # 显示语音积累进度
                            if speech_frames % 5 == 0:
                                self._log(f"🎵 语音帧积累: {speech_frames}/{min_speech_frames}")
                        
                        # 如果已经在识别,发送当前音频
                        if self.is_recognizing:
                            self._send_audio_frame(buf)
                
                else:
                    silence_frames += 1
                    
                    # 如果正在识别中第一次检测到静音,记录语音结束时间
                    if self.is_recognizing and silence_frames == 1:
                        self.speech_end_time = time.time()
                        
                        # 快速模式:如果已经有识别文本,提前触发回调
                        if self.current_text and self.on_result_callback:
                            self.on_result_callback(self.current_text, False)
                    
                    # 非语音时也保持小缓冲(包含前导静音)
                    if not self.is_recognizing:
                        self.audio_buffer.append(buf)
                        if len(self.audio_buffer) > self.max_buffer_frames:
                            self.audio_buffer.pop(0)
                        
                        # 容错机制:允许短暂的低能量波动
                        non_speech_count += 1
                        if non_speech_count > non_speech_tolerance:
                            speech_frames = 0  # 超过容错范围才重置
                            non_speech_count = 0
                    
                    # 如果正在识别，继续发送（保持连接）
                    if self.is_recognizing:
                        self._send_audio_frame(buf)
                        
                        # 动态调整结束条件
                        # 如果已有足够文本(>5字符),缩短等待时间
                        required_silence = max_silence_frames
                        if len(self.current_text) > 5:
                            required_silence = max(7, max_silence_frames // 2)  # 缩短到 0.25 秒或更少
                        
                        # 检测静音结束
                        if silence_frames >= required_silence:
                            # 计算实际说话时长(不包括静音等待时间)
                            if self.speech_start_time and self.speech_end_time:
                                self.last_speech_duration = self.speech_end_time - self.speech_start_time
                                self.speech_start_time = None
                                self.speech_end_time = None
                            
                            # 立即显示当前识别结果(不等最终结果)
                            if self.current_text:
                                self.last_final_text = self.current_text
                                self._log(f"💬 {self.current_text} (用时: {self.last_speech_duration:.1f}秒)")
                            
                            self._stop_recognition()
                            speech_frames = 0
                            silence_frames = 0
                    else:
                        speech_frames = 0
                
                time.sleep(interval)
                
            except Exception as e:
                self._log(f"❌ 监听错误: {e}")
                break
        
        if self.arecord_process:
            self.arecord_process.terminate()
    
    def _start_recognition(self):
        """开始识别会话"""
        self.current_text = ""
        self.is_recognizing = True
        self.frame_status = STATUS_FIRST_FRAME
        self.ws_connected = False
        self.ws_ready_event.clear()  # 清除事件标志
        
        # WebSocket 回调
        def on_message(ws, message):
            try:
                data = json.loads(message)
                code = data["code"]
                
                if code != 0:
                    return
                
                result_data = data["data"]["result"]["ws"]
                text = ""
                for i in result_data:
                    for w in i["cw"]:
                        text += w["w"]
                
                if text:
                    self.current_text += text
                    
                    # 判断是否为最终结果
                    is_final = (data["data"]["status"] == 2)
                    
                    # 实时回调
                    if self.on_result_callback:
                        self.on_result_callback(self.current_text, is_final)
                    
                    # 保存最终结果(但不在这里输出,在静音检测时输出更快)
                    if is_final:
                        self.last_final_text = self.current_text

                    
            except Exception as e:
                self._log(f"解析错误: {e}")
        
        def on_error(ws, error):
            pass
        
        def on_close(ws, a, b):
            self.ws_connected = False
        
        def on_open(ws):
            self.ws_connected = True
            # 发送第一帧
            self._send_first_frame()
            # 通知主线程可以开始发送了
            self.ws_ready_event.set()
        
        # 创建 WebSocket
        websocket.enableTrace(False)
        ws_url = self._create_url()
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        self.ws.on_open = on_open
        
        # 在新线程中运行
        threading.Thread(target=lambda: self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE}), daemon=True).start()
        
        # 等待连接建立（最多等待 3 秒）
        if not self.ws_ready_event.wait(timeout=3.0):
            self._log("❌ WebSocket 连接超时")
            self.is_recognizing = False
            return
    
    def _stop_recognition(self):
        """停止识别会话"""
        if not self.is_recognizing:
            return
        
        self.is_recognizing = False
        
        # 发送最后一帧
        self._send_last_frame()
        
        # 异步延迟关闭,不阻塞主线程
        def close_later():
            time.sleep(0.5)  # 缩短等待时间到 0.5 秒
            if self.ws:
                self.ws.close()
        
        threading.Thread(target=close_later, daemon=True).start()
    
    def _send_first_frame(self):
        """发送第一帧"""
        if not self.ws_connected:
            return
        
        frame_data = {
            "common": self.common_args,
            "business": self.business_args,
            "data": {
                "status": 0,
                "format": "audio/L16;rate=16000",
                "audio": "",
                "encoding": "raw"
            }
        }
        self.ws.send(json.dumps(frame_data))
        self.frame_status = STATUS_CONTINUE_FRAME
    
    def _send_audio_frame(self, audio_data):
        """发送音频帧"""
        if not self.ws_connected or not self.is_recognizing:
            return
        
        if self.frame_status == STATUS_FIRST_FRAME:
            self._send_first_frame()
        
        audio_base64 = str(base64.b64encode(audio_data), 'utf-8')
        frame_data = {
            "data": {
                "status": 1,
                "format": "audio/L16;rate=16000",
                "audio": audio_base64,
                "encoding": "raw"
            }
        }
        self.ws.send(json.dumps(frame_data))
    
    def _send_last_frame(self):
        """发送最后一帧"""
        if not self.ws_connected:
            return
        
        frame_data = {
            "data": {
                "status": 2,
                "format": "audio/L16;rate=16000",
                "audio": "",
                "encoding": "raw"
            }
        }
        self.ws.send(json.dumps(frame_data))
    
    def get_last_text(self):
        """获取最后一次完整识别结果"""
        return self.last_final_text
    
    def get_last_speech_duration(self):
        """获取最后一次说话时长(秒)"""
        return self.last_speech_duration
    
    def clear_text(self):
        """清除识别结果"""
        self.last_final_text = ""
        self.current_text = ""
        self.last_speech_duration = 0.0


def main():
    """测试函数"""
    from xunfei_config import XUNFEI_APPID, XUNFEI_API_KEY, XUNFEI_API_SECRET
    
    print("\n" + "="*60)
    print("讯飞语音听写 - 持续监听测试")
    print("="*60)
    print("程序会自动监听并识别，实时返回结果")
    print("说话后保持 0.5 秒静音即可自动结束")
    print("按 Ctrl+C 退出")
    print("="*60 + "\n")
    
    def on_result(text, is_final):
        """结果回调 - 实际使用时在这里处理识别结果"""
        pass  # 内部已经输出了,这里不重复输出
    
    asr = XunfeiASRStreaming(
        XUNFEI_APPID,
        XUNFEI_API_KEY,
        XUNFEI_API_SECRET,
        mic_device="plughw:Device",
        on_result_callback=on_result,
        verbose=True
    )
    
    asr.start_listening()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 退出")
        asr.stop_listening()


if __name__ == "__main__":
    main()
