#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
讯飞ASR适配器 - 兼容原ASRModule接口
用于替换基于ROS2的ASR，保持与主程序的接口一致
"""
import time
import threading
from xunfei_asr_streaming import XunfeiASRStreaming
from xunfei_config import XUNFEI_APPID, XUNFEI_API_KEY, XUNFEI_API_SECRET


class XunfeiASRAdapter:
    """
    讯飞ASR适配器
    提供与原ASRModule相同的接口，但使用讯飞流式识别
    """
    
    def __init__(self, mic_device="plughw:2,0", verbose=False):
        """
        初始化讯飞ASR适配器
        
        参数:
            mic_device: 麦克风设备 (USB Mic 为 plughw:2,0)
            verbose: 是否显示详细日志
        """
        self.text = ""  # 当前识别文本（兼容原接口）
        self.tts_is_playing = False  # TTS播放状态
        self.is_speech_active = False  # 语音活动状态（兼容原接口）
        self._last_speech_duration = None  # 最后一次说话时长
        self._last_speech_start = None  # 语音开始时间（兼容原接口，暂未使用）
        
        self.tts_stopped_time = 0.0  # TTS停止时间
        self.cooldown_duration = 0.0  # 取消冷却期，估算时长一到立即接受ASR
        
        # 线程锁
        self._text_lock = threading.Lock()
        
        # TTS抑制期间缓存的识别结果
        self._suppressed_results = []
        
        # 初始化讯飞ASR
        self.asr = XunfeiASRStreaming(
            XUNFEI_APPID,
            XUNFEI_API_KEY,
            XUNFEI_API_SECRET,
            mic_device=mic_device,
            on_result_callback=self._on_result_callback,
            verbose=verbose
        )
        
        # 启动监听
        self.asr.start_listening()
        print("🎤 讯飞ASR已启动，持续监听中...")
    
    def _on_result_callback(self, text, is_final):
        """
        识别结果回调
        
        参数:
            text: 识别文本
            is_final: 是否为最终结果
        """
        if is_final and text and text.strip():
            # 检查是否在TTS播放期间或冷却期
            current_time = time.time()
            if self.tts_is_playing or (current_time - self.tts_stopped_time < self.cooldown_duration):
                print(f"🔇 ASR被抑制(TTS播放中): '{text.strip()}'")
                # 缓存被抑制的结果，避免丢失
                with self._text_lock:
                    self._suppressed_results.append({
                        'text': text.strip(),
                        'duration': self.asr.get_last_speech_duration(),
                        'time': current_time
                    })
                return
            
            with self._text_lock:
                self.text = text.strip()
                # 获取说话时长
                self._last_speech_duration = self.asr.get_last_speech_duration()
                
            # 输出识别结果（与原ASR风格一致）
            if self._last_speech_duration is not None:
                print(f"🗣️ ASR: '{text}' (时长: {self._last_speech_duration:.2f}s)")
            else:
                print(f"🗣️ ASR: '{text}'")
    
    def get_text(self):
        """
        获取当前识别文本（兼容原接口）
        """
        with self._text_lock:
            return self.text
    
    def get_last_speech_duration(self):
        """
        获取最后一次说话时长（秒）
        不进行校正，返回实际说话时长
        """
        if self._last_speech_duration is None:
            return None
        try:
            return float(self._last_speech_duration)
        except Exception:
            return None
    
    def set_tts_playing(self, is_playing, estimated_duration=None):
        """
        设置TTS播放状态，用于抑制ASR识别
        
        参数:
            is_playing: True表示TTS正在播放，False表示TTS已停止
            estimated_duration: 估算的TTS播放时长(秒)，用于自动解除抑制
        """
        self.tts_is_playing = is_playing
        if not is_playing:
            # TTS停止时记录时间，用于冷却期判断
            self.tts_stopped_time = time.time()
            # 清空被抑制的结果
            with self._text_lock:
                if self._suppressed_results:
                    self._suppressed_results.clear()
        else:
            # 如果提供了估算时长，启动定时器自动解除抑制
            if estimated_duration and estimated_duration > 0:
                def auto_release():
                    time.sleep(estimated_duration + self.cooldown_duration)
                    if self.tts_is_playing:  # 如果还在播放状态，自动解除
                        self.set_tts_playing(False)
                threading.Thread(target=auto_release, daemon=True).start()
    
    def destroy_node(self):
        """
        销毁节点（兼容原接口）
        停止ASR监听
        """
        print("🛑 停止讯飞ASR...")
        self.asr.stop_listening()
    
    def get_logger(self):
        """
        返回日志对象（兼容原接口）
        """
        class DummyLogger:
            def info(self, msg):
                print(f"ℹ️ {msg}")
            
            def warn(self, msg):
                print(f"⚠️ {msg}")
            
            def error(self, msg):
                print(f"❌ {msg}")
        
        return DummyLogger()


# 兼容性别名（可以用 ASRModule 导入）
ASRModule = XunfeiASRAdapter


def main():
    """测试函数"""
    print("\n" + "="*60)
    print("讯飞ASR适配器测试")
    print("="*60)
    print("模拟原ASRModule的使用方式")
    print("按 Ctrl+C 退出")
    print("="*60 + "\n")
    
    # 初始化（与原ASRModule相同的方式）
    asr = ASRModule()
    
    prev_text = ""
    
    try:
        while True:
            # 轮询获取文本（与原主程序相同的方式）
            text = asr.get_text()
            
            if text and text != prev_text:
                # 获取说话时长
                duration = asr.get_last_speech_duration()
                
                if duration is not None:
                    print(f"\n✅ 检测到新文本: '{text}' (时长: {duration:.2f}s)")
                else:
                    print(f"\n✅ 检测到新文本: '{text}'")
                
                prev_text = text
                # 清空文本（与原主程序相同）
                asr.text = ""
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n👋 退出")
        asr.destroy_node()


if __name__ == "__main__":
    main()
