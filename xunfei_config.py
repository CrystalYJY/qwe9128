"""
讯飞星火 ASR 配置文件
请在讯飞开放平台获取这些参数: https://console.xfyun.cn/app/myapp
"""

# 讯飞配置 - 请替换为你的实际值
XUNFEI_APPID = "a8febd83"      # 应用 ID
XUNFEI_API_KEY = "1f855c874cf9b7672a87f1fd64e5f545"    # API Key
XUNFEI_API_SECRET = "YzgwMGY3NzljYjkyNGJhNWUxMzk2Y2Uy"  # API Secret

# ASR 参数配置（语音听写流式版）
ASR_CONFIG = {
    "domain": "iat",           # iat=语音听写
    "language": "zh_cn",       # zh_cn=中文
    "accent": "mandarin",      # 普通话
    "sample_rate": 16000,      # 采样率 16k
    "encoding": "raw",         # PCM 格式
    "vad_eos": 5000,           # 后端点检测（静音 5 秒停止）
    "dwa": "wpgs",             # 动态修正
    "ptt": 1,                  # 开启标点符号
}
