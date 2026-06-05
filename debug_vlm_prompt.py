#!/usr/bin/env python3
"""
VLM Prompt 调试工具
用于独立测试豆包视觉模型的prompt效果，无需连接机器人或摄像头
"""

import os
import base64
import json
from openai import OpenAI
import httpx
import cv2
import numpy as np
from pathlib import Path


class VLMPromptDebugger:
    def __init__(self):
        # 初始化OpenAI客户端,设置更长的超时时间(120秒)
        http_client = httpx.Client(
            timeout=30.0,  # 增加到120秒,避免请求超时
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        
        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key="4cf75ba1-9bb7-4004-9b13-39510f9afe29",
            http_client=http_client,
        )
        self.model_name = "doubao-seed-1-6-flash-250828"
        self.conversation_history = []
        
        # 上下文管理设置
        # 注意：每轮对话都会携带图片，token消耗较大
        # - 1轮: 最节省token，但记忆很短
        # - 2-3轮: 平衡记忆和成本，推荐用于调试
        # - 5轮: 较长记忆，适合复杂对话，但token消耗大
        self.max_history_turns = 3  # 默认保存3轮对话（6条消息）
        
    def load_image_as_base64(self, image_path: str) -> str:
        """从文件路径加载图片并转换为base64"""
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')
    
    def get_vision_prompt(self) -> str:
        """获取当前的vision prompt（可以在这里修改测试不同的prompt）"""
        return """You are my desk organization assistant. You can see the room and help me arrange items. The person in the picture is me.

⚠️ IMPORTANT: Each image is NEW. Previous bounding boxes are INVALID. Always detect based on CURRENT image only.

📋 OUTPUT FORMAT:
1. Answer: [Your response in English]
2. Chinese: [中文翻译]

Rules:
- Keep answers natural and concise
- Don't say "in the image/picture"

🔴 BOUNDING BOX RULE - READ CAREFULLY:

✅ Output JSON ONLY when user explicitly mentions a specific object name:
   - "这些笔" / "the pens" → mentions "笔/pen" → output JSON
   - "黑色支架" / "black stand" → mentions "支架/stand" → output JSON
   - "键盘" / "keyboard" → mentions "键盘/keyboard" → output JSON

❌ DO NOT output JSON when user does NOT mention specific objects:
   - "帮我整理桌子" / "organize my desk" → NO object name → NO JSON
   - "桌面很乱" / "desk is messy" → NO object name → NO JSON
   - "收拾一下" / "clean up" → NO object name → NO JSON

The rule is simple: If user's message contains NO specific object name, output NO JSON.

Examples:
User: "这些笔应该怎么放?"
→ User mentions "笔" → MUST output JSON with all visible pens
1. Answer: Let's put all the pens in the pen holder to keep them organized.
2. Chinese: 我们把所有的笔都放进笔筒里，这样会更整齐。
```json
{"objects": [{"object": "pen", "bboxes": [[x1,y1,x2,y2], [x1,y1,x2,y2]], "confidence": 0.9}], "image_size": [1280,720]}
```

User: "帮我整理一下桌子"
→ NO specific object mentioned → NO JSON
1. Answer: Let's organize it step by step!
2. Chinese: 我们一步步来整理吧！

User: "键盘放这儿合适吗?"
→ User mentions "键盘" → MUST output JSON with keyboard
1. Answer: Yes, the keyboard position looks good!
2. Chinese: 是的，键盘的位置看起来不错！
```json
{"objects": [{"object": "keyboard", "bboxes": [x1,y1,x2,y2], "confidence": 0.95}], "image_size": [1280,720]}
```

JSON format:
- Use normalized coordinates [0-1000]
- bboxes: [[x1,y1,x2,y2], ...] for multiple, [x1,y1,x2,y2] for single
- Include ALL instances of the mentioned object
"""

    def call_vlm(self, user_text: str, img_base64: str) -> dict:
        """调用VLM获取响应"""
        prompt = self.get_vision_prompt()
        
        # 🔥 优化策略：只在最新一轮带图片，历史对话只保留文字
        # 这样既能保持对话记忆，又能大幅节省token
        messages = []
        
        # 添加历史对话（只保留文字部分）
        for msg in self.conversation_history:
            if msg["role"] == "user":
                # 如果是user消息，提取文字部分
                if isinstance(msg["content"], list):
                    # 从多模态消息中提取文字
                    text_content = ""
                    for item in msg["content"]:
                        if item["type"] == "text":
                            text_content = item["text"]
                            break
                    messages.append({"role": "user", "content": text_content})
                else:
                    # 已经是纯文字
                    messages.append(msg)
            else:
                # assistant消息直接添加
                messages.append(msg)
        
        # 构建当前请求的完整消息（带图片）
        current_message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": f"{prompt}\n用户请求: {user_text}"
                }
            ]
        }
        
        messages.append(current_message)
        
        print("\n🔄 正在调用VLM...")
        print(f"📊 [Token优化] 发送 {len(messages)} 条消息（仅最新一轮带图片）")
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=500,
            top_p=1.0,
        )
        
        reply = response.choices[0].message.content
        
        # 解析响应
        result = {
            "raw_response": reply,
            "speech_response": "",
            "chinese": "",
            "objects": [],
            "image_size": None
        }
        
        # 提取Answer和Chinese
        if 'Answer:' in reply:
            speech_text = reply.split('Answer:')[1]
            if 'Chinese:' in speech_text:
                parts = speech_text.split('Chinese:')
                result["speech_response"] = parts[0].strip()
                chinese_part = parts[1]
                # 移除可能的JSON标记
                if '```' in chinese_part:
                    result["chinese"] = chinese_part.split('```')[0].strip()
                else:
                    result["chinese"] = chinese_part.strip()
            else:
                # 没有Chinese部分,移除可能的JSON标记
                if '```' in speech_text:
                    result["speech_response"] = speech_text.split('```')[0].strip()
                else:
                    result["speech_response"] = speech_text.strip()
            
            # 清理可能的标记
            result["speech_response"] = result["speech_response"].replace('1.', '').replace('2.', '').strip()
            result["chinese"] = result["chinese"].replace('2.', '').strip()
        
        # 提取JSON
        if '```json' in reply:
            json_str = reply.split("```json")[1].split("```")[0].strip()
            try:
                data = json.loads(json_str)
                if "objects" in data:
                    result["objects"] = data["objects"]
                if "image_size" in data:
                    result["image_size"] = data["image_size"]
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON解析失败: {e}")
        
        # 🔥 改进：保存完整的对话上下文（包括图片）
        # 这样VLM能记住物体位置和之前的讨论
        if result.get("speech_response") and result["speech_response"] != "未知回复":
            # 保存用户消息（带图片）
            self.conversation_history.append(current_message)
            
            # 保存助手回复（只保存纯文本答案，不包含JSON）
            self.conversation_history.append({
                "role": "assistant",
                "content": result["speech_response"]
            })
            
            # 限制历史长度
            if len(self.conversation_history) > self.max_history_turns * 2:
                self.conversation_history = self.conversation_history[-self.max_history_turns * 2:]
            
            # 调试：打印当前历史轮数
            current_turns = len(self.conversation_history) // 2
            print(f"📝 [对话历史] 当前保存 {current_turns} 轮（上限 {self.max_history_turns} 轮）")
        
        return result
    
    def draw_bboxes(self, image_path: str, objects: list, image_size: list = None) -> np.ndarray:
        """在图片上绘制bounding boxes"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ 无法读取图片: {image_path}")
            return None
        
        IMG_H, IMG_W = img.shape[:2]
        
        # 如果有image_size，进行坐标转换
        if image_size and len(image_size) == 2:
            scale_x = IMG_W / 1000
            scale_y = IMG_H / 1000
        else:
            scale_x = scale_y = 1.0
        
        for obj in objects:
            if "bboxes" not in obj:
                continue
            
            bboxes = obj["bboxes"]
            
            # 🔥 修复：处理多个bbox的情况（VLM可能返回多个物体）
            # 判断是单个bbox还是多个bbox
            if not bboxes:
                continue
            
            # 检查是否是嵌套列表（多个bbox）
            if isinstance(bboxes[0], list):
                # 多个bbox：[[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
                bbox_list = bboxes
            else:
                # 单个bbox：[x1,y1,x2,y2]
                bbox_list = [bboxes]
            
            # 为每个bbox绘制
            for idx, bbox in enumerate(bbox_list):
                if len(bbox) != 4:
                    continue
                
                x1, y1, x2, y2 = bbox
                
                # 坐标转换
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # 绘制矩形框
                color = (0, 255, 0)  # 绿色
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                label = obj.get("object", obj.get("name", "unknown"))
                confidence = obj.get("confidence", 1.0)
                
                # 如果有多个bbox，添加编号
                if len(bbox_list) > 1:
                    label_text = f"{label}_{idx+1} ({confidence:.2f})"
                else:
                    label_text = f"{label} ({confidence:.2f})"
                
                # 计算文字背景框
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                
                # 绘制文字背景
                cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                
                # 绘制文字
                cv2.putText(img, label_text, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
        
        return img
    
    def display_result(self, result: dict, image_path: str):
        """显示VLM结果"""
        print("\n" + "="*60)
        print("📊 VLM 响应结果")
        print("="*60)
        print(f"\n💬 英文回复: {result['speech_response']}")
        if result.get('chinese'):
            print(f"🇨🇳 中文翻译: {result['chinese']}")
        
        if result['objects']:
            print(f"\n🔍 检测到 {len(result['objects'])} 个物体:")
            for i, obj in enumerate(result['objects'], 1):
                name = obj.get("object", obj.get("name", "unknown"))
                bbox = obj.get("bboxes", [])
                conf = obj.get("confidence", 0.0)
                print(f"  [{i}] {name}: bbox={bbox}, confidence={conf:.2f}")
            
            # 绘制并显示bounding boxes
            annotated_img = self.draw_bboxes(image_path, result['objects'], result['image_size'])
            if annotated_img is not None:
                # 缩放图片以适应屏幕
                h, w = annotated_img.shape[:2]
                max_size = 1200
                if w > max_size or h > max_size:
                    scale = max_size / max(w, h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    annotated_img = cv2.resize(annotated_img, (new_w, new_h))
                
                cv2.imshow('VLM Detection Result', annotated_img)
                print("\n👁️ 已显示标注图片，按任意键继续...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("\n🔍 未检测到物体")
        
        print("\n📝 原始响应:")
        print("-" * 60)
        print(result['raw_response'])
        print("="*60 + "\n")


def main():
    print("="*60)
    print("🔧 VLM Prompt 调试工具")
    print("="*60)
    print("\n使用说明:")
    print("  1. 每次对话都输入图片路径（模拟实时摄像头）")
    print("  2. 输入用户消息进行测试")
    print("  3. 查看VLM响应和bounding box可视化")
    print("  4. 输入 'show' 查看当前图片")
    print("  5. 输入 'clear' 清空对话历史")
    print("  6. 输入 'quit' 或 'exit' 退出")
    print("\n💡 提示: 每次对话会自动使用新图片，保持对话记忆")
    print("\n" + "="*60 + "\n")
    
    debugger = VLMPromptDebugger()
    current_image_path = None
    
    while True:
        # 每次对话都输入图片路径
        print("\n" + "-"*60)
        image_path = input("📸 请输入本轮对话的图片路径 (或输入命令): ").strip()
        
        if image_path.lower() in ['quit', 'exit']:
            print("👋 再见！")
            break
        
        if image_path.lower() == 'clear':
            debugger.conversation_history = []
            print("🔄 已清空对话历史")
            continue
        
        if image_path.lower() == 'show':
            if current_image_path:
                img = cv2.imread(current_image_path)
                if img is not None:
                    h, w = img.shape[:2]
                    max_size = 1200
                    if w > max_size or h > max_size:
                        scale = max_size / max(w, h)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        img = cv2.resize(img, (new_w, new_h))
                    cv2.imshow('Current Image', img)
                    print("👁️ 已显示图片，按任意键关闭...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                print("⚠️ 还没有加载过图片")
            continue
        
        if not os.path.exists(image_path):
            print(f"❌ 图片不存在: {image_path}")
            continue
        
        try:
            current_img_base64 = debugger.load_image_as_base64(image_path)
            current_image_path = image_path
            print(f"✅ 已加载图片: {image_path}")
        except Exception as e:
            print(f"❌ 加载图片失败: {e}")
            continue
        
        # 获取用户消息
        user_message = input("💬 用户消息: ").strip()
        
        if user_message.lower() in ['quit', 'exit']:
            print("� 再见！")
            break
        
        if not user_message:
            print("⚠️ 请输入消息")
            continue
        
        # 调用VLM
        try:
            result = debugger.call_vlm(user_message, current_img_base64)
            debugger.display_result(result, current_image_path)
        except Exception as e:
            print(f"❌ VLM调用失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
