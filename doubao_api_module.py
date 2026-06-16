import os
import base64
import rclpy
from openai import OpenAI
import json
import traceback
from image_save_module import ImageSaveModule
from typing import Dict, List, Optional
import httpx

class DoubaoAPIModule:
    def __init__(self, prompt_profile: str = "vision"):
        # 使用httpx客户端启用连接复用（暂不启用HTTP/2以避免额外依赖）
        http_client = httpx.Client(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        
        # 初始化OpenAI客户端，配置豆包API
        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",  # 豆包API endpoint
            api_key="---",  # 替换为你的豆包API key
            http_client=http_client  # 使用自定义HTTP客户端，启用连接复用
        )
        self.model_name = "doubao-seed-1-6-vision-250815"  # 你的模型接入点ID
        self.conversation_history = []  # 用于多轮对话
        self.max_history_turns = 5  # 只保留1轮对话（2条消息），平衡记忆与bbox准确性
        self.object_coordinates: Dict[str, List[float]] = {}  # 格式: {"杯子": [x,y,z], ...}
        self.last_detected_objects: List[Dict] = []  # 存储最后一次检测到的物体信息
        self.vision_mode = False

        # prompt_profile:
        # - "vision": 实验组（2.py）保持原行为与原提示词
        # - "blind":  对照组（speak.py）使用纯文本盲人提示词（不依赖有图无图）
        self.prompt_profile = prompt_profile
    
    def clear_conversation_history(self):
        """清空对话历史（场景变化时调用）"""
        self.conversation_history = []
        rclpy.logging.get_logger("doubao_api").info("🔄 场景变化，已清空对话历史")
    
    def clean_bbox_from_history(self):
        """从对话历史中移除包含 bboxes 的回复，避免旧物体位置污染新识别
        
        保留纯文本对话内容，只清除包含 JSON 坐标的消息。
        """
        cleaned_history = []
        for msg in self.conversation_history:
            if msg["role"] == "assistant":
                content = msg.get("content", "")
                # 如果回复中包含 bboxes 或 JSON 代码块，跳过这条消息
                if "bboxes" in content or "```json" in content or '"objects"' in content:
                    continue
            cleaned_history.append(msg)
        
        self.conversation_history = cleaned_history
        if len(cleaned_history) < len(self.conversation_history):
            rclpy.logging.get_logger("doubao_api").info(f"🧹 已清理历史中的物体坐标信息")

    def keep_last_rounds(self, n: int = 1):
        """保留最近 n 轮（user+assistant），用于在场景变化时保留上一轮对话记忆。

        例如 n=1 会保留最近的 2 条消息（用户+助手）。
        """
        try:
            keep_messages = n * 2
            if len(self.conversation_history) > keep_messages:
                self.conversation_history = self.conversation_history[-keep_messages:]
            rclpy.logging.get_logger("doubao_api").info(f"🔒 场景变化，保留最近 {n} 轮对话记忆")
        except Exception:
            # 安全兜底：出错时清空
            self.conversation_history = []
            rclpy.logging.get_logger("doubao_api").info("🔄 场景变化，保留历史时出错，已清空对话历史")
    
    def trim_conversation_history(self):
        """保留最近N轮对话，避免token消耗过多"""
        max_messages = self.max_history_turns * 2  # 每轮包含user和assistant两条消息
        if len(self.conversation_history) > max_messages:
            # 保留最近的N轮对话
            self.conversation_history = self.conversation_history[-max_messages:]
            rclpy.logging.get_logger("doubao_api").info(f"📝 对话历史已精简至最近{self.max_history_turns}轮")

    def process_user_input(
        self,
        user_text: str,
        img_base64: Optional[str] = None,
        debug: bool = False,
        include_image: bool = True,
        session_logger=None,
    ) -> Dict:
        """
        处理用户输入
        :param user_text: 用户文本
        :param img_base64: 图片的base64编码
        :param debug: 是否开启调试模式
        :param include_image: 是否在本次请求中包含图片（False时复用历史中的图片上下文）
        """
        return self.call_doubao_vision(
            img_base64,
            user_text,
            debug,
            include_image,
            session_logger=session_logger,
        )

    def _build_vision_prompt(self) -> str:
        return """You are my desk organization assistant. You can see the room and help me arrange items. The person in the picture is me.

⚠️ IMPORTANT: Each image is NEW. Previous bounding boxes are INVALID. Always detect based on CURRENT image only.

📋 OUTPUT FORMAT:
1. Answer: [Your response in English]
2. Chinese: [中文翻译]

Rules:
- Keep answers natural and concise
- Don't say "in the image/picture"

� BOUNDING BOX RULE - READ CAREFULLY:

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
{"objects": [{"object": "keyboard", "bboxes": [[x1,y1,x2,y2]], "confidence": 0.95}], "image_size": [1280,720]}
```

JSON format:
- Use normalized coordinates [0-1000]
- bboxes: Always use nested array format [[x1,y1,x2,y2], ...], even for single object
- Include ALL instances of the mentioned object
"""

    def _build_blind_prompt(self) -> str:
        return """You are a social robot WITHOUT vision (blind control group). You cannot see anything.

OUTPUT FORMAT:
1. Answer: [natural English response]
2. Chinese: [中文翻译]

Rules:
- Keep answers natural and friendly
- Do NOT claim you can see
- Suggest where to look if asked about objects
- Do NOT output JSON
"""

    def call_doubao_vision(
        self,
        img_base64: Optional[str],
        user_text: str,
        debug_mode: bool = False,
        include_image: bool = True,
        session_logger=None,
    ) -> Dict:
        try:
            # 通过显式 profile 选择提示词；不依赖"有图/无图"。
            # - vision: 原实验组提示词
            # - blind:  盲人对照组提示词
            prompt = self._build_blind_prompt() if self.prompt_profile == "blind" else self._build_vision_prompt()

            # 🔥 改进：使用N轮记忆（只存Answer，不含JSON），平衡智能性与bbox准确性
            messages = self.conversation_history.copy()
            
            # 调试：打印即将发送给 VLM 的历史对话
            if messages:
                print(f"📚 [对话历史] 本次请求携带 {len(messages)} 条历史消息：")
                for i, msg in enumerate(messages):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    # 只打印前100个字符，避免刷屏
                    content_preview = str(content)[:100] + "..." if len(str(content)) > 100 else str(content)
                    print(f"  [{i+1}] {role}: {content_preview}")
            
            # 强制每轮对话都发送图片（避免 bbox 乱飞）
            if img_base64:
                messages.append({
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
                })
            else:
                # 兜底：盲人模式或无图时才使用纯文本
                messages.append({
                    "role": "user",
                    "content": f"{prompt}\n用户请求: {user_text}"
                })

            # 调用豆包视觉模型 - 优化参数加速响应
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,  # 适度提高，在速度和准确性间平衡
                max_tokens=500,   # 减少到500，显著加速
                top_p=1.0,        # 设为1消除采样差异，减少幻觉
            )
    
            reply = response.choices[0].message.content
            result = {
                "speech_response": reply,
                "raw_response": reply,
                "objects": [],
                "image_size": None
            }

            try:
                # 提取自然语言回复（确保只提取需要播报的部分）
                speech_text = reply
                chinese_translation = None  # 用于存储中文翻译
                
                # 新格式: "1. Answer: ... 2. Chinese: ..."
                if 'Answer:' in reply or 'answer:' in reply:
                    # 提取Answer部分
                    if 'Answer:' in reply:
                        speech_text = reply.split('Answer:')[1]
                    elif 'answer:' in reply:
                        speech_text = reply.split('answer:')[1]
                    
                    # 尝试提取中文翻译（如果存在）
                    if 'Chinese:' in speech_text or 'chinese:' in speech_text:
                        if 'Chinese:' in speech_text:
                            parts = speech_text.split('Chinese:')
                            speech_text = parts[0].strip()
                            chinese_part = parts[1]
                        else:
                            parts = speech_text.split('chinese:')
                            speech_text = parts[0].strip()
                            chinese_part = parts[1]
                        
                        # 移除可能的JSON标记
                        if '```' in chinese_part:
                            chinese_translation = chinese_part.split('```')[0].strip()
                        else:
                            chinese_translation = chinese_part.strip()
                    
                    # 移除"2."及之后的所有内容
                    if '2.' in speech_text:
                        speech_text = speech_text.split('2.')[0]
                    # 移除可能的JSON部分
                    if '```' in speech_text:
                        speech_text = speech_text.split('```')[0]
                    
                    speech_text = speech_text.strip()
                    
                # 旧格式: "回答：... 意图分析: ..."
                elif '回答：' in reply:
                    speech_text = reply.split('回答：')[1].split('意图分析')[0].strip()
                elif '回答:' in reply:
                    speech_text = reply.split('回答:')[1].split('意图分析')[0].strip()
                else:
                    # 如果没有明确标记，尝试其他分割方式
                    if '```json' in reply:
                        speech_text = reply.split('```json')[0].strip()
                    elif '『' in reply:
                        speech_text = reply.split('『')[0].strip()
                    
                    # 清理"1."和"Answer:"标记
                    speech_text = speech_text.replace('1.', '').replace('Answer:', '').replace('answer:', '').strip()
                
                # 最后清理所有可能残留的标记符号
                speech_text = speech_text.replace('回答：', '').replace('回答:', '').strip()
                speech_text = speech_text.replace('1.', '').replace('2.', '').strip()
                speech_text = speech_text.replace('『', '').replace('』', '').strip()
                # 移除可能的多余换行符，合并为单行
                speech_text = ' '.join(speech_text.split())
                
                # 如果提取到中文翻译，打印到终端（不发送给TTS）
                if chinese_translation:
                    print(f"🇨🇳🇨🇳 🇨🇳🇨🇳🇨🇳🇨🇳[中文] {chinese_translation}")
                    if session_logger is not None:
                        try:
                            session_logger.log_terminal_output(f"[中文翻译] {chinese_translation}")
                        except Exception:
                            pass

                # 兜底：有些情况下模型会只返回 JSON（例如 ```json {...} ```），导致 speech_text 变成空串。
                # 这会进一步导致 TTS 发送空字符串，影响实验记录。
                if not speech_text:
                    try:
                        # 先尝试从 JSON 中提取第一个 object 名称
                        json_part = None
                        if '```json' in reply:
                            json_part = reply.split('```json', 1)[1]
                            if '```' in json_part:
                                json_part = json_part.split('```', 1)[0]
                        elif reply.strip().startswith('{') and reply.strip().endswith('}'):
                            json_part = reply

                        obj_name = None
                        if json_part:
                            data = json.loads(json_part)
                            if isinstance(data, dict) and isinstance(data.get('objects'), list) and data['objects']:
                                first = data['objects'][0]
                                if isinstance(first, dict):
                                    obj_name = first.get('object') or first.get('name')

                        if obj_name:
                            speech_text = f"I can see an {obj_name}."
                        else:
                            speech_text = "I can see it."
                    except Exception:
                        speech_text = "I can see it."
                result["speech_response"] = speech_text
            
                # 提取JSON数据（仅实验组/视觉 prompt_profile）：
                # 对照组（blind）必须禁用 objects/bboxes/image_size，避免"无视觉时编坐标"。
                if self.prompt_profile != "blind":
                    json_str = ""
                    if '```json' in reply:
                        json_str = reply.split("```json")[1].split("```")[0].strip()
                    elif '```' in reply:
                        parts = reply.split("```")
                        if len(parts) >= 2:
                            json_str = parts[1].strip()
                    
                    if json_str:
                        try:
                            # 回退式策略：先尝试原始 JSON 加载
                            data = json.loads(json_str)
                            
                            if debug_mode:
                                rclpy.logging.get_logger("doubao_api").info(f"豆包模型原始响应：\n{json.dumps(data, indent=2, ensure_ascii=False)}")

                            # 用户需求：不再把 JSON 额外保存到 vlm_raw（避免 json/raw 混在一起）。
                            # 仍然保持把结构化 JSON 打印到 stdout，便于 full_terminal_output.log 里查看。
                            try:
                                print("[doubao_api] 豆包模型原始响应：")
                                print(json.dumps(data, indent=2, ensure_ascii=False))
                            except Exception:
                                pass

                            # 处理物体数据
                            if "objects" in data:
                                processed_objects = []
                                for obj in data["objects"]:
                                    bboxes = obj.get("bboxes")
                                    
                                    # 情况1：单个物体的嵌套格式 [[x1,y1,x2,y2]] -> [x1,y1,x2,y2]
                                    if isinstance(bboxes, list) and len(bboxes) == 1 and isinstance(bboxes[0], list) and len(bboxes[0]) == 4:
                                        processed_objects.append({
                                            "name": obj.get("object", "未知物体"),
                                            "bboxes": bboxes[0],  # 展平
                                            "confidence": obj.get("confidence", 0.9)
                                        })
                                    # 情况2：多个物体实例 [[x1,y1,x2,y2], [x1,y1,x2,y2], ...] -> 创建多个物体对象
                                    elif isinstance(bboxes, list) and len(bboxes) > 1 and all(isinstance(box, list) and len(box) == 4 for box in bboxes):
                                        for box in bboxes:
                                            processed_objects.append({
                                                "name": obj.get("object", "未知物体"),
                                                "bboxes": box,
                                                "confidence": obj.get("confidence", 0.9)
                                            })
                                    # 情况3：平面格式 [x1,y1,x2,y2] -> 直接使用
                                    elif isinstance(bboxes, list) and len(bboxes) == 4 and all(isinstance(x, (int, float)) for x in bboxes):
                                        processed_objects.append({
                                            "name": obj.get("object", "未知物体"),
                                            "bboxes": bboxes,
                                            "confidence": obj.get("confidence", 0.9)
                                        })
                                
                                result["objects"] = processed_objects
                                self.last_detected_objects = result["objects"]

                            if "image_size" in data:
                                result["image_size"] = data["image_size"]
                                self.last_image_size = data["image_size"]
                        except json.JSONDecodeError as je:
                            # 第一次解析失败，尝试修复常见的 JSON 格式错误并重试（静默处理）
                            try:
                                import re
                                
                                # 修复函数1：修复数组中缺少逗号的问题
                                def fix_array_commas(match):
                                    """修复数组内缺少逗号的数字，例如: [1280 720] → [1280, 720]"""
                                    content = match.group(1).strip()
                                    numbers = re.findall(r'\d+(?:\.\d+)?', content)
                                    return '[' + ', '.join(numbers) + ']'
                                
                                # 修复函数2：处理 bboxes 和 confidence 混在一起的情况
                                def separate_bbox_confidence(match):
                                    """将 bboxes 中多余的第5个数字分离为 confidence"""
                                    content = match.group(1).strip()
                                    numbers = re.findall(r'\d+(?:\.\d+)?', content)
                                    if len(numbers) >= 4:
                                        coords = ', '.join(numbers[:4])
                                        if len(numbers) >= 5:
                                            confidence = numbers[4]
                                            return f'"bboxes": [{coords}], "confidence": {confidence}'
                                        else:
                                            return f'"bboxes": [{coords}]'
                                    return match.group(0)
                                
                                # 应用修复：修复所有数组格式
                                fixed_json_str = re.sub(r'\[([0-9\s.]+)\]', fix_array_commas, json_str)
                                # 应用修复：特殊处理 bboxes 字段
                                fixed_json_str = re.sub(
                                    r'"bboxes"\s*:\s*\[([^\]]+)\]',
                                    separate_bbox_confidence,
                                    fixed_json_str
                                )
                                
                                # 尝试用修复后的 JSON 重新解析
                                data = json.loads(fixed_json_str)
                                # 静默修复成功，不输出日志（避免终端噪音）
                                
                                if debug_mode:
                                    rclpy.logging.get_logger("doubao_api").info(f"修复后的 JSON:\n{json.dumps(data, indent=2, ensure_ascii=False)}")
                                
                                try:
                                    print("[doubao_api] 修复后的豆包模型响应：")
                                    print(json.dumps(data, indent=2, ensure_ascii=False))
                                except Exception:
                                    pass
                                
                                # 处理物体数据（修复成功后的逻辑）
                                if "objects" in data:
                                    processed_objects = []
                                    for obj in data["objects"]:
                                        bboxes = obj.get("bboxes")
                                        
                                        # 情况1：单个物体的嵌套格式 [[x1,y1,x2,y2]] -> [x1,y1,x2,y2]
                                        if isinstance(bboxes, list) and len(bboxes) == 1 and isinstance(bboxes[0], list) and len(bboxes[0]) == 4:
                                            processed_objects.append({
                                                "name": obj.get("object", "未知物体"),
                                                "bboxes": bboxes[0],  # 展平
                                                "confidence": obj.get("confidence", 0.9)
                                            })
                                        # 情况2：多个物体实例 [[x1,y1,x2,y2], [x1,y1,x2,y2], ...] -> 创建多个物体对象
                                        elif isinstance(bboxes, list) and len(bboxes) > 1 and all(isinstance(box, list) and len(box) == 4 for box in bboxes):
                                            for box in bboxes:
                                                processed_objects.append({
                                                    "name": obj.get("object", "未知物体"),
                                                    "bboxes": box,
                                                    "confidence": obj.get("confidence", 0.9)
                                                })
                                        # 情况3：平面格式 [x1,y1,x2,y2] -> 直接使用
                                        elif isinstance(bboxes, list) and len(bboxes) == 4 and all(isinstance(x, (int, float)) for x in bboxes):
                                            processed_objects.append({
                                                "name": obj.get("object", "未知物体"),
                                                "bboxes": bboxes,
                                                "confidence": obj.get("confidence", 0.9)
                                            })
                                    
                                    result["objects"] = processed_objects
                                    self.last_detected_objects = result["objects"]

                                if "image_size" in data:
                                    result["image_size"] = data["image_size"]
                                    self.last_image_size = data["image_size"]
                                    
                            except Exception as fix_error:
                                # 修复也失败了，记录错误
                                rclpy.logging.get_logger("doubao_api").warn(f"JSON 自动修复也失败: {str(fix_error)}")
                                print(f"[WARN] [doubao_api]: JSON 自动修复也失败: {str(fix_error)}")
                                rclpy.logging.get_logger("doubao_api").warn(f"⚠️ VLM 返回的 JSON 格式有误，原始内容:\n{json_str}")
                                print(f"[WARN] [doubao_api]: ⚠️ VLM 返回的 JSON 格式有误，原始内容:\n{json_str}")
                                if debug_mode:
                                    rclpy.logging.get_logger("doubao_api").info(f"完整响应:\n{reply}")
                                    print(f"[INFO] [doubao_api]: 完整响应:\n{reply}")

            except (IndexError, json.JSONDecodeError, KeyError) as e:
                rclpy.logging.get_logger("doubao_api").warn(f"解析豆包模型响应失败: {str(e)}")
                print(f"[WARN] [doubao_api]: 解析豆包模型响应失败: {str(e)}")
                if debug_mode:
                    rclpy.logging.get_logger("doubao_api").info(f"完整响应内容: {reply}")
                    print(f"[INFO] [doubao_api]: 完整响应内容: {reply}")
            except Exception as e:
                rclpy.logging.get_logger("doubao_api").error(f"处理豆包响应时出错: {str(e)}")
                print(f"[ERROR] [doubao_api]: 处理豆包响应时出错: {str(e)}")

            # 🔥 改进：维护N轮记忆（只存储Answer文本，不含JSON/bbox）
            if result.get("speech_response") and result["speech_response"] != "未知回复":
                # 只保存纯文本答案（speech_response 已经是清理后的答案，不含 JSON/Intent）
                self.conversation_history.extend([
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": result["speech_response"]}
                ])
                self.trim_conversation_history()
                
                # 调试：打印当前历史轮数
                current_turns = len(self.conversation_history) // 2
                print(f"📝 [对话历史] 当前保存 {current_turns} 轮（上限 {self.max_history_turns} 轮）")

            return result

        except Exception as e:
            rclpy.logging.get_logger("doubao_api").error(f"豆包API调用失败: {str(e)}")
            print(f"[ERROR] [doubao_api]: 豆包API调用失败: {str(e)}")
            traceback.print_exc()
            return {
                "speech_response": "豆包模型请求失败",
                "raw_response": str(e),
                "objects": [],
                "image_size": None
            }

    def get_object_position(self, object_name: str) -> Optional[List[float]]:
        """获取已记录物体的坐标（供其他模块调用）"""
        return self.object_coordinates.get(object_name)

    def update_object_positions(self, image_module: 'ImageSaveModule', debug: bool = False):
        """更新物体位置坐标（与原方法保持一致）"""
        if not getattr(self, 'last_detected_objects', None):
            if debug:
                rclpy.logging.get_logger("doubao_api").info("没有检测到任何物体")
            return

        valid_objects = [
            obj for obj in self.last_detected_objects
            if "bboxes" in obj and isinstance(obj["bboxes"], list) and len(obj["bboxes"]) == 4
        ]

        if debug:
            rclpy.logging.get_logger("doubao_api").info(f"开始处理{len(valid_objects)}个物体的坐标转换")

        # 图像实际分辨率
        if image_module.latest_image is not None:
            IMG_H, IMG_W = image_module.latest_image.shape[:2]
        else:
            IMG_H, IMG_W = 720, 1280  # 默认值

        image_size = getattr(self, "last_image_size", None)

        corrected_objects = []
        for obj in valid_objects:
            x1_orig, y1_orig, x2_orig, y2_orig = obj["bboxes"]

            if image_size and isinstance(image_size, list) and len(image_size) == 2:
                scale_x = IMG_W / 1000
                scale_y = IMG_H / 1000
                
                x1 = int(x1_orig * scale_x)
                y1 = int(y1_orig * scale_y)
                x2 = int(x2_orig * scale_x)
                y2 = int(y2_orig * scale_y)
                
                # 不再对 VLM 输出的 bbox 进行额外的 Y 方向偏移修正，保留原始比例换算。
            else:
                x1, y1, x2, y2 = x1_orig, y1_orig, x2_orig, y2_orig

            # 更新 bbox
            obj["bboxes"] = [x1, y1, x2, y2]
            # 计算中心点
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            obj["center"] = [cx, cy]

            corrected_objects.append(obj)

        centers = [obj["center"] for obj in corrected_objects]
        world_coords = image_module.get_3d_coordinates(centers, debug=debug)

        self.object_coordinates.clear()
        for obj, coord in zip(corrected_objects, world_coords):
            if coord:
                self.object_coordinates[obj["name"]] = coord
                if debug:
                    rclpy.logging.get_logger("doubao_api").info(f"成功转换 {obj['name']}: {coord}")
            elif debug:
                rclpy.logging.get_logger("doubao_api").warn(f"无法获取 {obj['name']} 的3D坐标")

        # 注释掉内部保存，避免重复（外部调用者会异步保存）
        # image_module.save_annotated_image(corrected_objects)
