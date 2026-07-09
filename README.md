# Semantic-Driven Social Robot Interaction with VLMs

本仓库是论文 **Semantic-Driven Social Robot Interaction with VLMs: Embodied Feedback for Joint Attention and Intent Transparency** 的项目代码整理版本。

该论文已被 **RO-MAN 2026** 接收。

本项目探索如何将视觉语言模型（VLM）、语音识别、机器人注视控制和具身反馈结合起来，让社交机器人在与人交互时更清楚地表达“它理解了什么”“它正在关注哪里”以及“它接下来想做什么”。项目重点不是做一个通用软件包，而是支撑论文中的实体机器人交互实验和功能演示。

## 项目简介

在交互过程中，系统会结合用户语音和机器人视角下的视觉信息，理解用户意图、识别相关物体，并通过机器人的头部转向、注视行为和语音回复，把内部语义理解外显出来。

项目主要包含以下能力：

- 使用 VLM 理解用户语音和当前视觉场景；
- 识别用户提到的任务相关物体；
- 将图像中的物体位置转换为机器人坐标系下的位置；
- 控制机器人看向用户或目标物体；
- 通过语音回复用户；
- 记录交互过程、模型输出、时间信息和实验数据；
- 支持不同实验条件的对比，例如有视觉和注视反馈、无注视反馈、纯语音对照等。

## 演示视频

实体机器人交互演示视频：https://youtu.be/5RitVIL6RVA?si=vVh2BY2ncgeWqaHH

**视频链接：待补充**

视频展示了用户与实体机器人进行自然交互的过程，包括语音输入、VLM 语义理解、机器人语音反馈和注视行为。

## 代码结构

```text
.
├── 2.py                         # 主要实验条件：视觉理解 + 机器人注视反馈
├── no_gaze.py                   # 对照条件：使用视觉理解，但不执行目标物体注视
├── speak.py                     # 对照条件：纯语音 / blind prompt，不使用视觉信息
├── doubao_api_module.py         # VLM 调用、回复解析和物体定位逻辑
├── xunfei_asr_adapter.py        # 讯飞 ASR 适配层
├── xunfei_asr_streaming.py      # 讯飞流式语音识别实现
├── image_save_module.py         # 相机图像获取、保存和坐标转换相关工具
├── look_controller.py           # 机器人头部跟踪和注视控制
├── robot_env.py                 # 机器人语音和运动指令封装
├── conversation_logger.py       # 对话和实验日志记录
├── analyse/                     # 实验数据分析脚本
├── tools/                       # 调试、坐标检查和离线验证工具
└── test_*.py                    # 本地硬件测试和调试脚本
```

## 实验条件

仓库中保留了几个主要实验 / 对照条件：

| 文件 | 条件 | 说明 |
| --- | --- | --- |
| `2.py` | 视觉 + 注视反馈 | 使用视觉场景和 VLM 识别结果，并让机器人看向相关目标物体。 |
| `no_gaze.py` | 视觉但无目标注视 | 仍使用 VLM 和视觉信息，但抑制机器人看向目标物体的动作。 |
| `speak.py` | 纯语音 / blind control | 不依赖视觉信息，仅通过语音进行交互。 |

这些条件用于比较具身反馈对共同注意（joint attention）、用户理解和机器人意图透明度的影响。

## 核心模块

- `doubao_api_module.py`：负责调用 VLM，发送图像和文本输入，并解析自然语言回复和物体框。
- `xunfei_asr_adapter.py` / `xunfei_asr_streaming.py`：负责语音识别输入。
- `robot_env.py` / `look_controller.py` / `motorcommand.py`：负责机器人语音输出、头部跟踪和运动控制。
- `image_save_module.py`：负责读取相机图像、深度信息和相机参数，并支持图像坐标到空间坐标的转换。
- `conversation_logger.py`：负责保存对话轮次、模型原始输出、任务信息和时间信息。
- `analyse/`：用于处理实验日志、问卷、任务成功率、对话时长和案例分析。

## 运行环境

本项目依赖实体机器人和 ROS 2 环境。完整运行通常需要：

- Python 3.10 或以上；
- ROS 2 和 `rclpy`；
- 机器人相关 ROS 消息包，例如 `idmind_tabletop_msgs`、`strawberry_ros_msgs`；
- Azure Kinect 或兼容 RGB-D 相机；
- OpenCV、NumPy、Pillow、HTTPX、OpenAI Python SDK、websocket-client；
- 本地配置好的 ASR 和 VLM 服务凭据；
- 可用的机器人 TTS、头部运动和相机话题。

可参考安装的 Python 依赖：

```bash
pip install openai httpx websocket-client opencv-python numpy pillow pandas matplotlib seaborn scipy scikit-posthocs scikit-image
```

ROS 2、机器人消息包和硬件驱动需要在机器人工作空间中单独配置，不能只通过 `pip` 安装完成。

## 配置说明

API Key 和实验环境参数建议放在本地配置或环境变量中，不应提交到公开仓库。

可以使用类似方式配置：

```bash
export DOUBAO_API_KEY="your-doubao-api-key"
export XUNFEI_APPID="your-xunfei-appid"
export XUNFEI_API_KEY="your-xunfei-api-key"
export XUNFEI_API_SECRET="your-xunfei-api-secret"
```

如果在其他机器人平台或实验环境中复现，需要根据实际情况修改：

- ROS topic 名称；
- 麦克风设备编号；
- 相机话题和相机参数；
- 机器人 TTS 和运动控制接口；
- 日志保存路径。

## 运行方式

运行视觉 + 注视反馈条件：

```bash
python 2.py
```

运行无目标注视对照条件：

```bash
python no_gaze.py
```

运行纯语音 / blind control 条件：

```bash
python speak.py
```

运行前需要确认机器人、ROS 2 节点、相机、麦克风、ASR 服务、VLM 服务、TTS 和运动控制接口都已经启动。

## 数据分析

`analyse/` 目录中包含实验数据分析脚本，主要用于处理：

- 用户反应间隔；
- 人机对话时长；
- 机器人反馈时长；
- 问卷结果；
- 任务成功率；
- 对话轮次；
- 案例分析。

这些脚本通常依赖实验过程中生成的本地 CSV 或 JSON 文件。复用时可能需要根据自己的数据路径进行调整。

## 注意事项

- 本仓库是研究原型代码，不是通用软件库。
- 部分脚本包含特定机器人、相机、麦克风和 ROS topic 的假设。
- 代码主要用于说明论文中的系统实现、实验条件和数据处理流程。
- 公开仓库中不应包含真实 API Key、私人实验数据或参与者隐私信息。

## Citation

如果你使用或参考本项目，请引用：

```bibtex
@inproceedings{yu2026semantic,
  title = {Semantic-Driven Social Robot Interaction with VLMs: Embodied Feedback for Joint Attention and Intent Transparency},
  author = {Yu, Jinyao},
  booktitle = {Proceedings of RO-MAN 2026},
  year = {2026}
}
```

论文正式出版后，请根据最终论文信息补充完整作者列表、DOI、页码和出版方信息。
