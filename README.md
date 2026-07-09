# Semantic-Driven Social Robot Interaction with VLMs

Code release for the paper:

**Semantic-Driven Social Robot Interaction with VLMs: Embodied Feedback for Joint Attention and Intent Transparency**

Accepted by **RO-MAN 2026**.

This repository contains a research prototype for semantic-driven social robot interaction. The system uses a vision-language model (VLM), speech recognition, robot gaze control, and embodied feedback to help a tabletop social robot communicate attention, intent, and task-relevant object references during human-robot interaction.

## Overview

The project explores how a robot can make its internal understanding more transparent during interaction. Given user speech and visual context from the robot environment, the system can:

- interpret user intent with a VLM;
- identify task-relevant objects from camera input;
- convert detected object positions into robot-centered coordinates;
- control robot gaze and head movement toward people or objects;
- provide spoken responses through the robot interface;
- log conversation and interaction data for later analysis;
- compare interaction conditions such as vision-enabled, no-gaze, and text-only settings.

The prototype was developed for embodied interaction experiments with a physical robot. It is released as research code and may require project-specific robot hardware, ROS 2 topics, message definitions, and API credentials to run end to end.

## Demo Video

A physical robot interaction demo is available here:

**Video:** TODO: add demo video link

The video shows the robot interacting with a user in a real embodied setting, including speech input, VLM-based interpretation, robot feedback, and gaze behavior.

## Repository Structure

```text
.
├── 2.py                         # Main vision-enabled interaction condition
├── no_gaze.py                   # Control condition with no object-directed gaze
├── speak.py                     # Text-only / blind control condition
├── doubao_api_module.py         # VLM interface and object grounding logic
├── xunfei_asr_adapter.py        # ASR adapter
├── xunfei_asr_streaming.py      # Streaming ASR implementation
├── image_save_module.py         # Camera frame capture and image utilities
├── look_controller.py           # Head tracking and gaze control
├── robot_env.py                 # Robot speech and motor command wrapper
├── conversation_logger.py       # Interaction logging utilities
├── analyse/                     # Analysis scripts and experiment result processing
├── tools/                       # Debugging and calibration utilities
└── test_*.py                    # Local tests and hardware debugging scripts
```

## Interaction Conditions

The repository includes several experimental conditions:

| File | Condition | Description |
| --- | --- | --- |
| `2.py` | Vision + gaze | Uses visual context and object-directed robot feedback. |
| `no_gaze.py` | Vision without gaze | Uses visual/VLM information but suppresses object-directed gaze behavior. |
| `speak.py` | Text-only / blind control | Uses speech interaction without visual grounding. |

These conditions are useful for comparing how embodied feedback affects joint attention, user understanding, and perceived robot intent transparency.

## Main Components

- **VLM reasoning:** `doubao_api_module.py` sends image and text input to a VLM and parses natural-language responses plus object bounding boxes.
- **Speech input:** `xunfei_asr_adapter.py` and `xunfei_asr_streaming.py` provide streaming ASR integration.
- **Robot output:** `robot_env.py`, `look_controller.py`, and `motorcommand.py` handle speech and motor commands.
- **Image processing:** `image_save_module.py` captures RGB/depth frames and supports coordinate conversion.
- **Logging:** `conversation_logger.py` records interaction turns, model outputs, timing, and task-level information.
- **Analysis:** scripts in `analyse/` process experiment logs, questionnaires, timing metrics, task success, and case analyses.

## Requirements

This code was developed for a ROS 2 based social robot setup. A full run may require:

- Python 3.10+;
- ROS 2 with `rclpy`;
- robot-specific ROS message packages, including `idmind_tabletop_msgs` and `strawberry_ros_msgs`;
- Azure Kinect or compatible RGB-D camera topics;
- OpenCV, NumPy, Pillow, HTTPX, OpenAI Python SDK, websocket-client;
- ASR and VLM service credentials configured locally.

Example Python dependencies:

```bash
pip install openai httpx websocket-client opencv-python numpy pillow pandas matplotlib seaborn scipy scikit-posthocs scikit-image
```

ROS 2 and robot-specific packages need to be installed in the robot workspace rather than through `pip`.

## Configuration

API credentials should be provided through local configuration or environment variables. Do not commit real API keys to the repository.

Suggested environment variables:

```bash
export DOUBAO_API_KEY="your-doubao-api-key"
export XUNFEI_APPID="your-xunfei-appid"
export XUNFEI_API_KEY="your-xunfei-api-key"
export XUNFEI_API_SECRET="your-xunfei-api-secret"
```

If you adapt the code for another lab or robot platform, update the ROS topic names, microphone device names, and log output paths in the corresponding runtime scripts.

## Running

Run the main vision-enabled condition:

```bash
python 2.py
```

Run the no-gaze control condition:

```bash
python no_gaze.py
```

Run the text-only / blind control condition:

```bash
python speak.py
```

Before running, make sure the robot, ROS 2 nodes, camera topics, microphone, ASR service, VLM service, and TTS/motor interfaces are available.

## Analysis

The `analyse/` directory contains scripts used to process experiment data, including:

- user gap and timing analysis;
- questionnaire analysis;
- task success analysis;
- conversation turn and duration analysis;
- case-level analysis.

These scripts may expect local CSV or JSON files produced during experiments. Paths may need to be adjusted before reuse.

## Notes

- This is research prototype code, not a packaged software library.
- Some scripts contain local hardware paths, microphone device names, and ROS topic assumptions.
- The repository is intended to document the core functionality and experimental logic behind the accepted paper.
- For reproducibility, sensitive credentials and private participant data should be excluded from the public repository.

## Citation

If you use this code or build on this work, please cite:

```bibtex
@inproceedings{yu2026semantic,
  title = {Semantic-Driven Social Robot Interaction with VLMs: Embodied Feedback for Joint Attention and Intent Transparency},
  author = {Yu, Jinyao},
  booktitle = {Proceedings of RO-MAN 2026},
  year = {2026}
}
```

Please update the BibTeX entry with the final author list, DOI, page numbers, and publisher information once the proceedings version is available.
