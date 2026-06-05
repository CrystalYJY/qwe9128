"""Offline sanity check for JA geometry.

用途
- 把运行时日志里的一组 head/nose/obj/raw_gaze 数字复制到 SAMPLE。
- 这脚本不依赖 ROS，只是离线重算多种候选变换下的 dot/angle，帮助定位坐标口径问题。

用法
- 用项目 venv python 运行即可。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


# -------------------------
# Paste one log sample here
# -------------------------
SAMPLE = {
    "head_pos": [0.986, -0.313, 0.043],
    "nose_pos": [0.841, -0.304, 0.079],
    "obj_pos": [-0.418, 0.288, 0.95],
    "raw_gaze": [-0.995, 0.049, -0.086],
}


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v * 0.0
    return v / n


def angle_deg(dot: float) -> float:
    dot = float(np.clip(dot, -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def opencv_to_azure(p: np.ndarray) -> np.ndarray:
    """历史上常见的一种轴映射（仅用于诊断）。

OpenCV camera: x right, y down, z forward
Azure (common robotics): x forward, y left, z up

Mapping:
  x_az = z_cv
  y_az = -x_cv
  z_az = -y_cv
"""

    x, y, z = p.tolist()
    return np.array([z, -x, -y], dtype=float)


def mirror_y(v: np.ndarray) -> np.ndarray:
    out = v.copy()
    out[1] *= -1.0
    return out


@dataclass
class Case:
    name: str
    gaze: np.ndarray
    target: np.ndarray


def build_cases(head: np.ndarray, nose: np.ndarray, obj: np.ndarray, gaze: np.ndarray) -> List[Case]:
    origin = nose
    target = unit(obj - origin)

    cases: List[Case] = []
    cases.append(Case("baseline(raw_gaze vs nose->obj)", unit(gaze), target))
    cases.append(Case("mirror_target_y", unit(gaze), unit(mirror_y(target))))
    cases.append(Case("mirror_gaze_y", unit(mirror_y(gaze)), target))
    cases.append(Case("mirror_both_y", unit(mirror_y(gaze)), unit(mirror_y(target))))

    obj_az = opencv_to_azure(obj)
    cases.append(Case("obj_opencv_to_azure", unit(gaze), unit(obj_az - origin)))

    head_az = opencv_to_azure(head)
    nose_az = opencv_to_azure(nose)
    gaze_az = opencv_to_azure(gaze)  # 方向这么映射不一定物理正确，仅用来对照诊断
    cases.append(Case("all_opencv_to_azure(rough)", unit(gaze_az), unit(obj_az - nose_az)))

    return cases


def main() -> None:
    head = np.array(SAMPLE["head_pos"], dtype=float)
    nose = np.array(SAMPLE["nose_pos"], dtype=float)
    obj = np.array(SAMPLE["obj_pos"], dtype=float)
    gaze = np.array(SAMPLE["raw_gaze"], dtype=float)

    print("Input sample:")
    print("  head_pos:", head)
    print("  nose_pos:", nose)
    print("  obj_pos :", obj)
    print("  raw_gaze:", gaze)

    base_target = unit(obj - nose)
    print("\nBaseline target_dir (nose->obj):", np.round(base_target, 6))

    cases = build_cases(head, nose, obj, gaze)
    print("\nCandidates:")
    for c in cases:
        d = float(np.dot(unit(c.gaze), unit(c.target)))
        print(f"  {c.name:28s} dot={d:+.3f} angle={angle_deg(d):5.1f}°")


if __name__ == "__main__":
    main()
