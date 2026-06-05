#!/usr/bin/env python3
"""Head orientation sanity test.

用途
- 你做一组明确动作（向左/向右/抬头/低头/正视），脚本把 skeleton 中 nose/head 的 orientation
  转成 yaw/pitch/roll，并给出在各候选 forward axis (±X/±Y/±Z) 下的 gaze_dir。
- 用于快速判断 orientation 轴、镜像等问题。

运行
- 需要 ROS2 环境（订阅 /strawberry/azure/skeletons）。

备注
- 这是排障/对照脚本，不是主程序依赖。
"""

from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node

from strawberry_ros_msgs.msg import Skeletons


def _quat_to_rot_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-9:
        return np.eye(3)
    qx /= n
    qy /= n
    qz /= n
    qw /= n

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = 1 - 2 * (xx + yy)

    return np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], dtype=float)


def _rot_to_yaw_pitch_roll_zyx(R: np.ndarray) -> Tuple[float, float, float]:
    """返回 (yaw, pitch, roll) degrees using ZYX.

    约定：R = Rz(yaw) * Ry(pitch) * Rx(roll)
    """

    r20 = float(R[2, 0])
    r21 = float(R[2, 1])
    r22 = float(R[2, 2])
    r10 = float(R[1, 0])
    r00 = float(R[0, 0])

    pitch = math.asin(max(-1.0, min(1.0, -r20)))

    if abs(math.cos(pitch)) < 1e-6:
        yaw = 0.0
        roll = math.atan2(-float(R[0, 1]), float(R[1, 1]))
    else:
        yaw = math.atan2(r10, r00)
        roll = math.atan2(r21, r22)

    return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


@dataclass
class Sample:
    t: float
    yaw: float
    pitch: float
    roll: float
    gaze_by_axis: Dict[str, Tuple[float, float, float]]


class HeadOrientationSanityNode(Node):
    def __init__(self) -> None:
        super().__init__("head_orientation_sanity")
        self.sub = self.create_subscription(
            Skeletons, "/strawberry/azure/skeletons", self.on_msg, 10
        )
        self._last_print = 0.0

        out_dir = os.path.join(os.getcwd(), "tools", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, f"head_orientation_{int(time.time())}.csv")
        self._csv_f = open(self.csv_path, "w", newline="")
        self._csv = csv.writer(self._csv_f)
        self._csv.writerow(
            [
                "t",
                "yaw_deg",
                "pitch_deg",
                "roll_deg",
                "gaze_+X",
                "gaze_-X",
                "gaze_+Y",
                "gaze_-Y",
                "gaze_+Z",
                "gaze_-Z",
            ]
        )

        print(f"[tools] recording -> {self.csv_path}")

    def destroy_node(self):
        try:
            self._csv_f.close()
        except Exception:
            pass
        return super().destroy_node()

    def on_msg(self, msg: Skeletons) -> None:
        if not msg.skeletons:
            return

        sk = msg.skeletons[0]
        head = None
        for bp in sk.body_parts:
            if bp.name and bp.name.lower() == "head":
                head = bp
                break
        if head is None:
            return

        q = head.pose.orientation
        R = _quat_to_rot_matrix(q.x, q.y, q.z, q.w)
        yaw, pitch, roll = _rot_to_yaw_pitch_roll_zyx(R)

        cols = {
            "+X": R[:, 0],
            "-X": -R[:, 0],
            "+Y": R[:, 1],
            "-Y": -R[:, 1],
            "+Z": R[:, 2],
            "-Z": -R[:, 2],
        }
        gaze_by_axis: Dict[str, Tuple[float, float, float]] = {}
        for k, v in cols.items():
            u = _unit(np.asarray(v, dtype=float))
            gaze_by_axis[k] = (
                (float("nan"), float("nan"), float("nan"))
                if u is None
                else (float(u[0]), float(u[1]), float(u[2]))
            )

        t = time.time()
        self._csv.writerow(
            [
                f"{t:.3f}",
                f"{yaw:.2f}",
                f"{pitch:.2f}",
                f"{roll:.2f}",
                gaze_by_axis["+X"],
                gaze_by_axis["-X"],
                gaze_by_axis["+Y"],
                gaze_by_axis["-Y"],
                gaze_by_axis["+Z"],
                gaze_by_axis["-Z"],
            ]
        )
        self._csv_f.flush()

        if (t - self._last_print) >= 0.5:
            self._last_print = t
            print(
                "[tools] yaw={:.1f} pitch={:.1f} roll={:.1f} gaze+X={} gaze+Z={}".format(
                    yaw,
                    pitch,
                    roll,
                    tuple(round(x, 3) for x in gaze_by_axis["+X"]),
                    tuple(round(x, 3) for x in gaze_by_axis["+Z"]),
                )
            )


def main() -> None:
    rclpy.init()
    node = HeadOrientationSanityNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
