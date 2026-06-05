#!/usr/bin/env python3
"""Inspect skeleton header.frame_id and a few keypoint positions once.

Goal
- 确认 skeleton 关键点 position 使用的 frame_id（如果消息里带 header）。
- 快速对照 head/nose 的 x/y/z 符号与范围，判断与 object 3D 坐标是否同一口径。

Usage
- 需要 ROS2 环境，能订阅 /strawberry/azure/skeletons
- 打印一次后退出

注意
- 这是排障工具脚本，不参与主程序运行。
"""

from __future__ import annotations

import time

import rclpy
from rclpy.node import Node

from strawberry_ros_msgs.msg import Skeletons


def _get_part(skel, name: str):
    for p in skel.body_parts:
        if p.name.lower() == name:
            return p
    return None


class Once(Node):
    def __init__(self):
        super().__init__("inspect_skeleton_frame_once")
        self.sub = self.create_subscription(
            Skeletons,
            "/strawberry/azure/skeletons",
            self.cb,
            10,
        )
        self._done = False

    def cb(self, msg: Skeletons):
        if self._done:
            return
        if not msg.skeletons:
            return

        skel = msg.skeletons[0]
        head = _get_part(skel, "head")
        nose = _get_part(skel, "nose")

        frame_id = None
        stamp = None
        if hasattr(msg, "header"):
            frame_id = getattr(msg.header, "frame_id", None)
            stamp = getattr(msg.header, "stamp", None)

        print("=== /strawberry/azure/skeletons (first message) ===")
        print(f"frame_id: {frame_id}")
        if stamp is not None:
            print(f"stamp: {stamp.sec}.{stamp.nanosec}")

        if head is not None:
            hp = head.pose.position
            print(f"head.pos: x={hp.x:+.3f} y={hp.y:+.3f} z={hp.z:+.3f}")
        else:
            print("head.pos: <missing>")

        if nose is not None:
            np = nose.pose.position
            print(f"nose.pos: x={np.x:+.3f} y={np.y:+.3f} z={np.z:+.3f}")
        else:
            print("nose.pos: <missing>")

        parts = [bp.name for bp in skel.body_parts]
        print(f"body_parts: {parts}")

        self._done = True
        self.get_logger().info("Done. Shutting down...")
        time.sleep(0.1)
        rclpy.shutdown()


def main():
    rclpy.init()
    node = Once()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
