#!/usr/bin/env python3
"""订阅一次 /strawberry/azure/skeletons，打印 body_parts/pose/orientation 并退出。

用途
- 定位 RViz 里“脸朝向箭头”到底对应哪个 body_part / 哪个 orientation。
- 快速确认 skeleton 消息里有哪些关键点（head/nose/eyes/...）。

注意
- 这是排障工具脚本，不参与主程序运行。
"""

import time

import rclpy
from rclpy.node import Node

from strawberry_ros_msgs.msg import Skeletons


def _quat_to_rot_matrix(x: float, y: float, z: float, w: float):
    # 标准四元数转旋转矩阵（右手系）
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n < 1e-12:
        return None
    x /= n
    y /= n
    z /= n
    w /= n

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return [
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ]


def _mat_vec(m, v):
    return [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]


def _norm(v):
    return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5


def _unit(v):
    n = _norm(v)
    if n < 1e-12:
        return None
    return [v[0] / n, v[1] / n, v[2] / n]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


class InspectSkeletonOnce(Node):
    """订阅一次 skeletons，打印关键信息并退出。"""

    def __init__(self):
        super().__init__("inspect_skeleton_once")
        self._sub = self.create_subscription(
            Skeletons, "/strawberry/azure/skeletons", self._cb, 10
        )
        self._done = False
        self._t0 = time.time()

    def _cb(self, msg: Skeletons):
        if self._done:
            return
        self._done = True

        print("\n=== inspect_skeleton_once ===")
        print(f"received skeletons: {len(msg.skeletons)}")

        if not msg.skeletons:
            print("(no skeletons in message)")
            rclpy.shutdown()
            return

        sk = msg.skeletons[0]
        parts = getattr(sk, "body_parts", [])
        print(f"body_parts count: {len(parts)}")

        # 打印每个 part 的 name
        names = [getattr(p, "name", "") for p in parts]
        print("body_parts names:")
        print(names)

        def _get_part(name_lc: str):
            for p in parts:
                if getattr(p, "name", "").lower() == name_lc:
                    return p
            return None

        def _pinfo(p, label: str):
            if p is None:
                print(f"- {label}: <None>")
                return
            pose = getattr(p, "pose", None)
            if pose is None:
                print(f"- {label}: pose=<None>")
                return
            pos = getattr(pose, "position", None)
            ori = getattr(pose, "orientation", None)

            if pos is None:
                pos_s = "pos=<None>"
            else:
                pos_s = f"pos=({pos.x:.4f},{pos.y:.4f},{pos.z:.4f})"

            if ori is None:
                ori_s = "ori=<None>"
            else:
                n = (ori.x * ori.x + ori.y * ori.y + ori.z * ori.z + ori.w * ori.w) ** 0.5
                ori_s = f"ori=({ori.x:.4f},{ori.y:.4f},{ori.z:.4f},{ori.w:.4f}) | norm={n:.4f}"

            print(f"- {label}: {pos_s} | {ori_s}")

        _pinfo(_get_part("head"), "head")
        _pinfo(_get_part("nose"), "nose")
        _pinfo(_get_part("eye_left"), "eye_left")
        _pinfo(_get_part("eye_right"), "eye_right")

        # 自动推断 forward 轴：用 head->nose 的几何方向作为参考
        head = _get_part("head")
        nose = _get_part("nose")
        if head is not None and nose is not None:
            hp = head.pose.position
            np = nose.pose.position
            ref = _unit([np.x - hp.x, np.y - hp.y, np.z - hp.z])
            if ref is not None:
                q = nose.pose.orientation
                rm = _quat_to_rot_matrix(q.x, q.y, q.z, q.w)
                if rm is not None:
                    candidates = [
                        ("+X", [1.0, 0.0, 0.0]),
                        ("-X", [-1.0, 0.0, 0.0]),
                        ("+Y", [0.0, 1.0, 0.0]),
                        ("-Y", [0.0, -1.0, 0.0]),
                        ("+Z", [0.0, 0.0, 1.0]),
                        ("-Z", [0.0, 0.0, -1.0]),
                    ]
                    best = None
                    for name, axis in candidates:
                        v = _mat_vec(rm, axis)
                        vu = _unit(v)
                        if vu is None:
                            continue
                        s = _dot(vu, ref)
                        if best is None or s > best[1]:
                            best = (name, s, vu)
                    if best is not None:
                        name, score, vec = best
                        print(
                            f"best_forward_axis (vs head->nose) = {name} | dot={score:.4f} | world_vec=({vec[0]:.3f},{vec[1]:.3f},{vec[2]:.3f})"
                        )
                        print(f"ref(head->nose) = ({ref[0]:.3f},{ref[1]:.3f},{ref[2]:.3f})")

        dt = time.time() - self._t0
        print(f"done in {dt:.2f}s; shutting down")
        rclpy.shutdown()


def main():
    rclpy.init()
    node = InspectSkeletonOnce()
    try:
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=1.0)
        if not node._done:
            print("timeout: no skeleton message received")
            rclpy.shutdown()
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass


if __name__ == "__main__":
    main()
