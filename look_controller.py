import time
import math
import threading
import rclpy
from rclpy.node import Node
from strawberry_ros_msgs.msg import Skeletons

class LookController:
    def __init__(self, motor_controller, node: Node, on_skeleton_cb=None):
        self.motor_controller = motor_controller
        self.node = node
        self.skeleton_sub = None
        self.on_skeleton_cb = on_skeleton_cb # 外部回调 hook
        self.current_mode = "object"  # 默认模式：object 或 human
        self.object_target = None  # 存储当前注视的物体坐标
        self.object_gaze_timer = None  # 用于持续注视物体的定时器
        self.node.get_logger().info("✅ LookController 已初始化")
        
        self.skeleton_sub = self.node.create_subscription(
            Skeletons,
            '/strawberry/azure/skeletons',
            self.skeleton_callback,
            10
        )
        self.node.get_logger().info("   [SUB] ✅ 骨骼话题订阅已创建并永久保持")

    def set_mode(self, mode: str):
        """设置追踪模式
        Args:
            mode: "human" 用于人体追踪, "object" 用于物体观察
        """
        if mode in ["human", "object"]:
            self.current_mode = mode
            self.node.get_logger().info(f"切换到 {mode} 模式")
        else:
            self.node.get_logger().warn(f"未知模式: {mode}, 保持当前模式: {self.current_mode}")

    def look_at(self, point_3d, mode=None):
        """看向指定3D坐标点
        Args:
            point_3d: [x, y, z] 坐标
            mode: 可选，指定此次调用的模式，如果不指定则使用当前设置的模式
        """
        import time
        x, y, z = point_3d
        
        # 使用指定模式或当前模式
        current_mode = mode if mode is not None else self.current_mode
        
        if current_mode == "human":
            # head 坐标系：用于人体追踪
            yaw = math.atan2(-y, x)         # 左右偏转
            pitch = -math.atan2(z, math.sqrt(y ** 2 + x ** 2))  # 上下抬头
            #self.motor_controller.get_logger().info(
            #    f"[look_at] HUMAN模式 target=({x:.2f},{y:.2f},{z:.2f}) => yaw={yaw:.2f}, pitch={pitch:.2f}"
            #)
        else:  # object 模式
            # haru坐标系：用于物体观察
            yaw = math.atan2(x, z)
            pitch = math.atan2(y, math.sqrt(x**2 + z**2))
            #self.motor_controller.get_logger().info(
            #    f"[look_at] OBJECT模式 target=({x:.2f},{y:.2f},{z:.2f}) => yaw={yaw:.2f}, pitch={pitch:.2f}"
            #)

        # yaw控制分配
        if abs(yaw) <= 0.5:
            base_yaw = 0.0
            eye_yaw = max(min(yaw, 0.5), -0.5)
        else:
            base_yaw = max(min(yaw, 1.57), -1.57)
            eye_yaw = 0.0

        neck_pitch = max(min(pitch, 0.5), -0.5)

        # 实际发送命令
        self.motor_controller.send_command('base', base_yaw)
        time.sleep(0.5)

        self.motor_controller.send_command('neck_pitch', neck_pitch)
        time.sleep(0.3)

        self.motor_controller.send_command('left_eye_yaw', eye_yaw)
        self.motor_controller.send_command('right_eye_yaw', eye_yaw)
        time.sleep(0.3)



    def start_head_tracking(self):
        """开始/恢复人体头部追踪"""
        # 停止物体注视定时器
        if self.object_gaze_timer:
            self.object_gaze_timer.cancel()
            self.object_gaze_timer = None
        self.object_target = None
        self.set_mode("human")

    def stop_head_tracking(self):
        """停止人体头部追踪（用于观察物体或执行表情动作）"""
        self.set_mode("object")

    def _maintain_object_gaze(self):
        """内部函数：持续保持注视物体"""
        if self.current_mode == "object" and self.object_target is not None:
            # 重新发送注视指令（不带延时，避免阻塞）
            x, y, z = self.object_target
            yaw = math.atan2(x, z)
            pitch = math.atan2(y, math.sqrt(x**2 + z**2))
            
            if abs(yaw) <= 0.5:
                base_yaw = 0.0
                eye_yaw = max(min(yaw, 0.5), -0.5)
            else:
                base_yaw = max(min(yaw, 1.57), -1.57)
                eye_yaw = 0.0
            
            neck_pitch = max(min(pitch, 0.5), -0.5)
            
            # 快速发送命令（无延时）
            self.motor_controller.send_command('base', base_yaw)
            self.motor_controller.send_command('neck_pitch', neck_pitch)
            self.motor_controller.send_command('left_eye_yaw', eye_yaw)
            self.motor_controller.send_command('right_eye_yaw', eye_yaw)
            
            # 每2秒重新发送一次，保持姿态
            self.object_gaze_timer = threading.Timer(2.0, self._maintain_object_gaze)
            self.object_gaze_timer.start()

    def look_at_object(self, point_3d):
        """看向物体（使用物体观察模式），并持续保持注视"""
        # 停止旧的注视定时器
        if self.object_gaze_timer:
            self.object_gaze_timer.cancel()
        
        self.set_mode("object")
        self.object_target = point_3d  # 保存目标坐标
        self.look_at(point_3d, mode="object")
        
        # 启动持续注视定时器（2秒后重新发送一次指令）
        self.object_gaze_timer = threading.Timer(2.0, self._maintain_object_gaze)
        self.object_gaze_timer.start()
        self.node.get_logger().info(f"🎯 开始持续注视物体: {point_3d}")

    def skeleton_callback(self, msg: Skeletons):
        """处理骨骼数据的回调函数"""
        # 1. 优先调用外部 hook（无论模式如何都需要 feed 给 JA）
        if self.on_skeleton_cb:
            try:
                self.on_skeleton_cb(msg)
            except Exception as e:
                self.node.get_logger().error(f"External skeleton callback error: {e}")

        # 关键检查：如果当前不是人体追踪模式，则忽略消息
        if self.current_mode != "human":
            return

        if self.node.DEBUG_MODE if hasattr(self.node, 'DEBUG_MODE') else False:
            self.node.get_logger().debug("💀 [CALLBACK] 收到骨骼消息!")
        
        if not msg.skeletons:
            self.node.get_logger().warn("   [DATA] 消息中没有骨骼数据 (skeletons 列表为空)")
            return

        # 只追踪检测到的第一个人
        target_skeleton = msg.skeletons[0]
        head_part = None
        for part in target_skeleton.body_parts:
            if part.name == "head":
                head_part = part
                break
        
        if head_part:
            pos = head_part.pose.position
            x, y, z = pos.x, pos.y, pos.z
        #    self.node.get_logger().info(f"   [JOINT] 检测到头部坐标: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
            # 使用人体追踪模式调用底层look_at函数
            self.look_at([x, y, z], mode="human")
        else:
            self.node.get_logger().warn("   [JOINT] 在第一个骨骼中未找到 'head' 部位")