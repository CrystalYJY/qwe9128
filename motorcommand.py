import rclpy
from rclpy.node import Node
from idmind_tabletop_msgs.msg import MotorCommand
import time

class MotorController(Node):
    def __init__(self):
        super().__init__('motor_command_example_node')

        # 初始化所有需要发布的topic
        self._motor_publishers = {
            'base': self.create_publisher(MotorCommand, '/idmind_tabletop/cmd_base_pos', 10),
            'neck_pitch': self.create_publisher(MotorCommand, '/idmind_tabletop/cmd_neck_pitch_pos', 10),
            'neck_roll': self.create_publisher(MotorCommand, '/idmind_tabletop/cmd_neck_roll_pos', 10),
            'left_eye_yaw': self.create_publisher(MotorCommand, '/idmind_tabletop/cmd_left_eye_yaw_pos', 10),
            'right_eye_yaw': self.create_publisher(MotorCommand, '/idmind_tabletop/cmd_right_eye_yaw_pos', 10),
            'left_eye_roll': self.create_publisher(MotorCommand, '/idmind_tabletop/cmd_left_eye_roll_pos', 10),
            'right_eye_roll': self.create_publisher(MotorCommand, '/idmind_tabletop/cmd_right_eye_roll_pos', 10),
            'left_eye_stroke': self.create_publisher(MotorCommand, '/idmind_tabletop/cmd_left_eye_stroke_pos', 10),
            'right_eye_stroke': self.create_publisher(MotorCommand, '/idmind_tabletop/cmd_right_eye_stroke_pos', 10),
        }

    def send_command(self, motor_name, position, play_time=200, relative=False, disable_eyes_roll_sync=False):
        """发送 MotorCommand 消息"""
        if motor_name not in self._motor_publishers:
            self.get_logger().error(f"未知的电机名称: {motor_name}")
            return

        msg = MotorCommand()
        msg.position = float(position)
        msg.play_time = play_time
        msg.relative = relative
        msg.disable_eyes_roll_sync = disable_eyes_roll_sync

        self._motor_publishers[motor_name].publish(msg)
        self.get_logger().info(f"发送 {motor_name} 指令: position={position}, play_time={play_time}, relative={relative}")

    def perform_motion_sequence(self):
        """连续动作示例：可以根据需要改动这个序列"""
        # 左右转头（基座）
        self.send_command('base', 20)
        time.sleep(1.0)
        self.send_command('base', -20)
        time.sleep(1.0)

        # 点头
        self.send_command('neck_pitch', 15)
        time.sleep(0.8)
        self.send_command('neck_pitch', -15)
        time.sleep(0.8)

        # 左右倾斜头
        self.send_command('neck_roll', 10)
        time.sleep(0.8)
        self.send_command('neck_roll', -10)
        time.sleep(0.8)

        # 眼睛左右移动
        self.send_command('left_eye_yaw', 10)
        self.send_command('right_eye_yaw', 10)
        time.sleep(0.5)
        self.send_command('left_eye_yaw', -10)
        self.send_command('right_eye_yaw', -10)
        time.sleep(0.5)

        # 眼睛旋转
        self.send_command('left_eye_roll', 5)
        self.send_command('right_eye_roll', -5)
        time.sleep(0.5)

        # 眨眼
        self.send_command('left_eye_stroke', 100)
        self.send_command('right_eye_stroke', 100)
        time.sleep(0.3)
        self.send_command('left_eye_stroke', 0)
        self.send_command('right_eye_stroke', 0)

def main(args=None):
    rclpy.init(args=args)
    controller = MotorController()
    time.sleep(1.0)  # 等待连接建立
    controller.perform_motion_sequence()
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
