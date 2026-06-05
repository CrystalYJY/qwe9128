import rclpy
from rclpy.node import Node
from idmind_tabletop_msgs.msg import MotorCommand
#from haru2_core_msgs.msg import MotorCommand
import time

class BaseTestPublisher(Node):
    def __init__(self):
        super().__init__('base_test_publisher')
        #self.publisher = self.create_publisher(MotorCommand, '/idmind_tabletop/cmd_base_pos', 10)
        self.publisher = self.create_publisher(MotorCommand, '/idmind_tabletop/cmd_base_pos', 10)
        self.get_logger().info("等待话题连接建立中...")
        time.sleep(2)  # 等待连接建立

    def send_command(self, position):
        msg = MotorCommand()
        msg.position = position
        msg.play_time = 200
        msg.relative = False
        msg.disable_eyes_roll_sync = False
        self.publisher.publish(msg)
        self.get_logger().info(f"[TEST] 发布 base 指令: position={position}")

def main():
    rclpy.init()
    node = BaseTestPublisher()

    node.send_command(0.5)  # 转右90度
    rclpy.spin_once(node, timeout_sec=1.0)
    time.sleep(3)

    node.send_command(0.0)  # 转回正前方
    rclpy.spin_once(node, timeout_sec=1.0)
    time.sleep(3)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
