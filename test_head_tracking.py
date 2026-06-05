import rclpy
from rclpy.node import Node
from look_controller import MotorController, LookController

def main():
    rclpy.init()
    node = rclpy.create_node('test_head_tracking_node')
    motor_controller = MotorController()
    look_controller = LookController(motor_controller, node)
    look_controller.start_head_tracking('/strawberry/azure/skeletons')
    node.get_logger().info('开始自动追踪头部...')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    motor_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
