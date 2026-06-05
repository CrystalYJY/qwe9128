# test_look.py
import rclpy
import time
from look_controller import MotorController, LookController

def main():
    rclpy.init()
    node = rclpy.create_node('test_look_node')
    motor_controller = MotorController()
    look_controller = LookController(motor_controller,node)

    motor_controller.get_logger().info("初始化 MotorController 和 LookController...")

    # ✅ 你可以随便改这个目标点，看看机器人会朝哪里看
    target_point = [-0.46,0.43,1.22]  # x, y, z（单位米，来自相机坐标系）
    print(f"[TEST] 正在look_at点: {target_point}")
    look_controller.look_at(target_point)

    motor_controller.get_logger().info("保持程序运行 5 秒钟观察反应...")
    rclpy.spin(motor_controller)
    motor_controller.get_logger().info("程序结束，准备关闭节点")
    node.destroy_node()
    motor_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
