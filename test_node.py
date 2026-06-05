#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

class ImageListener(Node):
    def __init__(self):
        super().__init__('test_rgb_image_raw_node')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/azure_kinect/rgb/image_raw/compressed',
            self.image_callback,
            1
        )
        self.get_logger().info('已订阅 /azure_kinect/rgb/image_raw/compressed，等待消息...')

    def image_callback(self, msg):
        import numpy as np
        import cv2
        self.get_logger().info(f'收到一帧图像，时间戳: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}')
        # 解码压缩图像
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is not None:
            cv2.imwrite('/tmp/ros2_camera_latest.jpg', image)
            self.get_logger().info('已保存当前图片到 /tmp/ros2_camera_latest.jpg')
        else:
            self.get_logger().warn('图像解码失败')

def main():
    rclpy.init()
    node = ImageListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
