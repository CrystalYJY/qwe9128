import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
#from haru2_core_msgs.msg import TTSCommand, MotorCommand, TTSStatus
from idmind_tabletop_msgs.msg import TTSCommand, MotorCommand, TTSStatus
import time

class TTSSpeaker(Node):
    def __init__(self):
        super().__init__('tts_command_node')
        self.publisher = self.create_publisher(TTSCommand, '/idmind_tabletop/cmd_tts', 10)
        self.subscriber = self.create_subscription(TTSStatus, '/idmind_tabletop/tts_status', self.tts_status_callback, 10)
        self.tts_finished = True
        self._last_status = None

    def tts_status_callback(self, msg):
        """处理TTS状态回调"""
        # 状态去重，防止刷屏（仅针对控制台日志，不拦截消息）
        if self._last_status == msg.status:
            return
        
        # 调试：打印每次状态变化
        # self.get_logger().info(f"TTS Status changed: {self._last_status} -> {msg.status}")
        
        self._last_status = msg.status

        # PLAYING = 2, STOPPED = 1 (参考 2.py 和 no_gaze.py 的定义)
        if msg.status == 2:
            self.tts_finished = False
            self.get_logger().info('TTS 开始播放')
        elif msg.status == 1:
            self.tts_finished = True
            self.get_logger().info('TTS 播放完成')

    def speak(self, text: str, language_code='en', disable_lipsync=False):
        msg = TTSCommand()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.message = text
        msg.language_code = language_code
        msg.disable_lipsync = disable_lipsync
        self.publisher.publish(msg)
        self.get_logger().info(f'🗣️ 已发送语音消息: "{text}"')

class MotorController(Node):
    def __init__(self):
        super().__init__('motor_command_node')
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
        if motor_name not in self._motor_publishers:
            self.get_logger().error(f"未知的电机名称: {motor_name}")
            return
        msg = MotorCommand()
        msg.position = float(position)
        msg.play_time = play_time
        msg.relative = relative
        msg.disable_eyes_roll_sync = disable_eyes_roll_sync
        self._motor_publishers[motor_name].publish(msg)
        #self.get_logger().info(f"发送 {motor_name} 指令: position={position}, play_time={play_time}, relative={relative}")

    #an example motion sequence
    def perform_motion_sequence(self):
        self.send_command('base', 20)
        time.sleep(1.0)
        self.send_command('base', -20)
        time.sleep(1.0)
        self.send_command('neck_pitch', 15)
        time.sleep(0.8)
        self.send_command('neck_pitch', -15)
        time.sleep(0.8)
        self.send_command('neck_roll', 10)
        time.sleep(0.8)
        self.send_command('neck_roll', -10)
        time.sleep(0.8)
        self.send_command('left_eye_yaw', 10)
        self.send_command('right_eye_yaw', 10)
        time.sleep(0.5)
        self.send_command('left_eye_yaw', -10)
        self.send_command('right_eye_yaw', -10)
        time.sleep(0.5)
        self.send_command('left_eye_roll', 5)
        self.send_command('right_eye_roll', -5)
        time.sleep(0.5)
        self.send_command('left_eye_stroke', 100)
        self.send_command('right_eye_stroke', 100)
        time.sleep(0.3)
        self.send_command('left_eye_stroke', 0)
        self.send_command('right_eye_stroke', 0)

class RobotEnv:
    def __init__(self, args=None):
        #rclpy.init(args=args)
        self.tts_node = TTSSpeaker()
        self.motor_node = MotorController()
        time.sleep(1.0)  # 等待连接建立

    def speak(self, text, language_code='', disable_lipsync=False):
        self.tts_node.speak(text, language_code, disable_lipsync)
        time_to_wait = 1.5 + len(text) * 0.05
        rclpy.spin_once(self.tts_node, timeout_sec=time_to_wait)

    def shutdown(self):
        self.tts_node.destroy_node()
        self.motor_node.destroy_node()
        # 不在这里调用 rclpy.shutdown()，因为主程序会处理

def main(args=None):
    rclpy.init(args=args)
    env = RobotEnv()
    try:
        env.speak("Hello, I am ready.")
        time.sleep(2)
        env.move_motor('base', 90.0, 1.0)
        time.sleep(2)
        env.move_motor('base', -90.0, 1.0)
        time.sleep(2)
        env.move_motor('base', 0.0, 1.0)
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()