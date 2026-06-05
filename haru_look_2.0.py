import math
import time
import random
from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from strawberry_ros_msgs.msg import Skeletons
from std_msgs.msg import String, Bool
from haru2_core_msgs.srv import Routine
from haru2_core_msgs.msg import RoutineStatus, MotorCommand

# ================= 全局配置参数 =================
EYE_CONTACT_THRESHOLD = 0.35
CONTROL_RATE = 10.0
DT = 1.0 / CONTROL_RATE
HARDWARE_PLAY_TIME = 150

# 关系参数
FRIEND_JA_MULTIPLIER = 1.5
FRIEND_PITCH_MULTIPLIER = 0.9
ACQ_CHANGE_INTERVAL = 4.0
ACQ_SMOOTH_FACTOR = 0.03
ACQ_COINCIDENCE_CHANCE = 0.2
ACQ_EXCLUSION_ZONE = 0.6
OPP_SMOOTH_FACTOR_STARE = 0.05
OPP_SMOOTH_FACTOR_PANIC = 0.20

# RL调度信号参数
ATTENTION_SMOOTH_WINDOW = 5
WAIT_MIN_SECONDS = 0.4
WAIT_NO_BBOX_SECONDS = 0.9
WAIT_BBOX_EXTENSION_SECONDS = 1.2
WAIT_BBOX_EXPECT_TIMEOUT = 0.8
WAIT_MAX_SECONDS = 3.5


def quaternion_to_euler(x, y, z, w):
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    return 0, pitch, yaw


def fix_kinect_yaw(yaw_rad):
    if yaw_rad > 0:
        return yaw_rad - math.pi
    else:
        return yaw_rad + math.pi


class HaruLookFinal(Node):
    def __init__(self):
        super().__init__('haru_look_final')
        self.get_logger().info("=== Haru Look (实时日志版) 启动 ===")
        self.get_logger().info("已恢复终端实时状态打印 (0.5s/次)")

        # 1. 话题发布
        self._motor_publishers = {
            'base': self.create_publisher(MotorCommand, '/haru2/cmd_base_pos', 10),
            'neck_pitch': self.create_publisher(MotorCommand, '/haru2/cmd_neck_pitch_pos', 10),
            'neck_roll': self.create_publisher(MotorCommand, '/haru2/cmd_neck_roll_pos', 10),
            'left_eye_yaw': self.create_publisher(MotorCommand, '/haru2/cmd_left_eye_yaw_pos', 10),
            'right_eye_yaw': self.create_publisher(MotorCommand, '/haru2/cmd_right_eye_yaw_pos', 10),
            'left_eye_stroke': self.create_publisher(MotorCommand, '/haru2/cmd_left_eye_stroke_pos', 10),
            'right_eye_stroke': self.create_publisher(MotorCommand, '/haru2/cmd_right_eye_stroke_pos', 10),
        }

        qos_rel = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)

        # 2. 订阅
        self.create_subscription(Skeletons, '/strawberry/azure/skeletons', self._skeleton_callback, 10)
        self.create_subscription(String, '/haru/social_relation', self._relation_callback, qos_rel)
        self.create_subscription(Bool, '/haru/system_active', self._active_cb, 10)
        self.create_subscription(RoutineStatus, '/haru2/routine_status', self._routine_status_cb, 10)
        # 由对话/VLM模块提供：用户说完一轮、当前轮是否有bbox
        self.create_subscription(Bool, '/haru/user_turn_done', self._user_turn_done_cb, 10)
        self.create_subscription(Bool, '/haru/vlm_bbox_ready', self._vlm_bbox_ready_cb, 10)

        # 表情服务
        self.routine_client = self.create_client(Routine, '/haru2/execute_routine')
        self.expr_active_pub = self.create_publisher(Bool, "/haru/expression_active", qos_rel)
        # 给RL部署节点的输入信号
        self.rl_waiting_pub = self.create_publisher(Bool, '/rl/waiting_active', 10)
        self.rl_attention_pub = self.create_publisher(Bool, '/rl/user_attention', 10)

        # 3. 状态变量
        self.system_active = False
        self.current_relation = "Unrelated"
        self.expression_active = False

        self.has_user = False
        self.last_person_vec = None
        self.person_vec_smoothed = None
        self.user_yaw = 0.0
        self.user_pitch = 0.0
        self.last_person_seen_time = 0.0

        # 行为变量
        self.global_current_yaw = 0.0
        self.global_current_pitch = 0.0
        self.first_gaze_mode = False
        self.first_gaze_start_time = 0.0

        # 表情控制
        self.pending_expression = False
        self.expr_inflight = False
        self.expr_inflight_time = 0.0

        # 关系专用变量
        self.acq_target_yaw = 0.0
        self.acq_target_pitch = 0.0
        self.acq_last_change_time = 0.0
        self.acq_is_coincidence = False
        self.opp_target_yaw = 0.0
        self.opp_target_pitch = 0.0
        self.opp_is_caught = False

        # ★★★ 日志流控变量 ★★★
        self.last_log_time = 0.0

        # RL信号状态
        self.attention_history = deque(maxlen=ATTENTION_SMOOTH_WINDOW)
        self.user_attention = False

        self.waiting_active = False
        self.wait_start_time = 0.0
        self.wait_deadline = 0.0
        self.waiting_expect_bbox = False
        self.waiting_bbox_seen = False
        self.last_waiting_log = 0.0

        self.create_timer(DT, self._control_loop)

    def _skeleton_callback(self, msg: Skeletons):
        prev_has = self.has_user
        if not msg.skeletons:
            self.has_user = False;
            return

        sk = msg.skeletons[0]
        head = next((p for p in sk.body_parts if p.name == "head"), None)
        if not head: self.has_user = False; return

        pt = head.pose.position
        vec_raw = np.array([-pt.y, -pt.z, pt.x], dtype=float)

        alpha = 0.3
        if self.person_vec_smoothed is None:
            self.person_vec_smoothed = vec_raw
        else:
            self.person_vec_smoothed = (1.0 - alpha) * self.person_vec_smoothed + alpha * vec_raw

        o = head.pose.orientation
        _, pitch, raw_yaw = quaternion_to_euler(o.x, o.y, o.z, o.w)
        self.user_yaw = fix_kinect_yaw(raw_yaw)
        self.user_pitch = pitch
        self.has_user = True
        self.last_person_seen_time = time.time()

        if (not prev_has) and self.has_user:
            if self.current_relation != "Opponent":
                self.first_gaze_mode = True
                self.first_gaze_start_time = time.time()
                self.get_logger().info("👀 初次检测到人 -> 进入 Initial Gaze")

    def _relation_callback(self, msg: String):
        new_rel = (msg.data or "Unrelated").strip()
        if new_rel != self.current_relation:
            self.get_logger().info(f"🔄 关系切换: {self.current_relation} -> {new_rel}")
            self.current_relation = new_rel
            self.acq_last_change_time = 0
            self.opp_is_caught = False
            if new_rel in ("Friends", "Opponent"):
                self.pending_expression = True
            else:
                self.pending_expression = False

    def _active_cb(self, msg: Bool):
        self.system_active = bool(msg.data)
        if not self.system_active:
            self.waiting_active = False
            self._go_to_default_pose()

    def _routine_status_cb(self, msg: RoutineStatus):
        playing = bool(getattr(msg, "playing", False))
        if playing:
            self.expression_active = True
            self.expr_inflight = False
        else:
            if not self.expr_inflight: self.expression_active = False

    def _user_turn_done_cb(self, msg: Bool):
        if not bool(msg.data):
            return
        now = time.time()
        self.waiting_active = True
        self.wait_start_time = now
        self.wait_deadline = now + WAIT_MIN_SECONDS
        self.waiting_expect_bbox = True
        self.waiting_bbox_seen = False
        self.get_logger().info("[RL信号] 用户回合结束 -> waiting_active=True")

    def _vlm_bbox_ready_cb(self, msg: Bool):
        if not bool(msg.data):
            return
        now = time.time()
        if self.waiting_active:
            self.waiting_bbox_seen = True
            self.waiting_expect_bbox = False
            self.wait_deadline = max(self.wait_deadline, now + WAIT_BBOX_EXTENSION_SECONDS)
            self.get_logger().info("[RL信号] 收到bbox -> 延长等待窗口")

    def _update_waiting_state(self, now: float):
        if not self.system_active:
            self.waiting_active = False
            return

        if not self.waiting_active:
            return

        elapsed = now - self.wait_start_time

        # bbox在预期时间内没来：按“无bbox更快”策略走短等待
        if self.waiting_expect_bbox and elapsed >= WAIT_BBOX_EXPECT_TIMEOUT:
            self.waiting_expect_bbox = False
            self.wait_deadline = max(self.wait_deadline, self.wait_start_time + WAIT_NO_BBOX_SECONDS)

        hard_deadline = self.wait_start_time + WAIT_MAX_SECONDS
        final_deadline = min(self.wait_deadline, hard_deadline)

        if now >= final_deadline:
            self.waiting_active = False
            self.waiting_expect_bbox = False
            self.waiting_bbox_seen = False
            self.get_logger().info("[RL信号] waiting窗口结束 -> waiting_active=False")

    def _publish_rl_signals(self, now: float):
        looking_now = bool(self.has_user and abs(self.user_yaw) < EYE_CONTACT_THRESHOLD)
        self.attention_history.append(1 if looking_now else 0)
        att_ratio = sum(self.attention_history) / len(self.attention_history) if self.attention_history else 0.0
        self.user_attention = att_ratio >= 0.5

        self.rl_waiting_pub.publish(Bool(data=bool(self.waiting_active)))
        self.rl_attention_pub.publish(Bool(data=bool(self.user_attention)))

        if (now - self.last_waiting_log) > 1.0:
            self.last_waiting_log = now
            self.get_logger().info(
                f"[RL信号] waiting={int(self.waiting_active)} attention={int(self.user_attention)} "
                f"att_ratio={att_ratio:.2f} bbox_seen={int(self.waiting_bbox_seen)}"
            )

    def _control_loop(self):
        now = time.time()

        self._update_waiting_state(now)
        self._publish_rl_signals(now)

        if not self.system_active: return

        # 1. 无人
        if not self.has_user or (now - self.last_person_seen_time) > 1.0:
            self._go_to_default_pose(play_time=250)
            self.first_gaze_mode = False
            return

        # 2. 表情 (最高优先级)
        if self.expression_active or self.expr_inflight:
            self._handle_expression_timeout(now)
            return

        if self.pending_expression:
            self._trigger_expression(self.current_relation)
            self.pending_expression = False
            return

        # 3. 初次见面
        if self.first_gaze_mode:
            if (now - self.first_gaze_start_time) < 3.0:
                self._basic_tracking(150)
                return
            else:
                self.first_gaze_mode = False
                self.get_logger().info("Initial Gaze 结束 -> 进入关系模式")

        # 4. 关系行为
        rel = self.current_relation
        if rel == "Friends":
            self._update_friends_behavior(now)
        elif rel == "Acquaintance":
            self._update_acquaintance_behavior(now)
        elif rel == "Opponent":
            self._update_opponent_behavior(now)
        else:
            self._update_acquaintance_behavior(now)

    # --- 行为逻辑 (带日志) ---
    def _update_friends_behavior(self, now):
        is_looking = abs(self.user_yaw) < EYE_CONTACT_THRESHOLD
        target_yaw, target_pitch = 0.0, 0.0

        # 控制打印频率 (0.5秒一次)
        should_log = (now - self.last_log_time > 0.5)

        if is_looking:
            if should_log:
                self.get_logger().info(f"[💖 朋友] 对视跟随中... (Pitch:{math.degrees(self.user_pitch):.1f}°)")
                self.last_log_time = now

            if self.person_vec_smoothed is not None:
                vec = self.person_vec_smoothed
                target_yaw = math.atan2(vec[0], vec[2])
                target_pitch = self.user_pitch * FRIEND_PITCH_MULTIPLIER
                self._distribute_look_standard(target_yaw, target_pitch, 150, 0.35)
        else:
            if should_log:
                self.get_logger().info(f"[🤝 朋友] 共同关注 (看你所看)...")
                self.last_log_time = now

            target_yaw = self.user_yaw * FRIEND_JA_MULTIPLIER
            target_pitch = self.user_pitch * FRIEND_PITCH_MULTIPLIER
            self._distribute_look_standard(target_yaw, target_pitch, 200, 0.2)

        self.global_current_yaw = target_yaw
        self.global_current_pitch = target_pitch

    def _update_acquaintance_behavior(self, now):
        is_looking = abs(self.user_yaw) < EYE_CONTACT_THRESHOLD
        should_log = (now - self.last_log_time > 0.5)

        if is_looking:
            if should_log:
                self.get_logger().info(f"[👀 陌生人] 礼貌回视...")
                self.last_log_time = now

            target_yaw, target_pitch = 0.0, 0.0
            if self.person_vec_smoothed is not None:
                vec = self.person_vec_smoothed
                target_yaw = math.atan2(vec[0], vec[2])
                target_pitch = self.user_pitch * 0.3

            self.global_current_yaw = target_yaw
            self.global_current_pitch = target_pitch
            self._distribute_look_standard(target_yaw, target_pitch, 150, 0.25)
            self.acq_last_change_time = 0
        else:
            self._acq_wander_decision(now)
            diff_yaw = self.acq_target_yaw - self.global_current_yaw
            diff_pitch = self.acq_target_pitch - self.global_current_pitch

            self.global_current_yaw += diff_yaw * ACQ_SMOOTH_FACTOR
            self.global_current_pitch += diff_pitch * ACQ_SMOOTH_FACTOR

            if should_log:
                mode_str = "🎲 巧合" if self.acq_is_coincidence else "🍃 避嫌"
                self.get_logger().info(f"[{mode_str} 陌生人] 漫游中... (Target:{self.acq_target_yaw:.2f})")
                self.last_log_time = now

            self._distribute_look_standard(self.global_current_yaw, self.global_current_pitch, HARDWARE_PLAY_TIME, 0.1)

    def _update_opponent_behavior(self, now):
        is_looking = abs(self.user_yaw) < EYE_CONTACT_THRESHOLD
        current_smooth = OPP_SMOOTH_FACTOR_STARE
        should_log = (now - self.last_log_time > 0.5)

        if is_looking:
            if not self.opp_is_caught:
                self._opp_pick_panic_target()
                self.opp_is_caught = True

            if should_log:
                self.get_logger().info(f"[🙈 敌人] 被抓包! 扭头装傻 (Panic)")
                self.last_log_time = now

            current_smooth = OPP_SMOOTH_FACTOR_PANIC
        else:
            self.opp_is_caught = False
            if self.person_vec_smoothed is not None:
                vec = self.person_vec_smoothed
                self.opp_target_yaw = math.atan2(vec[0], vec[2])
                self.opp_target_pitch = self.user_pitch * 0.5

            if should_log:
                self.get_logger().info(f"[👀 敌人] 偷偷盯着你看... (Staring)")
                self.last_log_time = now

            current_smooth = OPP_SMOOTH_FACTOR_STARE

        diff_yaw = self.opp_target_yaw - self.global_current_yaw
        diff_pitch = self.opp_target_pitch - self.global_current_pitch
        self.global_current_yaw += diff_yaw * current_smooth
        self.global_current_pitch += diff_pitch * current_smooth

        stroke = 0.0 if self.opp_is_caught else 0.35
        self._distribute_look_sneaky(self.global_current_yaw, self.global_current_pitch, HARDWARE_PLAY_TIME, stroke)

    def _acq_wander_decision(self, now):
        if (now - self.acq_last_change_time) > ACQ_CHANGE_INTERVAL:
            if random.random() < ACQ_COINCIDENCE_CHANCE:
                # 巧合：随机看任意位置
                self.acq_target_yaw = random.uniform(-1.5, 1.5)
                self.acq_is_coincidence = True
            else:
                # 避嫌：原来的逻辑是只要不看人就行
                # ★★★ 修改：强制选大角度 (Head Turn) ★★★
                # 意思是在 [-1.5, -0.8] 和 [0.8, 1.5] 这两个区间里选，避开中间
                if random.random() < 0.5:
                    self.acq_target_yaw = random.uniform(-1.5, -0.8)  # 猛向右转
                else:
                    self.acq_target_yaw = random.uniform(0.8, 1.5)  # 猛向左转

                # 如果随机到的方向恰好是人脸方向，就取反，确保看向反方向
                if abs(self.acq_target_yaw - self.user_yaw) < ACQ_EXCLUSION_ZONE:
                    self.acq_target_yaw = -self.acq_target_yaw

                self.acq_is_coincidence = False

            # 也可以让抬头的幅度随机范围变大一点
            self.acq_target_pitch = random.uniform(-0.4, 0.4)
            self.acq_last_change_time = now

    def _opp_pick_panic_target(self):
        if self.user_yaw > 0:
            self.opp_target_yaw = random.uniform(-1.4, -0.6)
        else:
            self.opp_target_yaw = random.uniform(0.6, 1.4)
        self.opp_target_pitch = random.uniform(0.15, 0.4)

    def _basic_tracking(self, play_time):
        if self.person_vec_smoothed is not None:
            vec = self.person_vec_smoothed
            yaw = math.atan2(vec[0], vec[2])
            pitch = self.user_pitch * 0.5
            self._distribute_look_standard(yaw, pitch, play_time, 0.3)

    def _distribute_look_standard(self, total_yaw, pitch, play_time, stroke):
        total_yaw = max(min(total_yaw, 1.5), -1.5)
        pitch = max(min(pitch, 0.5), -0.5)
        eye_limit = 0.25
        eye_yaw = max(min(total_yaw, eye_limit), -eye_limit)
        base_yaw = total_yaw - eye_yaw
        self._send_motor('base', base_yaw, play_time)
        self._send_motor('neck_pitch', pitch, play_time)
        self._send_motor('left_eye_yaw', eye_yaw, play_time)
        self._send_motor('right_eye_yaw', eye_yaw, play_time)
        self._send_motor('neck_roll', base_yaw * 0.05, play_time)
        self._send_motor('left_eye_stroke', stroke, play_time)
        self._send_motor('right_eye_stroke', stroke, play_time)

    def _distribute_look_sneaky(self, total_yaw, pitch, play_time, stroke):
        total_yaw = max(min(total_yaw, 1.5), -1.5)
        pitch = max(min(pitch, 0.5), -0.5)
        eye_limit = 0.35
        eye_yaw = max(min(total_yaw, eye_limit), -eye_limit)
        base_yaw = total_yaw - eye_yaw
        self._send_motor('base', base_yaw, play_time)
        self._send_motor('neck_pitch', pitch, play_time)
        self._send_motor('left_eye_yaw', eye_yaw, play_time)
        self._send_motor('right_eye_yaw', eye_yaw, play_time)
        self._send_motor('neck_roll', 0.0, play_time)
        self._send_motor('left_eye_stroke', stroke, play_time)
        self._send_motor('right_eye_stroke', stroke, play_time)

    def _send_motor(self, name, pos, play_time):
        if name not in self._motor_publishers: return
        msg = MotorCommand()
        msg.position = float(pos);
        msg.play_time = int(play_time);
        msg.disable_eyes_roll_sync = False
        self._motor_publishers[name].publish(msg)

    def _go_to_default_pose(self, play_time=500):
        for name in self._motor_publishers: self._send_motor(name, 0.0, play_time)

    def _trigger_expression(self, rel):
        self.get_logger().info(f"🎭 触发表情 (关系: {rel})")
        self.expression_active = True
        self.expr_active_pub.publish(Bool(data=True))
        self.expr_inflight = True
        self.expr_inflight_time = time.time()
        rid = 0
        if rel == "Friends":
            rid = 2068
        elif rel == "Opponent":
            rid = 2066
            self._distribute_look_sneaky(0.8, 0.2, 250, 0.0)
        else:
            rid = random.choice([2032, 2052, 2093])
        if self.routine_client.service_is_ready():
            req = Routine.Request();
            req.routine = int(rid)
            self.routine_client.call_async(req)

    def _handle_expression_timeout(self, now):
        if self.expr_inflight and (now - self.expr_inflight_time) > 1.0:
            self.expr_inflight = False


def main(args=None):
    rclpy.init(args=args)
    node = HaruLookFinal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node();
        rclpy.shutdown()


if __name__ == "__main__":
    main()