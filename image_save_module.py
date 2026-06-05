import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
import cv2
import numpy as np
import base64
import sys 
import time
from typing import Optional, List, Dict
from cv_bridge import CvBridge
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from rclpy.qos import QoSProfile

def imgmsg_to_cv2(img_msg):
    bridge = CvBridge()
    try:
        # 检查是否是压缩图像
        if hasattr(img_msg, 'format'):  # CompressedImage
            return bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        else:  # 普通Image消息
            return bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
    except Exception as e:
        rclpy.logging.get_logger("img_utils").error(f"图像转换失败: {e}")
        return None

def depth_imgmsg_to_cv2(img_msg):
    bridge = CvBridge()
    try:
        # 深度图像通常是16UC1或32FC1格式，保持原始格式
        if hasattr(img_msg, 'format'):  # CompressedImage
            return bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
        else:  # 普通Image消息
            return bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
    except Exception as e:
        rclpy.logging.get_logger("img_utils").error(f"深度图像转换失败: {e}")
        return None

class ImageSaveModule(Node):
    def __init__(self):
        super().__init__('image_save_module')
        self.bridge = CvBridge()
        qos_profile = QoSProfile(depth=1)

        self.rgb_sub = self.create_subscription(
            CompressedImage,
            "/azure_kinect/rgb/image_raw/compressed", 
            self.pic_callback,
            qos_profile
        )

        # ✅ 使用对齐到 RGB 的深度图像
        self.depth_sub = self.create_subscription(
            Image,
            "/azure_kinect/depth_to_rgb/image_raw",
            self.depth_callback,
            qos_profile
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/azure_kinect/rgb/camera_info",
            self.camera_info_callback,
            qos_profile
        )

        self.latest_image: Optional[np.ndarray] = None
        self.latest_image_base64: Optional[str] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.camera_intrinsics: Optional[Dict] = None
        self.image_captured = False
        self.depth_captured = False  # 添加深度图像捕获标志

    def check_sensor_status(self) -> dict:
        return {
            "image_ready": self.latest_image_base64 is not None,
            "depth_ready": self.latest_depth is not None,
            "camera_info_ready": self.camera_intrinsics is not None,
            "image_topic": "/azure_kinect/rgb/image_raw/compressed",
            "depth_topic": "/azure_kinect/depth_to_rgb/image_raw",
            "camera_info_topic": "/azure_kinect/rgb/camera_info",
            "last_update": self.get_clock().now().nanoseconds / 1e9
        }

    def reset_capture(self):
        self.image_captured = False
        self.depth_captured = False  # 重置深度图像捕获标志
        #self.latest_image = None
        #self.latest_depth = None

    def wait_for_data(self, timeout=5.0) -> bool:
        start_time = time.time()
        while rclpy.ok():
            if self.latest_image_base64 is not None and self.latest_depth is not None and self.camera_intrinsics is not None:
                return True
            if time.time() - start_time > timeout:
                msg = "等待传感器数据超时"
                # ROS 日志
                self.get_logger().warn(msg)
                # 同时打印到终端，便于 full_terminal_output 捕获
                try:
                    print(f"⚠️ {msg}")
                except Exception:
                    pass
                return False
            time.sleep(0.1)

    def pic_callback(self, msg):
        if self.image_captured:
            return
        cv_image = imgmsg_to_cv2(msg)
        if cv_image is not None:
            self.latest_image = cv_image
            # 优化：压缩图像减小上传大小，使用更低的质量参数加速编码
            # 同时适度降低分辨率（640x360）减少传输量
            height, width = cv_image.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                resized = cv2.resize(cv_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                resized = cv_image
            
            # JPEG质量70（默认95），在视觉质量和文件大小间平衡
            _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
            self.latest_image_base64 = base64.b64encode(buffer).decode('utf-8')
            self.image_captured = True

    def depth_callback(self, msg):
        if self.depth_captured:
            return  # 避免重复处理
        depth_image = depth_imgmsg_to_cv2(msg)
        if depth_image is not None:
            # 确保深度图像是float32格式用于计算
            if depth_image.dtype == np.uint16:
                # 如果是uint16，转换为float32（单位：毫米转米）
                self.latest_depth = depth_image.astype(np.float32) / 1000.0
            else:
                self.latest_depth = depth_image.astype(np.float32)
            self.depth_captured = True

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_intrinsics = {
            "fx": msg.k[0],
            "fy": msg.k[4],
            "cx": msg.k[2],
            "cy": msg.k[5]
        }

    def get_3d_coordinates(self, pixel_points: List[List[int]], debug: bool = False) -> List[Optional[List[float]]]:
        if self.latest_depth is None or self.camera_intrinsics is None:
            msg = "❌ 缺少深度图或相机内参，无法计算3D坐标"
            # ROS 日志
            try:
                self.get_logger().warn(msg)
            except Exception:
                pass
            # 始终把完整警告打印到 stdout，使 full_terminal_output.log 能捕获详细信息
            try:
                print(f"⚠️ {msg} -- latest_depth={None if self.latest_depth is None else 'present'}, camera_intrinsics={None if self.camera_intrinsics is None else 'present'}")
            except Exception:
                pass
            return [None] * len(pixel_points)

        fx = self.camera_intrinsics["fx"]
        fy = self.camera_intrinsics["fy"]
        cx_intr = self.camera_intrinsics["cx"]
        cy_intr = self.camera_intrinsics["cy"]

        coords = []
        window = 5

        for idx, (u, v) in enumerate(pixel_points):
            if not (0 <= u < self.latest_depth.shape[1] and 0 <= v < self.latest_depth.shape[0]):
                if debug:
                    msg = f"像素坐标 ({u},{v}) 超出图像范围"
                    self.get_logger().warn(msg)
                    try:
                        print(f"⚠️ {msg}")
                    except Exception:
                        pass
                coords.append(None)
                continue

            depths = []
            for dy in range(-window, window + 1):
                for dx in range(-window, window + 1):
                    uu = u + dx
                    vv = v + dy
                    if 0 <= uu < self.latest_depth.shape[1] and 0 <= vv < self.latest_depth.shape[0]:
                        z = self.latest_depth[vv, uu]
                        if z > 0 and not np.isnan(z):
                            depths.append(z)

            if not depths:
                # 详细警告，包括像素位置、邻域窗口大小、深度数组形状等信息
                msg = (
                    f"未找到深度信息: 像素({u},{v}) 在半径{window}的邻域内无有效深度值。"
                    f" depth_shape={None if self.latest_depth is None else self.latest_depth.shape}"
                )
                try:
                    self.get_logger().warn(msg)
                except Exception:
                    pass
                # 始终打印到 stdout，确保 full_terminal_output.log 能捕获完整警告文本
                try:
                    print(f"⚠️ {msg}")
                except Exception:
                    pass
                coords.append(None)
                continue

            z_avg = float(np.mean(depths))
            x = (u - cx_intr) * z_avg / fx
            y = (v - cy_intr) * z_avg / fy
            coords.append([x, y, z_avg])

            if debug:
                self.get_logger().info(f"[3D转换] 像素({u},{v}) -> 深度{z_avg:.3f}m -> 坐标[{x:.3f}, {y:.3f}, {z_avg:.3f}]")

        return coords

    def get_latest_image_base64(self):
        return self.latest_image_base64

    def save_annotated_image(self, objects: List[Dict]):
        """生成标注图与深度灰度图并写入临时目录，返回文件路径。

        说明
        - 不再固定保存到 ~/haru_saved_images，也不在这里打印“已保存…”（避免重复保存/刷屏）。
        - 统一由 ConversationSessionLogger.save_images() 复制到会话目录 images/。
        """

        if self.latest_image is None:
            return {}

        # 使用OpenCV直接绘制，获得更好的视觉效果
        annotated = self.latest_image.copy()

        for obj in objects:
            name = obj.get("name", "未知")
            bbox = obj.get("bboxes")
            confidence = obj.get("confidence", 1.0)
            
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                
                # 绘制绿色矩形框
                color = (0, 255, 0)  # 绿色 (BGR格式)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签文本
                label_text = f"{name} ({confidence:.2f})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                
                # 计算文字背景框大小
                (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                
                # 绘制文字背景（绿色）
                cv2.rectangle(annotated, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                
                # 绘制黑色文字
                cv2.putText(annotated, label_text, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)

        import time
        import os

        # 临时写入：后续由 session_logger 复制到会话目录（避免重复保存）
        base_dir = os.path.join("/tmp", "haru_images")
        os.makedirs(base_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        marked_path = os.path.join(base_dir, f"marked_image_{timestamp}.jpg")
        depth_path = os.path.join(base_dir, f"depth_image_{timestamp}.jpg")
        rgb_path = os.path.join(base_dir, f"rgb_image_{timestamp}.jpg")

        out = {}
        try:
            # 保存 rgb 原图（BGR）
            cv2.imwrite(rgb_path, self.latest_image)
            out["rgb"] = rgb_path
        except Exception as e:
            self.get_logger().error(f"保存RGB图失败: {e}")

        try:
            cv2.imwrite(marked_path, annotated)
            out["marked"] = marked_path
        except Exception as e:
            self.get_logger().error(f"保存标注图失败: {e}")

        if self.latest_depth is not None:
            depth_image_normalized = cv2.normalize(self.latest_depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_image_gray = depth_image_normalized.astype(np.uint8)
            try:
                cv2.imwrite(depth_path, depth_image_gray)
                out["depth"] = depth_path
            except Exception as e:
                self.get_logger().error(f"保存深度图失败: {e}")

        return out

    def save_overlay_image(self):
        """生成深度叠加图并写入临时目录，返回文件路径。

        说明
        - 同 save_annotated_image：不再固定写入 ~/haru_saved_images，也不 print。
        - 统一由 ConversationSessionLogger.save_images() 复制到会话目录 images/。
        """

        if self.latest_image is None or self.latest_depth is None:
            return None

        # 深度图归一化并伪彩色
        depth_normalized = cv2.normalize(self.latest_depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

        # 确保 RGB 图为三通道（处理可能的 RGBA 格式）
        if self.latest_image.shape[2] == 4:
            self.latest_image = cv2.cvtColor(self.latest_image, cv2.COLOR_RGBA2BGR)

        # 尺寸对齐：调整深度图尺寸与 RGB 图一致
        if self.latest_image.shape[:2] != depth_colored.shape[:2]:
            depth_colored = cv2.resize(depth_colored, 
                                    (self.latest_image.shape[1], self.latest_image.shape[0]))

        # 通道对齐：确保深度图为三通道
        if len(depth_colored.shape) == 2 or depth_colored.shape[2] != 3:
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_GRAY2BGR)

        # 叠加图像
        overlay = cv2.addWeighted(self.latest_image, 0.6, depth_colored, 0.4, 0)
        
        import time
        import os
        base_dir = os.path.join("/tmp", "haru_images")
        os.makedirs(base_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        overlay_path = os.path.join(base_dir, f"overlay_image_{timestamp}.jpg")
        try:
            cv2.imwrite(overlay_path, overlay)
            return overlay_path
        except Exception as e:
            self.get_logger().error(f"保存叠加图失败: {e}")
            return None
