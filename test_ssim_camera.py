
#!/usr/bin/env python3
"""
测试 SSIM 相似度算法 - 从真实相机获取图像

使用说明：
1. 运行脚本后，第一个场景准备好（杯子在旁边）
2. 按任意键捕获第一张图片
3. 改变场景（把杯子拿在手里）
4. 按任意键捕获第二张图片
5. 脚本会计算并显示两张图片的 SSIM 相似度
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import sys
import time

class SSIMTester(Node):
    def __init__(self):
        super().__init__('ssim_tester')
        
        # 订阅 RGB 图像话题
        self.rgb_sub = self.create_subscription(
            CompressedImage,
            "/azure_kinect/rgb/image_raw/compressed",
            self.image_callback,
            10
        )
        
        self.latest_image = None
        self.image_ready = False
        
        print("=" * 70)
        print("SSIM 相似度测试 - 真实相机版")
        print("=" * 70)
        print("\n等待相机数据...")
    
    def image_callback(self, msg):
        """接收 RGB 图像"""
        try:
            # 解码压缩图像
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.latest_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.image_ready = True
        except Exception as e:
            self.get_logger().error(f"图像解码失败: {e}")
    
    def wait_for_image(self):
        """等待接收到图像"""
        print("等待相机数据...", end='', flush=True)
        start_time = time.time()
        while not self.image_ready and time.time() - start_time < 5.0:
            rclpy.spin_once(self, timeout_sec=0.1)
            print(".", end='', flush=True)
        print()
        
        if not self.image_ready:
            print("❌ 超时：未接收到相机数据")
            return False
        
        print("✅ 相机数据就绪")
        return True
    
    def capture_image(self, scene_name):
        """捕获当前图像"""
        print(f"\n{'='*70}")
        print(f"场景 {scene_name}")
        print(f"{'='*70}")
        print("准备好场景后，按 Enter 键捕获图像...")
        input()
        
        # 捕获最新的图像
        rclpy.spin_once(self, timeout_sec=0.1)
        
        if self.latest_image is None:
            print("❌ 未能捕获图像")
            return None
        
        img = self.latest_image.copy()
        print(f"✅ 已捕获图像 (尺寸: {img.shape[1]}x{img.shape[0]})")
        
        # 保存图像供查看
        filename = f"ssim_test_{scene_name}_{int(time.time())}.jpg"
        cv2.imwrite(filename, img)
        print(f"💾 已保存到: {filename}")
        
        return img
    
    def calculate_ssim(self, img1, img2):
        """计算两张图像的 SSIM 相似度"""
        # 降采样到 256x256
        small1 = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_AREA)
        small2 = cv2.resize(img2, (256, 256), interpolation=cv2.INTER_AREA)
        
        # 转换为灰度图
        gray1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
        
        # 计算 SSIM
        similarity = ssim(gray1, gray2, data_range=255)
        
        return similarity
    
    def run_test(self):
        """运行测试流程"""
        # 等待相机准备好
        if not self.wait_for_image():
            return
        
        # 捕获第一张图像（杯子在旁边）
        print("\n📸 请准备场景1：杯子放在旁边")
        img1 = self.capture_image("scene1")
        if img1 is None:
            return
        
        # 捕获第二张图像（杯子拿在手里）
        print("\n📸 请准备场景2：把杯子拿在手里")
        img2 = self.capture_image("scene2")
        if img2 is None:
            return
        
        # 计算相似度
        print(f"\n{'='*70}")
        print("计算 SSIM 相似度...")
        print(f"{'='*70}")
        
        similarity = self.calculate_ssim(img1, img2)
        
        print(f"\n📊 结果:")
        print(f"  SSIM 相似度: {similarity:.6f}")
        print(f"  差异程度: {(1 - similarity) * 100:.2f}%")
        print(f"  当前阈值: 0.95")
        
        if similarity >= 0.95:
            print(f"  判定: ❌ 场景未变化 (>= 0.95) - 检测不到差异")
        else:
            print(f"  判定: ✅ 场景已变化 (< 0.95) - 成功检测到差异")
        
        # 建议
        print(f"\n💡 分析:")
        if similarity >= 0.95:
            print("  场景变化很小或者:")
            print("  - 杯子在画面中占比较小")
            print("  - 手和杯子的位置与原位置重叠")
            print("  建议: 可以将阈值降低到 0.90-0.93 提高灵敏度")
        else:
            print("  SSIM 成功检测到结构变化!")
            print("  当前阈值 0.95 合适，能够检测到你拿杯子的动作")
        
        print(f"\n{'='*70}")

def main():
    rclpy.init()
    
    try:
        tester = SSIMTester()
        tester.run_test()
    except KeyboardInterrupt:
        print("\n\n测试被中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            tester.destroy_node()
        except:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()
