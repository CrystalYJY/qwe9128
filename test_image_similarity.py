#!/usr/bin/env python3
"""
图像相似度测试脚本（ROS2版本）
自动从ROS2话题订阅图像，间隔15秒捕获两张图片进行对比测试
"""

import cv2
import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile

class ImageSimilarityTester(Node):
    """图像相似度测试器（ROS2版本）"""
    
    def __init__(self):
        super().__init__('image_similarity_tester')
        self.bridge = CvBridge()
        self.results = []
        
        # 存储捕获的两张图片
        self.image1 = None
        self.image2 = None
        self.image1_time = None
        self.image2_time = None
        
        # 订阅RGB图像话题
        qos_profile = QoSProfile(depth=1)
        self.rgb_sub = self.create_subscription(
            CompressedImage,
            "/azure_kinect/rgb/image_raw/compressed",
            self.image_callback,
            qos_profile
        )
        
        self.get_logger().info("🎯 图像相似度测试节点已启动")
        self.get_logger().info("📡 等待图像话题: /azure_kinect/rgb/image_raw/compressed")
    
    def image_callback(self, msg):
        """图像回调函数"""
        try:
            # 转换图像
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            if self.image1 is None:
                # 捕获第一张图片
                self.image1 = cv_image
                self.image1_time = time.time()
                self.get_logger().info("✅ 已捕获第一张图片")
                self.get_logger().info(f"   尺寸: {cv_image.shape}")
                self.get_logger().info("⏳ 等待15秒后捕获第二张图片...")
                
            elif self.image2 is None:
                # 检查时间间隔
                elapsed = time.time() - self.image1_time
                if elapsed >= 15.0:
                    # 捕获第二张图片
                    self.image2 = cv_image
                    self.image2_time = time.time()
                    self.get_logger().info(f"✅ 已捕获第二张图片（间隔: {elapsed:.1f}秒）")
                    self.get_logger().info(f"   尺寸: {cv_image.shape}")
                    
                    # 开始对比测试
                    self.get_logger().info("\n" + "="*70)
                    self.get_logger().info("开始相似度测试...")
                    self.get_logger().info("="*70)
                    self.compare_all_methods()
                    
        except Exception as e:
            self.get_logger().error(f"图像处理失败: {e}")
    
    def method1_histogram_comparison(self, img1, img2):
        """方法1：直方图对比"""
        start = time.time()
        
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        elapsed = time.time() - start
        return similarity, elapsed
    
    def method2_structural_similarity(self, img1, img2):
        """方法2：结构相似性 (SSIM)"""
        start = time.time()
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        if gray1.shape != gray2.shape:
            h, w = min(gray1.shape[0], gray2.shape[0]), min(gray1.shape[1], gray2.shape[1])
            gray1 = cv2.resize(gray1, (w, h))
            gray2 = cv2.resize(gray2, (w, h))
        
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        gray1 = gray1.astype(np.float64)
        gray2 = gray2.astype(np.float64)
        
        mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        similarity = float(np.mean(ssim_map))
        
        elapsed = time.time() - start
        return similarity, elapsed
    
    def method3_feature_matching(self, img1, img2):
        """方法3：特征点匹配 (ORB)"""
        start = time.time()
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        orb = cv2.ORB_create(nfeatures=500)
        
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return 0.0, time.time() - start
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        good_matches = [m for m in matches if m.distance < 50]
        similarity = len(good_matches) / max(len(kp1), len(kp2)) if max(len(kp1), len(kp2)) > 0 else 0
        
        elapsed = time.time() - start
        return similarity, elapsed
    
    def method4_simple_pixel_diff(self, img1, img2):
        """方法4：简单像素差异"""
        start = time.time()
        
        if img1.shape != img2.shape:
            h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h))
            img2 = cv2.resize(img2, (w, h))
        
        diff = cv2.absdiff(img1, img2)
        diff_ratio = np.sum(diff) / (img1.shape[0] * img1.shape[1] * img1.shape[2] * 255)
        
        similarity = 1.0 - diff_ratio
        
        elapsed = time.time() - start
        return similarity, elapsed
    
    def method5_downsample_comparison(self, img1, img2, size=(64, 64)):
        """方法5：降采样快速对比"""
        start = time.time()
        
        small1 = cv2.resize(img1, size)
        small2 = cv2.resize(img2, size)
        
        mse = np.mean((small1.astype(float) - small2.astype(float)) ** 2)
        
        max_mse = 255 ** 2
        similarity = 1.0 - (mse / max_mse)
        
        elapsed = time.time() - start
        return similarity, elapsed
    
    def compare_all_methods(self):
        """对比所有方法"""
        if self.image1 is None or self.image2 is None:
            self.get_logger().error("图片未准备好")
            return
        
        methods = [
            ("方法1: 直方图对比", self.method1_histogram_comparison),
            ("方法2: 结构相似性(SSIM)", self.method2_structural_similarity),
            ("方法3: 特征点匹配(ORB)", self.method3_feature_matching),
            ("方法4: 简单像素差异", self.method4_simple_pixel_diff),
            ("方法5: 降采样快速对比", self.method5_downsample_comparison),
        ]
        
        results = []
        print("\n")
        for name, method in methods:
            try:
                similarity, elapsed = method(self.image1, self.image2)
                results.append({
                    'name': name,
                    'similarity': similarity,
                    'elapsed': elapsed
                })
                
                print(f"{name}")
                print(f"  相似度: {similarity:.4f} (0=完全不同, 1=完全相同)")
                print(f"  耗时: {elapsed*1000:.2f}ms")
                
                if similarity > 0.95:
                    status = "✅ 场景几乎相同，可以复用缓存"
                elif similarity > 0.85:
                    status = "⚠️  场景有轻微变化，建议更新"
                elif similarity > 0.70:
                    status = "⚠️  场景有明显变化，必须更新"
                else:
                    status = "❌ 场景完全不同，必须更新"
                print(f"  判断: {status}")
                print()
                
            except Exception as e:
                print(f"{name}")
                print(f"  ❌ 执行失败: {e}")
                print()
        
        print("="*70)
        print("测试总结")
        print("="*70)
        
        if results:
            fastest = min(results, key=lambda x: x['elapsed'])
            print(f"⚡ 最快方法: {fastest['name']} ({fastest['elapsed']*1000:.2f}ms)")
            
            print("\n💡 推荐方案:")
            print("   - 追求速度: 【方法5: 降采样快速对比】")
            print("   - 追求精度: 【方法2: 结构相似性(SSIM)】")
            print("   - 平衡方案: 【方法1: 直方图对比】")
            print("\n   阈值建议:")
            print("   - 相似度 > 0.95: 场景基本没变，可以复用缓存")
            print("   - 相似度 0.85-0.95: 有轻微变化，建议更新")
            print("   - 相似度 < 0.85: 场景变化明显，必须更新")
            print("\n   📊 实际测试结果分析:")
            for r in results:
                sim = r['similarity']
                if sim > 0.95:
                    conclusion = "✅ 推荐使用缓存"
                elif sim > 0.85:
                    conclusion = "⚠️  谨慎使用缓存"
                else:
                    conclusion = "❌ 不建议使用缓存"
                print(f"   {r['name']}: {sim:.4f} → {conclusion}")
        
        print("\n✅ 测试完成！节点将自动退出...")
        
        try:
            cv2.imwrite("/tmp/test_image1.jpg", self.image1)
            cv2.imwrite("/tmp/test_image2.jpg", self.image2)
            print("💾 测试图片已保存:")
            print("   图片1: /tmp/test_image1.jpg")
            print("   图片2: /tmp/test_image2.jpg")
        except:
            pass
        
        rclpy.shutdown()


def main():
    """主函数"""
    print("="*70)
    print("🎯 图像相似度测试工具（ROS2版本）")
    print("="*70)
    print("📡 自动从ROS2话题订阅图像")
    print("⏱️  捕获间隔: 15秒")
    print("🔍 测试5种相似度算法")
    print("="*70)
    print()
    
    rclpy.init()
    
    try:
        node = ImageSimilarityTester()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n👋 用户中断")
    except Exception as e:
        print(f"❌ 错误: {e}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
