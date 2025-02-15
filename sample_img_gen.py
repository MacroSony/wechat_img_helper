import os
import cv2
import numpy as np


def generate_16_9_images():
    # 配置参数
    output_dir = "output_images"  # 输出目录
    num_images = 36              # 生成图片数量
    base_width = 1280            # 基准宽度（保持16:9比例）
    
    # 计算对应高度（确保整除）
    height = base_width * 9 // 16
    img_size = (base_width, height)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成36张不同颜色的图片
    for i in range(num_images):
        # 创建渐变色背景
        gradient = np.zeros((height, base_width, 3), dtype=np.uint8)
        
        # 生成HSV颜色（色相均匀分布）
        hue = int((i * 180) / num_images)  # OpenCV的H范围是0-180
        saturation = 255
        value = 255
        
        # 纯色图片
        hsv_color = np.uint8([[[hue, saturation, value]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        gradient[:, :] = bgr_color

        # 添加文字信息
        text = f"Image {i+1}/{num_images}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = base_width / 1000 + 0.5  # 动态字体大小
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # 文字居中位置
        text_x = (base_width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        # 添加文字阴影效果
        cv2.putText(gradient, text, (text_x+2, text_y+2), font,
                    font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        cv2.putText(gradient, text, (text_x, text_y), font,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # 保存图片
        filename = os.path.join(output_dir, f"image_{i+1:02d}.jpg")
        cv2.imwrite(filename, gradient)
    
    print(f"成功生成 {num_images} 张 16:9 图片到 {output_dir} 目录")
    
if __name__ == "__main__":
    generate_16_9_images()