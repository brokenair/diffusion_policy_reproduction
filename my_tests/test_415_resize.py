import pyrealsense2 as rs
import numpy as np
import cv2

# 创建 pipeline
pipeline = rs.pipeline()
config = rs.config()

# 使用最小的原始分辨率，然后裁剪+resize 到 128x128
config.enable_stream(rs.stream.color, 320, 240, rs.format.bgr8, 30)

# 启动
print("启动相机...")
pipeline.start(config)

# 创建可调节大小的窗口
cv2.namedWindow('Original 320x240', cv2.WINDOW_NORMAL)
cv2.namedWindow('Center Crop 240x240', cv2.WINDOW_NORMAL)
cv2.namedWindow('Resized 128x128', cv2.WINDOW_NORMAL)

try:
    print("按 Q 退出...")
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if color_frame:
            # 获取原始图像 (320x240)
            color_image = np.asanyarray(color_frame.get_data())
            h, w = color_image.shape[:2]
            
            # 中心裁剪为正方形 (240x240)
            # 从 320x240 裁剪中心 240x240
            crop_size = min(h, w)  # 240
            start_x = (w - crop_size) // 2  # (320-240)//2 = 40
            start_y = (h - crop_size) // 2  # (240-240)//2 = 0
            cropped = color_image[start_y:start_y+crop_size, start_x:start_x+crop_size]
            
            # Resize 到 128x128（不会拉伸，因为已经是正方形）
            resized = cv2.resize(cropped, (128, 128))
            
            # 显示三个窗口对比
            cv2.imshow('Original 320x240', color_image)
            cv2.imshow('Center Crop 240x240', cropped)
            cv2.imshow('Resized 128x128', resized)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("相机已关闭")

