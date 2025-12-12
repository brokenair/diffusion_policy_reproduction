import pyrealsense2 as rs
import numpy as np
import cv2

# 创建 pipeline
pipeline = rs.pipeline()
config = rs.config()

# 配置流（D415 支持）
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 启动
print("启动相机...")
pipeline.start(config)

try:
    for i in range(300):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow('RealSense D415', color_image)
            cv2.waitKey(1)
            if i == 0:
                print("✓ 相机工作正常！")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()