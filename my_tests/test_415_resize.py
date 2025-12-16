import pyrealsense2 as rs
import numpy as np
import cv2
import math

# ===== 裁剪框配置 =====
# 指定裁剪框的左上角和右下角坐标（相对于 640x480 图像）
CROP_TOP_LEFT = (20, 10)      # (x, y) 左上角坐标
CROP_BOTTOM_RIGHT = (620, 340)  # (x, y) 右下角坐标

# ===== 目标像素数配置 =====
TARGET_PIXELS = [20000, 15000, 10000]  # 三个目标像素数

# ===== 1.5w像素版本的中心裁剪配置 =====
# 对于15000像素版本，先resize到1.5w像素，然后在这个基础上进行中心裁剪
CENTER_CROP_FOR_15K_WIDTH = 140   # 在1.5w像素图像上中心裁剪的宽度
CENTER_CROP_FOR_15K_HEIGHT = 76  # 在1.5w像素图像上中心裁剪的高度

def calculate_resize_dimensions(crop_width, crop_height, target_pixels):
    """
    计算保持宽高比的目标尺寸，使总像素数接近目标值
    
    Args:
        crop_width: 裁剪后的宽度
        crop_height: 裁剪后的高度
        target_pixels: 目标像素数
    
    Returns:
        (new_width, new_height): 新的宽度和高度
    """
    aspect_ratio = crop_width / crop_height
    
    # 根据目标像素数和宽高比计算新尺寸
    # target_pixels = new_width * new_height
    # new_height = new_width / aspect_ratio
    # target_pixels = new_width * (new_width / aspect_ratio)
    # target_pixels = new_width^2 / aspect_ratio
    # new_width = sqrt(target_pixels * aspect_ratio)
    
    new_width = math.sqrt(target_pixels * aspect_ratio)
    new_height = new_width / aspect_ratio
    
    # 四舍五入到最近的整数
    new_width = int(round(new_width))
    new_height = int(round(new_height))
    
    # 确保至少为1像素
    new_width = max(1, new_width)
    new_height = max(1, new_height)
    
    return (new_width, new_height)

# 创建 pipeline
pipeline = rs.pipeline()
config = rs.config()

# 配置流：初始分辨率 640x480
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动
print("启动相机...")
pipeline.start(config)

# 验证裁剪框坐标
crop_x1, crop_y1 = CROP_TOP_LEFT
crop_x2, crop_y2 = CROP_BOTTOM_RIGHT

if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
    raise ValueError("裁剪框坐标无效：左上角必须在右下角的左上方")
if crop_x1 < 0 or crop_y1 < 0 or crop_x2 > 640 or crop_y2 > 480:
    print(f"警告：裁剪框坐标超出图像范围 (640x480)")
    print(f"  左上角: {CROP_TOP_LEFT}, 右下角: {CROP_BOTTOM_RIGHT}")

crop_width = crop_x2 - crop_x1
crop_height = crop_y2 - crop_y1
print(f"裁剪框尺寸: {crop_width}x{crop_height} (像素数: {crop_width * crop_height})")
print(f"裁剪框位置: 左上角 {CROP_TOP_LEFT}, 右下角 {CROP_BOTTOM_RIGHT}")

# 计算三个目标尺寸（所有版本都基于原始裁剪尺寸）
resize_dims = []
for target_pixels in TARGET_PIXELS:
    w, h = calculate_resize_dimensions(crop_width, crop_height, target_pixels)
    print(f"目标 {target_pixels} 像素 -> 尺寸: {w}x{h} (实际像素数: {w*h})")
    resize_dims.append((w, h, target_pixels))

# 对于1.5w像素版本，需要验证中心裁剪尺寸
# 先找到1.5w像素版本的尺寸
idx_15k = TARGET_PIXELS.index(15000)
w_15k, h_15k, _ = resize_dims[idx_15k]
print(f"\n1.5w像素版本尺寸: {w_15k}x{h_15k}")
print(f"中心裁剪配置: {CENTER_CROP_FOR_15K_WIDTH}x{CENTER_CROP_FOR_15K_HEIGHT}")

# 验证中心裁剪尺寸是否超出1.5w像素图像
if CENTER_CROP_FOR_15K_WIDTH > w_15k or CENTER_CROP_FOR_15K_HEIGHT > h_15k:
    print(f"警告：中心裁剪尺寸 {CENTER_CROP_FOR_15K_WIDTH}x{CENTER_CROP_FOR_15K_HEIGHT} 超出1.5w像素图像 {w_15k}x{h_15k}")
    center_crop_w = min(CENTER_CROP_FOR_15K_WIDTH, w_15k)
    center_crop_h = min(CENTER_CROP_FOR_15K_HEIGHT, h_15k)
    print(f"  已自动调整为: {center_crop_w}x{center_crop_h}")
else:
    center_crop_w = CENTER_CROP_FOR_15K_WIDTH
    center_crop_h = CENTER_CROP_FOR_15K_HEIGHT

# 创建窗口
cv2.namedWindow('Original 640x480', cv2.WINDOW_NORMAL)
cv2.namedWindow('Cropped', cv2.WINDOW_NORMAL)
for i, (w, h, pixels) in enumerate(resize_dims):
    cv2.namedWindow(f'Resized {pixels}px ({w}x{h})', cv2.WINDOW_NORMAL)
# 为1.5w像素版本添加中心裁剪窗口
cv2.namedWindow(f'Resized 15000px Center Crop ({center_crop_w}x{center_crop_h})', cv2.WINDOW_NORMAL)

try:
    print("\n按 Q 退出...")
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if color_frame:
            # 获取原始图像 (640x480)
            color_image = np.asanyarray(color_frame.get_data())
            h, w = color_image.shape[:2]
            
            # 在原始图像上绘制裁剪框（用于可视化，不叠加文字）
            vis_image = color_image.copy()
            cv2.rectangle(vis_image, CROP_TOP_LEFT, CROP_BOTTOM_RIGHT, (0, 255, 0), 2)
            
            # 执行裁剪
            cropped = color_image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # 生成三个不同大小的压缩图像
            resized_images = []
            resized_15k_full = None  # 保存完整的1.5w像素图像
            
            for idx, (new_w, new_h, target_pixels) in enumerate(resize_dims):
                # 所有版本都直接resize整个裁剪区域
                resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                resized_images.append((resized, target_pixels))
                
                # 保存1.5w像素版本的完整图像
                if target_pixels == 15000:
                    resized_15k_full = resized
            
            # 对1.5w像素版本进行中心裁剪
            center_cropped_15k = None
            if resized_15k_full is not None:
                h_15k, w_15k = resized_15k_full.shape[:2]
                
                # 计算中心裁剪的起始位置
                center_x = w_15k // 2
                center_y = h_15k // 2
                center_crop_x1 = center_x - center_crop_w // 2
                center_crop_y1 = center_y - center_crop_h // 2
                center_crop_x2 = center_crop_x1 + center_crop_w
                center_crop_y2 = center_crop_y1 + center_crop_h
                
                # 执行中心裁剪
                center_cropped_15k = resized_15k_full[center_crop_y1:center_crop_y2, center_crop_x1:center_crop_x2]
            
            # 显示所有窗口（不叠加文字）
            cv2.imshow('Original 640x480', vis_image)
            cv2.imshow('Cropped', cropped)
            for i, (resized, target_pixels) in enumerate(resized_images):
                w, h, _ = resize_dims[i]
                cv2.imshow(f'Resized {target_pixels}px ({w}x{h})', resized)
            
            # 显示1.5w像素版本的中心裁剪
            if center_cropped_15k is not None:
                cv2.imshow(f'Resized 15000px Center Crop ({center_crop_w}x{center_crop_h})', center_cropped_15k)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("相机已关闭")
