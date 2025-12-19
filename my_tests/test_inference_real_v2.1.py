"""
实机 PushT 推理脚本。

使用训练好的 checkpoint 进行实机推理控制。

使用方法：
    1. 在文件开头配置 CKPT_PATH 和 CFG_PATH
    2. 运行: python my_tests/test_inference_real.py
"""

from __future__ import annotations

import sys
import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import dill
from pathlib import Path
from omegaconf import OmegaConf
import hydra

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_dict, get_real_obs_resolution
)
from lagrange_01.scripts.robot_controller import RobotController
from lagrange_01.utils.trajectory_utils import minimal_jerk_trajectory
from lagrange_01.utils.filter_utils import LowPassFilter
import pinocchio as pin


# ============================================================================
# 配置参数
# ============================================================================

# ===== 你需要改的两个路径 =====
CKPT_PATH = "outputs/2025-12-13/10-23-02/checkpoints/latest.ckpt"
CFG_PATH = "image_pusht_real_v1_diffusion_policy_cnn.yaml"
DEVICE = "cuda:0"

# 机器人配置（从录制脚本获取）
ROOT = ROOT_DIR / "lagrange_01"
CONFIG_PATH = ROOT / "config" / "motor_calibration.yaml"
URDF_PATH = ROOT / "assets" / "urdf" / "Lagrange_01_sim.urdf"
EE_FRAME = "tool_link"

# 初始位置（与录制脚本一致）
INIT_POSITION = np.array([-0.35, 0.0, 0.122])  # x, y, z (米)

# 控制频率（与录制脚本一致）
CONTROL_HZ = 10

# 推理配置开关
ENABLE_RECORDING = True  # 是否记录推理过程
RECORD_OUTPUT_DIR = "data/inference_recordings"  # 记录保存目录

# ===== 图像处理配置（与录制脚本一致）=====
# ROI 裁剪框配置（相对于 640x480 图像）
CROP_TOP_LEFT = (20, 10)      # (x, y) 左上角坐标
CROP_BOTTOM_RIGHT = (620, 340)  # (x, y) 右下角坐标

# 目标像素数配置（推理使用15000px，显示使用20000px）
TARGET_PIXELS = [20000, 15000, 10000]  # 三个目标像素数（2万、1.5万、1万）
INFERENCE_TARGET_PIXELS = 15000  # 推理使用的像素数
DISPLAY_TARGET_PIXELS = 20000    # 显示使用的像素数

# 低通滤波参数
FILTER_TAU = 0.2  # 时间常数（秒），越小响应越快，但可能不够平滑


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
    import math
    aspect_ratio = crop_width / crop_height
    
    # 根据目标像素数和宽高比计算新尺寸
    new_width = math.sqrt(target_pixels * aspect_ratio)
    new_height = new_width / aspect_ratio
    
    # 四舍五入到最近的整数
    new_width = int(round(new_width))
    new_height = int(round(new_height))
    
    # 确保至少为1像素
    new_width = max(1, new_width)
    new_height = max(1, new_height)
    
    return (new_width, new_height)


def process_image(frame_rgb, crop_x1, crop_y1, crop_x2, crop_y2, target_pixels):
    """
    处理图像：ROI裁剪 + 降分辨率（与录制脚本一致）
    
    Args:
        frame_rgb: RGB格式的原始图像 (640x480)
        crop_x1, crop_y1: ROI左上角坐标
        crop_x2, crop_y2: ROI右下角坐标
        target_pixels: 目标像素数
    
    Returns:
        numpy.ndarray: 处理后的图像 (H, W, C)
    """
    # 执行ROI裁剪
    cropped = frame_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # 计算目标尺寸
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    new_w, new_h = calculate_resize_dimensions(crop_width, crop_height, target_pixels)
    
    # 降采样到目标尺寸
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return resized


def load_policy(checkpoint_path, config_path, device='cuda:0'):
    """
    加载 checkpoint 和 policy。
    """
    print(f"加载 checkpoint: {checkpoint_path}")
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    # 如果提供了 config_path，使用它覆盖 cfg
    if config_path is not None:
        config_cfg = OmegaConf.load(config_path)
        # 合并配置
        cfg = OmegaConf.merge(cfg, config_cfg)
    
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # 获取 policy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.eval().to(device)
    
    # 设置推理参数（对于 diffusion 模型）
    if 'diffusion' in cfg.name:
        # 优先使用配置文件中的推理步数
        if hasattr(cfg.policy, 'num_inference_steps') and cfg.policy.num_inference_steps is not None:
            policy.num_inference_steps = cfg.policy.num_inference_steps
        else:
            # 如果没有配置，使用默认值（根据scheduler类型）
            if hasattr(cfg.policy.noise_scheduler, '_target_'):
                if 'ddim' in cfg.policy.noise_scheduler._target_.lower():
                    policy.num_inference_steps = 16  # DDIM 默认值
                else:
                    policy.num_inference_steps = cfg.policy.noise_scheduler.num_train_timesteps  # DDPM 使用训练步数
            else:
                policy.num_inference_steps = 16  # 兜底默认值
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    
    # 获取 normalizer
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)

    # 重要：设置 normalizer 后需要再次移动到设备，确保 normalizer 的参数也在 GPU 上
    policy.to(device)

    return policy, cfg, device


def main():
    """
    实机 PushT 推理主程序。
    """
    
    # 使用文件开头配置的路径
    checkpoint = CKPT_PATH
    config = CFG_PATH
    device = DEVICE
    
    # 创建记录目录
    if ENABLE_RECORDING:
        record_dir = Path(RECORD_OUTPUT_DIR)
        record_dir.mkdir(parents=True, exist_ok=True)
        print(f"记录目录: {record_dir}")
    
    # 加载 policy
    print("\n" + "=" * 60)
    print("加载模型")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint}")
    print(f"Config: {config}")
    print(f"Device: {device}")
    policy, cfg, device = load_policy(checkpoint, config, device)
    # 确保 device 是 torch.device 对象
    if not isinstance(device, torch.device):
        device = torch.device(device)
    print(f"Device type: {type(device)}, Device value: {device}")
    n_obs_steps = cfg.n_obs_steps
    print(f"n_obs_steps: {n_obs_steps}")
    print(f"action_dim: {policy.action_dim}")
    print(f"horizon: {policy.horizon}")
    
    # 初始化RealSense相机（640x480，与录制脚本一致）
    print("\n" + "=" * 60)
    print("初始化相机")
    print("=" * 60)
    pipeline = rs.pipeline()
    config_rs = rs.config()
    config_rs.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config_rs)
    print("相机初始化完成 (640x480)")
    
    # 计算裁剪和显示尺寸
    crop_x1, crop_y1 = CROP_TOP_LEFT
    crop_x2, crop_y2 = CROP_BOTTOM_RIGHT
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    
    # 计算显示图像尺寸
    display_w, display_h = calculate_resize_dimensions(crop_width, crop_height, DISPLAY_TARGET_PIXELS)
    print(f"裁剪框尺寸: {crop_width}x{crop_height}")
    print(f"推理图像尺寸: {calculate_resize_dimensions(crop_width, crop_height, INFERENCE_TARGET_PIXELS)}")
    print(f"显示图像尺寸: {display_w}x{display_h}")
    
    # 创建显示窗口
    cv2.namedWindow('Inference View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Inference View', display_w * 2, display_h * 2)
    
    # 初始化机器人控制器
    print("\n" + "=" * 60)
    print("初始化机器人控制器")
    print("=" * 60)
    controller = RobotController.from_config(
        CONFIG_PATH,
        urdf_path=URDF_PATH,
        end_effector=EE_FRAME
    )
    controller.connect()
    
    # 重要：先失能电机，才能读取到初始位姿
    print("失能电机以读取初始位姿...")
    controller.disable_all()
    time.sleep(0.5)
    
    # 读取当前关节角度和末端位姿
    print("读取当前位姿...")
    q_current = controller.get_joint_positions(request_update=True)
    current_pose = controller.get_end_effector_pose()
    current_position = current_pose.translation
    
    print(f"  当前关节角度: {q_current}")
    print(f"  当前位置: {current_position}")
    
    # 目标位置和姿态（与录制脚本一致）
    target_position = INIT_POSITION.copy()
    target_orientation = np.array([
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    
    # 创建目标位姿
    target_pose = pin.SE3(target_orientation, target_position)
    
    # 规划到初始位置
    print(f"\n规划到初始位置: {target_position}")
    print("  求解IK...")
    q_target, ik_error, success = controller.inverse_kinematics(
        target_pose, q_init=q_current,
        max_iterations=100, tolerance=1e-4
    )
    if not success:
        print(f"  ✗ IK求解失败 (error: {ik_error:.6f})")
        return 1
    
    print(f"  ✓ IK求解成功 (error: {ik_error:.6f})")
    
    # 规划平滑轨迹
    print("  规划轨迹...")
    dt = 1.0 / CONTROL_HZ
    duration = 3.0  # 移动到初始位置的时间
    q_traj, _, _ = minimal_jerk_trajectory(q_current, q_target, duration, dt)
    print(f"  ✓ 轨迹规划完成 ({len(q_traj)}个点)")
    
    # 执行轨迹
    print("  执行轨迹...")
    for q in q_traj:
        controller.move_to_joint_positions(
            q,
            limit_speed=[1.2, 1.2, 2.0, 1.0, 1.0, 1.5],
            acceleration=[4.0, 4.0, 5.0, 4.0, 4.0, 4.0]
        )
        time.sleep(dt)
    
    print("  ✓ 到达初始位置")
    
    # 初始化观察缓冲区
    obs_buffer = {
        'image': [],  # 存储 n_obs_steps 帧图像
        'agent_pos': []  # 存储 n_obs_steps 帧位置
    }
    
    # 填充初始观察缓冲区（读取 n_obs_steps 帧相同图像和位置）
    print(f"\n初始化观察缓冲区 (n_obs_steps={n_obs_steps})...")
    current_ee_pose = controller.get_end_effector_pose()
    current_ee_pos_xy = current_ee_pose.translation[:2]
    
    for _ in range(n_obs_steps):
        # 读取一帧图像
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            frame_bgr = np.asanyarray(color_frame.get_data())
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # 使用新的图像处理逻辑（ROI裁剪 + 降采样到15000px）
            frame_processed = process_image(frame_rgb, crop_x1, crop_y1, crop_x2, crop_y2, INFERENCE_TARGET_PIXELS)
            obs_buffer['image'].append(frame_processed)
            obs_buffer['agent_pos'].append(current_ee_pos_xy.copy())
        time.sleep(0.1)
    
    print("  ✓ 观察缓冲区初始化完成")
    
    # 初始化低通滤波器（用于平滑目标位置）
    target_filter = LowPassFilter(
        tau=FILTER_TAU,
        dt=dt,
        initial_value=current_ee_pos_xy.copy()
    )
    
    # 工作空间配置（与录制脚本一致）
    WORKSPACE_CENTER = INIT_POSITION[:2]
    WORKSPACE_X_HALF = 0.15
    WORKSPACE_Y_HALF = 0.25
    
    # 推理循环
    print("\n" + "=" * 60)
    print("开始推理控制")
    print("=" * 60)
    print("推理频率: 2Hz (每0.5秒)")
    print("控制频率: 10Hz (每0.1秒)")
    print("每次推理执行5个点")
    print("按 Ctrl+C 停止推理")
    print("=" * 60)
    
    step_count = 0
    action_queue = []  # action执行队列
    last_inference_time = time.time()  # 上次推理时间
    inference_interval = 0.5  # 推理间隔（秒）
    actions_per_inference = 5  # 每次推理执行的action数量
    latest_actions_clipped = None  # 最新的推理结果用于显示
    
    # 如果记录，创建视频写入器
    video_writer = None
    if ENABLE_RECORDING:
        record_file = Path(RECORD_OUTPUT_DIR) / f"inference_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(record_file),
            fourcc,
            CONTROL_HZ,
            (display_w, display_h)
        )
        print(f"开始记录到: {record_file}")
    
    try:
        while True:
            loop_start_time = time.time()
            current_time = time.time()
            
            # 读取相机图像（每次循环都读取，用于更新观察和显示）
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                print("  警告: 无法读取相机图像")
                time.sleep(0.1)
                continue
            
            # 获取原始图像 (640x480, BGR格式)
            frame_bgr = np.asanyarray(color_frame.get_data())
            
            # 转换为RGB（用于推理）
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # 处理图像（用于推理和显示）
            frame_inference = process_image(frame_rgb, crop_x1, crop_y1, crop_x2, crop_y2, INFERENCE_TARGET_PIXELS)  # 推理用（15000px）
            frame_display = process_image(frame_rgb, crop_x1, crop_y1, crop_x2, crop_y2, DISPLAY_TARGET_PIXELS)   # 显示用（20000px）
            
            # 获取当前末端执行器位置
            current_ee_pose = controller.get_end_effector_pose()
            current_ee_pos_xy = current_ee_pose.translation[:2]
            
            # 更新观察缓冲区（移除最旧的，添加最新的）
            obs_buffer['image'].pop(0)
            obs_buffer['image'].append(frame_inference)
            obs_buffer['agent_pos'].pop(0)
            obs_buffer['agent_pos'].append(current_ee_pos_xy.copy())
            
            # 检查是否需要推理（每0.5秒一次，或队列为空）
            if len(action_queue) == 0 or (current_time - last_inference_time) >= inference_interval:
                # 进行推理
                print(f"  推理中... (队列长度: {len(action_queue)})")
                inference_start_time = time.time()
                
                # 构建观察字典（格式: (T, H, W, C) 或 (T, D)）
                # 注意：get_real_obs_dict 期望输入格式为 (T, H, W, C)，uint8 类型
                obs_dict_np = {
                    'image': np.stack(obs_buffer['image']),  # (n_obs_steps, H, W, C) uint8
                    'agent_pos': np.stack(obs_buffer['agent_pos'])  # (n_obs_steps, 2) float
                }
                
                # 使用 real_inference_util 转换观察格式（会转换为 float32 并调整维度）
                obs_dict_np = get_real_obs_dict(obs_dict_np, cfg.task.shape_meta)
                
                # 转换为 torch tensor 并移动到设备
                # 确保 device 是 torch.device 对象
                if not isinstance(device, torch.device):
                    device = torch.device(device)
                
                # 转换为 torch tensor 并移动到设备
                obs_dict = dict_apply(
                    obs_dict_np,
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device)  # 添加 batch 维度并移动到设备
                )
                
                # 递归函数，确保所有 tensor 都在正确的设备上
                def to_device_recursive(obj, dev):
                    if isinstance(obj, torch.Tensor):
                        if obj.device != dev:
                            return obj.to(dev)
                        return obj
                    elif isinstance(obj, dict):
                        return {k: to_device_recursive(v, dev) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return type(obj)(to_device_recursive(item, dev) for item in obj)
                    else:
                        return obj
                
                # 双重检查：递归确保所有数据都在正确的设备上
                obs_dict = to_device_recursive(obs_dict, device)
                
                # 推理
                with torch.no_grad():
                    result = policy.predict_action(obs_dict)
                    action = result['action'][0].detach().cpu().numpy()  # 移除 batch 维度
                
                # 限制所有目标在工作空间内
                actions_clipped = np.clip(
                    action,
                    WORKSPACE_CENTER - np.array([WORKSPACE_X_HALF, WORKSPACE_Y_HALF]),
                    WORKSPACE_CENTER + np.array([WORKSPACE_X_HALF, WORKSPACE_Y_HALF])
                )
                
                # 将前5个action加入队列
                actions_to_execute = actions_clipped[:actions_per_inference]
                action_queue.extend(actions_to_execute)
                latest_actions_clipped = actions_clipped  # 保存用于显示
                
                last_inference_time = current_time
                inference_time = time.time() - inference_start_time
                print(f"  ✓ 推理完成 ({inference_time*1000:.1f}ms), 队列长度: {len(action_queue)}")
            
            # 从队列中取出一个action执行
            if len(action_queue) > 0:
                target_xy_raw = action_queue.pop(0)
                
                # 低通滤波平滑目标位置
                target_xy = target_filter.update(target_xy_raw)
                
                # 构建目标位姿（保持 z 和姿态不变）
                target_position_3d = np.array([target_xy[0], target_xy[1], INIT_POSITION[2]])
                target_pose = pin.SE3(target_orientation, target_position_3d)
                
                # 执行动作：直接 IK 控制
                q_current_ik = controller.get_joint_positions()
                q_target_ik, ik_error, success = controller.inverse_kinematics(
                    target_pose, q_init=q_current_ik,
                    fixed_iterations=10
                )
                
                if success:
                    controller.move_to_joint_positions(
                        q_target_ik,
                    )
                else:
                    print(f"  ✗ IK求解失败 (error: {ik_error:.6f})")
            else:
                # 队列为空，使用当前位置作为目标（保持不动）
                target_xy = current_ee_pos_xy
            
            # 显示图像
            img_display_bgr = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)
            
            # 在图像上绘制当前位置和目标位置
            # 将工作空间坐标转换为图像坐标（简化显示）
            # 假设图像中心对应工作空间中心
            img_h, img_w = frame_display.shape[:2]
            center_x, center_y = img_w // 2, img_h // 2
            
            # 当前位置（蓝色点）
            offset_x = int((current_ee_pos_xy[0] - WORKSPACE_CENTER[0]) / WORKSPACE_X_HALF * (img_w // 2))
            offset_y = int((current_ee_pos_xy[1] - WORKSPACE_CENTER[1]) / WORKSPACE_Y_HALF * (img_h // 2))
            img_x = center_x + offset_x
            img_y = center_y - offset_y  # y 轴翻转
            cv2.circle(img_display_bgr, (img_x, img_y), 5, (255, 0, 0), -1)  # 蓝色
            
            # 绘制所有目标位置（红色，由深到浅）
            if latest_actions_clipped is not None:
                num_targets_to_show = len(latest_actions_clipped)
                # 颜色从深红(255)到浅红(50)，BGR格式
                color_start = 255
                color_end = 50
                radius_start = 6
                radius_end = 1
                for i, target_xy_i in enumerate(latest_actions_clipped):
                    # 计算颜色：从深到浅
                    if num_targets_to_show > 1:
                        color_intensity = int(color_start - (color_start - color_end) * i / (num_targets_to_show - 1))
                    else:
                        color_intensity = color_start
                    color = (0, 0, color_intensity)  # BGR格式，红色
                    
                    # 转换为图像坐标
                    offset_x = int((target_xy_i[0] - WORKSPACE_CENTER[0]) / WORKSPACE_X_HALF * (img_w // 2))
                    offset_y = int((target_xy_i[1] - WORKSPACE_CENTER[1]) / WORKSPACE_Y_HALF * (img_h // 2))
                    img_x = center_x + offset_x
                    img_y = center_y - offset_y  # y 轴翻转
                    
                    # 绘制目标点，第一个最大，后续逐渐变小
                    if num_targets_to_show > 1:
                        radius = int(radius_start - (radius_start - radius_end) * i / (num_targets_to_show - 1))
                    else:
                        radius = radius_start
                    radius = max(1, radius)  # 确保至少为1
                    cv2.circle(img_display_bgr, (img_x, img_y), radius, color, -1)
            
            # 添加文本信息
            cv2.putText(img_display_bgr, f"Step: {step_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_display_bgr, f"Target: [{target_xy[0]:.3f}, {target_xy[1]:.3f}]", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(img_display_bgr, f"Queue: {len(action_queue)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow('Inference View', img_display_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n用户按 'q' 退出")
                break
            
            # 记录视频
            if ENABLE_RECORDING and video_writer is not None:
                video_writer.write(img_display_bgr)
            
            step_count += 1
            
            # 控制频率
            elapsed = time.time() - loop_start_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"  警告: 控制循环超时 ({elapsed:.3f}s > {dt:.3f}s)")
    
    except KeyboardInterrupt:
        print("\n\n收到 Ctrl+C，停止推理")
    
    finally:
        # 清理资源
        print("\n清理资源...")
        pipeline.stop()
        cv2.destroyAllWindows()
        
        if ENABLE_RECORDING and video_writer is not None:
            video_writer.release()
            if record_file is not None:
                print(f"记录已保存到: {record_file}")
        
        print("完成")


if __name__ == '__main__':
    main()

