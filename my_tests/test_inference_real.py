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
CFG_PATH = "image_pusht_real_diffusion_policy_cnn.yaml"
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

# 图像处理
INFERENCE_RESOLUTION = (128, 128)  # 推理使用的分辨率
DISPLAY_RESOLUTION = (240, 240)  # 显示使用的分辨率

# 低通滤波参数
FILTER_TAU = 0.2  # 时间常数（秒），越小响应越快，但可能不够平滑


def resize_image(image, target_size):
    """
    将图像resize到目标尺寸（中心裁剪然后resize）。
    
    参考 test_415_resize.py 的逻辑：
    1. 中心裁剪为正方形（取最小边）
    2. Resize到目标尺寸
    """
    h, w = image.shape[:2]
    
    # 中心裁剪为正方形
    crop_size = min(h, w)
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    cropped = image[start_y:start_y+crop_size, start_x:start_x+crop_size]
    
    # Resize到目标尺寸
    resized = cv2.resize(cropped, target_size)
    
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
    
    # 初始化RealSense相机
    print("\n" + "=" * 60)
    print("初始化相机")
    print("=" * 60)
    pipeline = rs.pipeline()
    config_rs = rs.config()
    config_rs.enable_stream(rs.stream.color, 320, 240, rs.format.bgr8, 30)
    pipeline.start(config_rs)
    print("相机初始化完成")
    
    # 创建显示窗口
    cv2.namedWindow('Inference View (240x240)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Inference View (240x240)', 480, 480)
    
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
            frame_128 = resize_image(frame_rgb, INFERENCE_RESOLUTION)
            obs_buffer['image'].append(frame_128)
            obs_buffer['agent_pos'].append(current_ee_pos_xy.copy())
        time.sleep(0.1)
    
    print("  ✓ 观察缓冲区初始化完成")
    
    # 初始化低通滤波器（用于平滑目标位置）
    target_filter = LowPassFilter(
        tau=FILTER_TAU,
        dt=dt,
        initial_value=current_ee_pos_xy.copy()
    )
    
    # 推理循环
    print("\n" + "=" * 60)
    print("开始推理控制")
    print("=" * 60)
    print("按 Ctrl+C 停止推理")
    print("=" * 60)
    
    step_count = 0
    is_first_step = True
    
    # 如果记录，创建视频写入器
    video_writer = None
    if ENABLE_RECORDING:
        record_file = Path(RECORD_OUTPUT_DIR) / f"inference_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(record_file),
            fourcc,
            CONTROL_HZ,
            DISPLAY_RESOLUTION
        )
        print(f"开始记录到: {record_file}")
    
    try:
        while True:
            loop_start_time = time.time()
            
            # 读取相机图像
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                print("  警告: 无法读取相机图像")
                time.sleep(0.1)
                continue
            
            # 获取原始图像 (320x240, BGR格式)
            frame_bgr = np.asanyarray(color_frame.get_data())
            
            # 转换为RGB（用于推理）
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # 处理图像（用于推理和显示）
            frame_128 = resize_image(frame_rgb, INFERENCE_RESOLUTION)  # 推理用
            frame_240 = resize_image(frame_rgb, DISPLAY_RESOLUTION)  # 显示用
            
            # 获取当前末端执行器位置
            current_ee_pose = controller.get_end_effector_pose()
            current_ee_pos_xy = current_ee_pose.translation[:2]
            
            # 更新观察缓冲区（移除最旧的，添加最新的）
            obs_buffer['image'].pop(0)
            obs_buffer['image'].append(frame_128)
            obs_buffer['agent_pos'].pop(0)
            obs_buffer['agent_pos'].append(current_ee_pos_xy.copy())
            
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
            
            # 获取第一个动作（只执行第一步）
            action_first = action[0]  # (2,) - xy 目标位置
            target_xy_raw = action_first
            
            # 限制在工作空间内（与录制脚本一致的工作空间）
            WORKSPACE_CENTER = INIT_POSITION[:2]
            WORKSPACE_X_HALF = 0.15
            WORKSPACE_Y_HALF = 0.25
            target_xy_raw = np.clip(
                target_xy_raw,
                WORKSPACE_CENTER - np.array([WORKSPACE_X_HALF, WORKSPACE_Y_HALF]),
                WORKSPACE_CENTER + np.array([WORKSPACE_X_HALF, WORKSPACE_Y_HALF])
            )
            
            # 低通滤波平滑目标位置
            target_xy = target_filter.update(target_xy_raw)
            
            # 构建目标位姿（保持 z 和姿态不变）
            target_position_3d = np.array([target_xy[0], target_xy[1], INIT_POSITION[2]])
            target_pose = pin.SE3(target_orientation, target_position_3d)
            
            # 执行动作
            if is_first_step:
                # 第一步：规划到目标位置（3秒）
                print(f"步骤 {step_count}: 规划到目标位置 {target_xy}")
                q_current_ik = controller.get_joint_positions()
                q_target_ik, ik_error, success = controller.inverse_kinematics(
                    target_pose, q_init=q_current_ik,
                    fixed_iterations=10  # 固定迭代次数，实时控制
                )
                
                if success:
                    # 规划平滑轨迹（3秒）
                    duration = 3.0
                    q_traj, _, _ = minimal_jerk_trajectory(
                        q_current_ik, q_target_ik, duration, dt
                    )
                    
                    # 执行轨迹
                    for q in q_traj:
                        controller.move_to_joint_positions(
                            q,
                        )
                        time.sleep(dt / len(q_traj))
                    
                    is_first_step = False
                else:
                    print(f"  ✗ IK求解失败 (error: {ik_error:.6f})")
            else:
                # 后续步骤：直接 IK 控制
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
            
            # 显示图像
            img_240_bgr = cv2.cvtColor(frame_240, cv2.COLOR_RGB2BGR)
            
            # 在图像上绘制当前位置和目标位置
            # 将工作空间坐标转换为图像坐标（简化显示）
            # 假设图像中心对应工作空间中心
            img_h, img_w = frame_240.shape[:2]
            center_x, center_y = img_w // 2, img_h // 2
            
            # 当前位置（蓝色点）
            if len(obs_buffer['agent_pos']) > 0:
                current_pos_xy = obs_buffer['agent_pos'][-1]
                offset_x = int((current_pos_xy[0] - WORKSPACE_CENTER[0]) / WORKSPACE_X_HALF * (img_w // 2))
                offset_y = int((current_pos_xy[1] - WORKSPACE_CENTER[1]) / WORKSPACE_Y_HALF * (img_h // 2))
                img_x = center_x + offset_x
                img_y = center_y - offset_y  # y 轴翻转
                cv2.circle(img_240_bgr, (img_x, img_y), 5, (255, 0, 0), -1)  # 蓝色
            
            # 目标位置（红色点）
            offset_x = int((target_xy[0] - WORKSPACE_CENTER[0]) / WORKSPACE_X_HALF * (img_w // 2))
            offset_y = int((target_xy[1] - WORKSPACE_CENTER[1]) / WORKSPACE_Y_HALF * (img_h // 2))
            img_x = center_x + offset_x
            img_y = center_y - offset_y  # y 轴翻转
            cv2.circle(img_240_bgr, (img_x, img_y), 5, (0, 0, 255), -1)  # 红色
            
            # 添加文本信息
            cv2.putText(img_240_bgr, f"Step: {step_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_240_bgr, f"Target: [{target_xy[0]:.3f}, {target_xy[1]:.3f}]", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.imshow('Inference View (240x240)', img_240_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n用户按 'q' 退出")
                break
            
            # 记录视频
            if ENABLE_RECORDING and video_writer is not None:
                video_writer.write(img_240_bgr)
            
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

