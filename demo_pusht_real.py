"""
实机 PushT 数据采集脚本。

使用真实机器人录制演示数据，通过鼠标控制末端执行器在 xy 平面移动。

使用方法：
    python demo_pusht_real.py -o data/pusht_real_demo.zarr

操作说明：
    - 鼠标窗口：点击一次开始控制，再点击一次结束 episode
    - 控制时移动鼠标即可控制末端执行器在 xy 平面移动
    - 按 'Q' 退出程序
    - 按 'R' 重试当前 episode
    - 按 'S' 保存当前 episode（即使未完成）
    - 按 'P' 暂停/继续
"""

from __future__ import annotations

import sys
import time
import threading
import cv2
import numpy as np
import click
import pyrealsense2 as rs
from pathlib import Path

from diffusion_policy.common.replay_buffer import ReplayBuffer
from lagrange_01.scripts.robot_controller import RobotController
from lagrange_01.utils.trajectory_utils import minimal_jerk_trajectory
import pinocchio as pin


# ============================================================================
# 配置参数
# ============================================================================

ROOT = Path(__file__).resolve().parent / "lagrange_01"
CONFIG_PATH = ROOT / "config" / "motor_calibration.yaml"
URDF_PATH = ROOT / "assets" / "urdf" / "Lagrange_01_sim.urdf"
EE_FRAME = "tool_link"

# 初始位置（目标位置）
INIT_POSITION = np.array([-0.35, 0.0, 0.12])  # x, y, z (米)

# 工作空间参数（相对于初始位置）
WORKSPACE_CENTER = INIT_POSITION[:2]  # xy 平面中心
WORKSPACE_X_HALF = 0.15  # x 方向半宽度 (总宽度 0.3)
WORKSPACE_Y_HALF = 0.25  # y 方向半宽度 (总宽度 0.5)

# 视频分辨率
VIDEO_RESOLUTIONS = {
    '240x240': (240, 240),
    '180x180': (180, 180),
    '128x128': (128, 128),
}


class MouseTargetWindow:
    """
    鼠标目标窗口（Toggle 模式）。
    
    - 点击一次：开始控制（目标设为末端执行器当前位置）
    - 控制时：移动鼠标即可更新目标（不需要按下）
    - 再次点击：停止控制（结束 episode）
    """
    
    def __init__(self, workspace_center, workspace_half_extent, window_size=800):
        self.window_size = window_size
        self.workspace_center = np.asarray(workspace_center, dtype=float)
        self.workspace_half_extent = np.asarray(workspace_half_extent, dtype=float)
        self._target = np.zeros(2, dtype=float)  # 当前目标（xy平面）
        self._is_controlling = False  # 是否处于控制状态（toggle）
        self._control_start_pos = np.zeros(2, dtype=float)  # 开始控制时的末端位置
        self._control_start_mouse = np.zeros(2, dtype=float)  # 开始控制时的鼠标位置
        self._current_ee_pos = np.zeros(2, dtype=float)  # 当前末端位置（由外部更新）
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._init_error = None
        self._should_close = False
        self.root = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        
        if not self._ready.wait(timeout=5):
            raise RuntimeError("鼠标窗口初始化超时")
        if self._init_error is not None:
            raise self._init_error
    
    def set_current_pos(self, pos_xy):
        """设置末端执行器当前位置（由环境调用）"""
        with self._lock:
            self._current_ee_pos = np.array(pos_xy, dtype=float)
    
    def toggle_control(self, mouse_plane_pos):
        """切换控制状态（由鼠标点击调用）"""
        with self._lock:
            if not self._is_controlling:
                # 开始控制：目标设为末端执行器当前位置
                self._is_controlling = True
                self._control_start_pos = self._current_ee_pos.copy()
                self._control_start_mouse = np.array(mouse_plane_pos, dtype=float)
                self._target = self._control_start_pos.copy()
            else:
                # 停止控制：结束 episode
                self._is_controlling = False
            return self._is_controlling
    
    def update_target(self, mouse_plane_pos):
        """更新目标（由鼠标移动调用）"""
        with self._lock:
            if self._is_controlling:
                # 目标 = 末端初始位置 + 鼠标移动量
                delta = mouse_plane_pos - self._control_start_mouse
                self._target = self._control_start_pos + delta
                # 限制在工作空间内
                self._target = np.clip(
                    self._target,
                    self.workspace_center - self.workspace_half_extent,
                    self.workspace_center + self.workspace_half_extent
                )
    
    def stop_control_external(self):
        """外部调用：强制停止控制"""
        with self._lock:
            self._is_controlling = False
    
    def get_target(self):
        """获取目标位置（xy平面）"""
        with self._lock:
            return self._target.copy()
    
    def is_controlling(self):
        """是否正在控制"""
        with self._lock:
            return self._is_controlling
    
    def _run(self):
        try:
            import tkinter as tk
        except Exception as exc:
            self._init_error = exc
            self._ready.set()
            return
        
        self.root = tk.Tk()
        self.root.title("Mouse Control - CLICK to toggle")
        
        canvas = tk.Canvas(
            self.root,
            width=self.window_size,
            height=self.window_size,
            bg="#111111",
            highlightthickness=0,
        )
        canvas.pack()
        
        # 绘制边框
        canvas.create_rectangle(
            2, 2, self.window_size - 2, self.window_size - 2,
            outline="#444"
        )
        
        # 绘制网格
        for i in range(1, 8):
            pos = i * self.window_size // 8
            canvas.create_line(pos, 0, pos, self.window_size, fill="#333")
            canvas.create_line(0, pos, self.window_size, pos, fill="#333")
        
        # 中心十字
        cx, cy = self.window_size // 2, self.window_size // 2
        canvas.create_line(cx - 20, cy, cx + 20, cy, fill="#666", width=2)
        canvas.create_line(cx, cy - 20, cx, cy + 20, fill="#666", width=2)
        
        # 目标指示器
        self.cross_h = canvas.create_line(0, 0, 0, 0, fill="#888888", width=2)
        self.cross_v = canvas.create_line(0, 0, 0, 0, fill="#888888", width=2)
        
        # 状态文本
        self.status_text = canvas.create_text(
            self.window_size // 2, 25,
            fill="#888888",
            text="Click to start controlling",
            font=("Arial", 14, "bold")
        )
        
        # 坐标文本
        self.coord_text = canvas.create_text(
            self.window_size // 2, self.window_size - 20,
            fill="#cccccc",
            text="(0.00, 0.00) m",
            font=("Arial", 12)
        )
        
        self.canvas = canvas
        
        # 绑定鼠标事件
        canvas.bind("<ButtonPress-1>", self._on_click)
        canvas.bind("<Motion>", self._on_motion)
        
        self._ready.set()
        
        def check_close():
            if self._should_close:
                self.root.quit()
            else:
                self.root.after(100, check_close)
        
        self.root.after(100, check_close)
        self.root.mainloop()
        
        try:
            self.root.withdraw()
        except:
            pass
    
    def _canvas_to_plane(self, x_pixel, y_pixel):
        """将画布坐标转换为平面坐标（相对于工作空间中心）"""
        x_clamped = min(max(x_pixel, 0.0), self.window_size)
        y_clamped = min(max(y_pixel, 0.0), self.window_size)
        norm_x = x_clamped / self.window_size * 2.0 - 1.0
        norm_y = 1.0 - (y_clamped / self.window_size * 2.0)
        # 转换为实际坐标（相对于工作空间中心）
        plane_pos = np.array([
            norm_x * self.workspace_half_extent[0],
            norm_y * self.workspace_half_extent[1]
        ], dtype=float)
        # 加上工作空间中心
        return self.workspace_center + plane_pos
    
    def _plane_to_canvas(self, plane_pos):
        """将平面坐标转换为画布坐标"""
        # 相对于工作空间中心
        rel_pos = plane_pos - self.workspace_center
        norm_x = rel_pos[0] / self.workspace_half_extent[0]
        norm_y = rel_pos[1] / self.workspace_half_extent[1]
        x_pixel = (norm_x + 1.0) / 2.0 * self.window_size
        y_pixel = (1.0 - norm_y) / 2.0 * self.window_size
        return x_pixel, y_pixel
    
    def _update_display(self, target_pos, is_controlling):
        """更新显示"""
        x, y = self._plane_to_canvas(target_pos)
        
        if is_controlling:
            self.canvas.coords(self.cross_h, 0, y, self.window_size, y)
            self.canvas.coords(self.cross_v, x, 0, x, self.window_size)
            self.canvas.itemconfigure(self.cross_h, fill="#ff3333", width=3)
            self.canvas.itemconfigure(self.cross_v, fill="#ff3333", width=3)
            self.canvas.itemconfigure(self.status_text, text="● CONTROLLING (click to end episode)", fill="#ff3333")
        else:
            self.canvas.coords(self.cross_h, 0, y, self.window_size, y)
            self.canvas.coords(self.cross_v, x, 0, x, self.window_size)
            self.canvas.itemconfigure(self.cross_h, fill="#888888", width=2)
            self.canvas.itemconfigure(self.cross_v, fill="#888888", width=2)
            self.canvas.itemconfigure(self.status_text, text="○ Click to start", fill="#888888")
        
        self.canvas.itemconfigure(self.coord_text, text=f"Target: ({target_pos[0]:.3f}, {target_pos[1]:.3f}) m")
    
    def _on_click(self, event):
        """鼠标点击：切换控制状态"""
        mouse_pos = self._canvas_to_plane(event.x, event.y)
        is_controlling = self.toggle_control(mouse_pos)
        with self._lock:
            target = self._target.copy()
        self._update_display(target, is_controlling)
    
    def _on_motion(self, event):
        """鼠标移动：如果在控制状态则更新目标"""
        mouse_pos = self._canvas_to_plane(event.x, event.y)
        self.update_target(mouse_pos)
        with self._lock:
            is_controlling = self._is_controlling
            target = self._target.copy()
        self._update_display(target, is_controlling)
    
    def close(self):
        """线程安全地关闭窗口"""
        self._should_close = True
        self._thread.join(timeout=0.5)


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


@click.command()
@click.option('-o', '--output', required=True, help='输出 zarr 文件路径')
@click.option('-hz', '--control_hz', default=10, type=int, help='控制频率')
@click.option('--mouse-window', default=800, type=int, help='鼠标控制窗口大小')
def main(output, control_hz, mouse_window):
    """
    采集实机 PushT 任务的演示数据。
    """
    
    # 创建 replay buffer
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')
    
    # 初始化RealSense相机（参考 test_415_resize.py）
    print("初始化RealSense相机...")
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 使用最小的原始分辨率，然后裁剪+resize
    # 320x240 可以中心裁剪为 240x240，然后resize到其他尺寸
    config.enable_stream(rs.stream.color, 320, 240, rs.format.bgr8, 30)
    
    # 启动相机
    print("启动相机...")
    pipeline.start(config)
    
    # 创建显示窗口（显示240x240图像）
    cv2.namedWindow('Recording View (240x240)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Recording View (240x240)', 480, 480)  # 放大显示，更清晰
    
    # 初始化机器人控制器
    print("初始化机器人控制器...")
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
    
    # 目标位置和姿态（使用 circle_demo_real.py 中的期望姿态）
    target_position = INIT_POSITION.copy()
    # 期望的末端执行器姿态（向下）
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
    dt = 1.0 / control_hz
    duration = 3.0  # 移动到初始位置的时间
    q_traj, _, _ = minimal_jerk_trajectory(q_current, q_target, duration, dt)
    print(f"  ✓ 轨迹规划完成 ({len(q_traj)}个点)")
    
    # 执行轨迹
    print("  执行轨迹...")
    for q in q_traj:
        controller.move_to_joint_positions(q, limit_speed=0.5, acceleration=3.0)
        time.sleep(dt)
    
    print("  ✓ 到达初始位置")
    
    # 创建鼠标控制窗口
    mouse_window = MouseTargetWindow(
        workspace_center=WORKSPACE_CENTER,
        workspace_half_extent=[WORKSPACE_X_HALF, WORKSPACE_Y_HALF],
        window_size=mouse_window
    )
    
    # 控制频率
    dt = 1.0 / control_hz
    
    print("\n" + "=" * 60)
    print("实机 PushT 数据采集")
    print("=" * 60)
    print(f"配置: {control_hz}Hz -> {output}")
    print(f"工作空间: x=[{WORKSPACE_CENTER[0]-WORKSPACE_X_HALF:.2f}, {WORKSPACE_CENTER[0]+WORKSPACE_X_HALF:.2f}], "
          f"y=[{WORKSPACE_CENTER[1]-WORKSPACE_Y_HALF:.2f}, {WORKSPACE_CENTER[1]+WORKSPACE_Y_HALF:.2f}]")
    print("")
    print("操作：")
    print("  鼠标窗口: 点击开始控制，再点击结束 episode")
    print("  Q - 退出 | R - 重试 | S - 保存 | P - 暂停")
    print("=" * 60)
    
    # 设置终端为非阻塞输入
    import select
    import termios
    import tty
    
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    
    def get_key():
        """非阻塞获取按键"""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None
    
    try:
        # Episode 循环
        while True:
            episode = list()
            
            # 使用 episode 数量作为种子
            seed = replay_buffer.n_episodes
            print(f'\n开始 episode {seed}')
            
            # 重置鼠标窗口
            current_ee_pose = controller.get_end_effector_pose()
            current_ee_pos_xy = current_ee_pose.translation[:2]
            mouse_window.set_current_pos(current_ee_pos_xy)
            with mouse_window._lock:
                mouse_window._target = current_ee_pos_xy.copy()
                mouse_window._is_controlling = False
            
            print(f"  当前位置: ({current_ee_pos_xy[0]:.3f}, {current_ee_pos_xy[1]:.3f})")
            print(f"  点击鼠标窗口开始控制...")
            
            # 状态变量
            retry = False
            pause = False
            done = False
            force_save = False
            quit_flag = False
            step_count = 0
            
            # Step 循环
            while not done:
                # 处理按键
                key = get_key()
                if key:
                    if key.lower() == 'q':
                        quit_flag = True
                        break
                    elif key.lower() == 'r':
                        retry = True
                        break
                    elif key.lower() == 's':
                        force_save = True
                        done = True
                    elif key.lower() == 'p':
                        pause = not pause
                
                if pause:
                    time.sleep(0.1)
                    continue
                
                # 更新鼠标窗口中的末端执行器当前位置
                current_ee_pose = controller.get_end_effector_pose()
                current_ee_pos_xy = current_ee_pose.translation[:2]
                mouse_window.set_current_pos(current_ee_pos_xy)
                
                # 获取目标位置（xy平面）
                target_xy = mouse_window.get_target()
                is_recording = mouse_window.is_controlling()
                
                # 检查是否结束 episode（再次点击停止控制）
                if not is_recording and step_count > 0:
                    # 控制已停止，结束 episode
                    done = True
                    break
                
                # 读取RealSense相机图像（无论是否录制都读取，用于显示）
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                        print("  警告: 无法读取相机图像")
                        time.sleep(0.1)
                        continue
                    
                # 获取原始图像 (320x240, BGR格式)
                frame_bgr = np.asanyarray(color_frame.get_data())
                
                # 转换为RGB（用于数据处理和保存）
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    
                    # 生成多个分辨率的图像
                    images = {}
                    for name, size in VIDEO_RESOLUTIONS.items():
                        images[name] = resize_image(frame_rgb, size)
                    
                # 显示240x240图像（实时显示，无论是否在录制）
                img_240_display = images['240x240'].copy()
                # 转换为BGR用于OpenCV显示
                img_240_bgr = cv2.cvtColor(img_240_display, cv2.COLOR_RGB2BGR)
                cv2.imshow('Recording View (240x240)', img_240_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    quit_flag = True
                    break
                
                # 只有在控制时才记录数据
                if is_recording:
                    # 获取当前状态（末端执行器位置和姿态）
                    current_ee_pose = controller.get_end_effector_pose()
                    current_pos = current_ee_pose.translation
                    current_rot = current_ee_pose.rotation
                    
                    # 状态：末端执行器位置 (3) + 姿态（旋转矩阵的某些元素或四元数）
                    # 为了简化，我们使用位置和旋转矩阵的前两列（6个元素）
                    state = np.concatenate([
                        current_pos,  # 3
                        current_rot[:, 0],  # 3 (旋转矩阵第一列)
                        current_rot[:, 1],  # 3 (旋转矩阵第二列)
                    ])  # 总共9维
                    
                    # 动作：目标位置（xy平面，z保持固定）
                    action = np.array([target_xy[0], target_xy[1]], dtype=np.float32)
                    
                    # 记录数据
                    data = {
                        'img_240x240': images['240x240'],
                        'img_180x180': images['180x180'],
                        'img_128x128': images['128x128'],
                        'state': np.float32(state),
                        'action': np.float32(action),
                    }
                    episode.append(data)
                    step_count += 1
                
                # 执行动作：将目标xy位置转换为末端执行器位姿
                target_position_3d = np.array([target_xy[0], target_xy[1], INIT_POSITION[2]])
                target_pose = pin.SE3(target_orientation, target_position_3d)
                
                # 使用IK求解目标关节角度
                q_current = controller.get_joint_positions(request_update=False)
                q_target, _, _ = controller.inverse_kinematics(
                    target_pose, q_init=q_current,
                    fixed_iterations=5  # 固定迭代次数，实时控制
                )
                
                # 发送命令
                controller.move_to_joint_positions(
                    q_target, limit_speed=0.5, acceleration=3.0
                )
                
                # 显示状态
                if step_count > 0 and step_count % 10 == 0:
                    print(f'\r  Steps: {step_count} | Target: ({target_xy[0]:.3f}, {target_xy[1]:.3f})', end='', flush=True)
                
                # 控制频率
                start_time = time.time()
                elapsed = time.time() - start_time
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            print()  # 换行
            
            # Episode 结束：自动关闭控制状态
            mouse_window.stop_control_external()
            
            # 检查是否退出
            if quit_flag:
                break
            
            # 保存或重试
            if retry:
                continue
            
            if len(episode) > 0:
                # 保存 episode
                data_dict = dict()
                for key in episode[0].keys():
                    data_dict[key] = np.stack([x[key] for x in episode])
                
                try:
                    replay_buffer.add_episode(data_dict, compressors='disk')
                    print(f'已保存 episode {seed}，共 {len(episode)} 步')
                except AssertionError:
                    print(f'错误：数据形状不匹配！')
                    print(f'提示：rm -rf {output} 或使用新路径')
                except Exception as e:
                    print(f'保存失败: {e}')
            else:
                print(f'Episode {seed} 为空，跳过')
    
    except KeyboardInterrupt:
        print("\n中断")
    
    finally:
        # 恢复终端设置
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except:
            pass
        
        # 关闭窗口和环境
        try:
            mouse_window.close()
        except:
            pass
        try:
            cv2.destroyAllWindows()
        except:
            pass
        try:
            pipeline.stop()
        except:
            pass
        try:
            controller.disable_all()
            controller.close()
        except:
            pass


if __name__ == "__main__":
    main()

