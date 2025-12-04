"""
MuJoCo 版本的 PushT 数据采集脚本。

使用 MuJoCo 原生 viewer 显示场景，通过鼠标控制 stick。

使用方法：
    python demo_pusht_mujoco.py -o data/pusht_mujoco_demo.zarr

操作说明：
    - MuJoCo viewer 显示 3D 场景
    - 鼠标控制窗口：点击一次开始控制，再点击一次停止；控制时移动鼠标即可
    - 按 'Q' 退出程序
    - 按 'R' 重试当前 episode
    - 按 'S' 保存当前 episode（即使未完成）
    - 按 'P' 暂停/继续
    
注意：每个 episode 结束后需要重新点击鼠标才能开始下一个 episode
"""

import numpy as np
import click
import time
import threading
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.pusht.pusht_mujoco_env import PushTMujocoEnv


class MouseTargetWindow:
    """
    鼠标目标窗口（Toggle 模式）。
    
    - 点击一次：开始控制（目标设为 stick 当前位置）
    - 控制时：移动鼠标即可更新目标（不需要按下）
    - 再次点击：停止控制（锁定当前目标）
    - Episode 结束：自动停止控制（避免下次突变）
    """
    
    def __init__(self, plane_half_extent, window_size=800):
        self.window_size = window_size
        self.half_extent = np.asarray(plane_half_extent, dtype=float)
        self._target = np.zeros(2, dtype=float)  # 当前目标
        self._is_controlling = False  # 是否处于控制状态（toggle）
        self._control_start_pos = np.zeros(2, dtype=float)  # 开始控制时的 stick 位置
        self._control_start_mouse = np.zeros(2, dtype=float)  # 开始控制时的鼠标位置
        self._current_stick_pos = np.zeros(2, dtype=float)  # 当前 stick 位置（由外部更新）
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._init_error = None
        self._should_close = False  # 关闭标志
        self.root = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        
        if not self._ready.wait(timeout=5):
            raise RuntimeError("鼠标窗口初始化超时")
        if self._init_error is not None:
            raise self._init_error
    
    def set_current_pos(self, pos):
        """设置 stick 当前位置（由环境调用）"""
        with self._lock:
            self._current_stick_pos = np.array(pos, dtype=float)
    
    def toggle_control(self, mouse_plane_pos):
        """切换控制状态（由鼠标点击调用）"""
        with self._lock:
            if not self._is_controlling:
                # 开始控制：目标设为 stick 当前位置
                self._is_controlling = True
                self._control_start_pos = self._current_stick_pos.copy()
                self._control_start_mouse = np.array(mouse_plane_pos, dtype=float)
                self._target = self._control_start_pos.copy()
            else:
                # 停止控制：锁定当前目标
                self._is_controlling = False
            return self._is_controlling
    
    def update_target(self, mouse_plane_pos):
        """更新目标（由鼠标移动调用）"""
        with self._lock:
            if self._is_controlling:
                # 目标 = stick初始位置 + 鼠标移动量
                delta = mouse_plane_pos - self._control_start_mouse
                self._target = self._control_start_pos + delta
                # 限制在范围内
                self._target = np.clip(self._target, -self.half_extent, self.half_extent)
    
    def stop_control_external(self):
        """外部调用：强制停止控制（episode 结束时）"""
        with self._lock:
            self._is_controlling = False
    
    def get_target(self):
        """获取目标位置"""
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
        
        # 绑定鼠标事件（只需要点击和移动）
        canvas.bind("<ButtonPress-1>", self._on_click)
        canvas.bind("<Motion>", self._on_motion)
        
        self._ready.set()
        
        # 使用 after 循环检查是否需要关闭
        def check_close():
            if self._should_close:
                self.root.quit()
            else:
                self.root.after(100, check_close)
        
        self.root.after(100, check_close)
        self.root.mainloop()
        
        # mainloop 退出后只隐藏窗口，不调用 destroy()
        # 避免 "Tcl_AsyncDelete: async handler deleted by the wrong thread" 错误
        try:
            self.root.withdraw()
        except:
            pass
    
    def _canvas_to_plane(self, x_pixel, y_pixel):
        """将画布坐标转换为平面坐标"""
        x_clamped = min(max(x_pixel, 0.0), self.window_size)
        y_clamped = min(max(y_pixel, 0.0), self.window_size)
        norm_x = x_clamped / self.window_size * 2.0 - 1.0
        norm_y = 1.0 - (y_clamped / self.window_size * 2.0)
        return np.array([
            norm_x * self.half_extent[0],
            norm_y * self.half_extent[1]
        ], dtype=float)
    
    def _plane_to_canvas(self, plane_pos):
        """将平面坐标转换为画布坐标"""
        norm_x = plane_pos[0] / self.half_extent[0]
        norm_y = plane_pos[1] / self.half_extent[1]
        x_pixel = (norm_x + 1.0) / 2.0 * self.window_size
        y_pixel = (1.0 - norm_y) / 2.0 * self.window_size
        return x_pixel, y_pixel
    
    def _update_display(self, target_pos, is_controlling):
        """更新显示"""
        # 将目标位置转换为画布坐标
        x, y = self._plane_to_canvas(target_pos)
        
        # 更新十字线
        if is_controlling:
            self.canvas.coords(self.cross_h, 0, y, self.window_size, y)
            self.canvas.coords(self.cross_v, x, 0, x, self.window_size)
            self.canvas.itemconfigure(self.cross_h, fill="#ff3333", width=3)
            self.canvas.itemconfigure(self.cross_v, fill="#ff3333", width=3)
            self.canvas.itemconfigure(self.status_text, text="● CONTROLLING (click to stop)", fill="#ff3333")
        else:
            self.canvas.coords(self.cross_h, 0, y, self.window_size, y)
            self.canvas.coords(self.cross_v, x, 0, x, self.window_size)
            self.canvas.itemconfigure(self.cross_h, fill="#888888", width=2)
            self.canvas.itemconfigure(self.cross_v, fill="#888888", width=2)
            self.canvas.itemconfigure(self.status_text, text="○ Click to start", fill="#888888")
        
        self.canvas.itemconfigure(self.coord_text, text=f"Target: ({target_pos[0]:.2f}, {target_pos[1]:.2f}) m")
    
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
        # 等待线程结束（最多 0.5 秒）
        self._thread.join(timeout=0.5)


@click.command()
@click.option('-o', '--output', required=True, help='输出 zarr 文件路径')
@click.option('-rs', '--render_size', default=480, type=int, help='录制图像分辨率')
@click.option('-hz', '--control_hz', default=10, type=int, help='控制频率')
@click.option('--mouse-window', default=800, type=int, help='鼠标控制窗口大小')
def main(output, render_size, control_hz, mouse_window):
    """
    采集 MuJoCo PushT 任务的演示数据。
    """
    
    # 创建 replay buffer
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')
    
    # 创建环境
    env = PushTMujocoEnv(
        render_size=render_size,
        # 这个不能开
        render_action=False,
    )
    
    # 创建鼠标控制窗口
    mouse_window = MouseTargetWindow(
        plane_half_extent=env.plane_half_extent,
        window_size=mouse_window
    )
    
    # 控制频率
    dt = 1.0 / control_hz
    
    print("=" * 60)
    print("PushT MuJoCo 数据采集")
    print("=" * 60)
    print(f"配置: {render_size}x{render_size} @ {control_hz}Hz -> {output}")
    print("")
    print("操作：")
    print("  鼠标窗口: 点击切换控制（ON/OFF），控制时移动鼠标即可")
    print("  Q - 退出 | R - 重试 | S - 保存 | P - 暂停")
    print("")
    print("注意: Episode 结束后会自动停止控制，需重新点击开始")
    print("=" * 60)
    
    # 设置终端为非阻塞输入
    import sys
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
            
            # 设置种子并重置环境
            env.seed(seed)
            obs = env.reset()
            info = env._get_info()
            
            # 打开 MuJoCo viewer
            env.render(mode='human')
            
            # 状态变量
            retry = False
            pause = False
            done = False
            force_save = False
            quit_flag = False
            step_count = 0
            
            print(f"  初始位置: stick=({obs[0]:.2f}, {obs[1]:.2f}), t_block=({obs[2]:.2f}, {obs[3]:.2f}), yaw={obs[4]:.2f}")
            print(f"  点击鼠标窗口开始控制...")
            
            # 初始化鼠标窗口的目标为 stick 当前位置
            stick_pos = env._get_stick_pos()
            mouse_window.set_current_pos(stick_pos)
            with mouse_window._lock:
                mouse_window._target = stick_pos.copy()
                mouse_window._is_controlling = False  # 确保控制状态关闭
            
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
                        break
                    elif key.lower() == 'p':
                        pause = not pause
                
                if pause:
                    time.sleep(0.1)
                    env.render(mode='human')
                    continue
                
                # 更新鼠标窗口中的 stick 当前位置
                stick_pos = env._get_stick_pos()
                mouse_window.set_current_pos(stick_pos)
                
                # 获取目标位置（始终返回目标，松开后保持锁定位置）
                action = mouse_window.get_target()
                is_recording = mouse_window.is_controlling()
                
                # 只有在点击控制时才记录数据
                if is_recording:
                    # 记录数据
                    state = np.concatenate([
                        info['pos_agent'],
                        info['block_pose']
                    ])
                    
                    # 获取图像
                    img = env.render(mode='rgb_array')
                    
                    data = {
                        'img': img,
                        'state': np.float32(state),
                        'action': np.float32(action),
                        'n_contacts': np.float32([info['n_contacts']])
                    }
                    episode.append(data)
                    step_count += 1
                
                # 执行动作
                start_time = time.time()
                obs, reward, done, info = env.step(action)
                
                # 更新 viewer
                env.render(mode='human')
                
                # 显示状态
                if step_count > 0 and step_count % 10 == 0:
                    print(f'\r  Steps: {step_count} | Reward: {reward:.3f}', end='', flush=True)
                
                # 控制频率
                elapsed = time.time() - start_time
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            print()  # 换行
            
            # Episode 结束：自动关闭控制状态（避免下次突变）
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
                    if done:
                        print(f'任务成功！')
                except AssertionError:
                    print(f'错误：数据形状不匹配！当前 img: {data_dict["img"].shape}')
                    print(f'提示：rm -rf {output} 或使用新路径')
                except Exception as e:
                    print(f'保存失败: {e}')
            else:
                print(f'Episode {seed} 为空，跳过')

            # 继续下一个 episode
    
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
            env.close()
        except:
            pass


if __name__ == "__main__":
    main()
