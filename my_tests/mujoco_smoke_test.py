#!/usr/bin/env python3
"""
轻量级 MuJoCo 冒烟测试脚本。

在 robodiff Conda 环境中运行本脚本，可以验证 mujoco==3.3.7
是否能够创建模型、步进仿真并读取关键状态。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from textwrap import dedent

import mujoco
import numpy as np

# 注意是半尺寸，这里的平面size="0.8 0.8 0.1"，其实边长是1.6
SCENE_XML = dedent(
    """
    <mujoco model="robodiff_smoke_test">
      <compiler angle="degree" inertiafromgeom="true"/>
      <option timestep="0.01" gravity="0 0 -9.8"/>
      <default>
        <geom
          condim="3"
          friction="0.3 0 0"
          rgba="0.5 0.5 0.5 1"
          solref="0.005 1"
          solimp="0.99 0.999 0.0001"/>
      </default>
      <visual>
        <headlight diffuse="0.7 0.7 0.7" specular="0.1 0.1 0.1"/>
        <rgba haze="0 0 0 0"/>
        <global azimuth="90" elevation="90"/>
      </visual>
      <worldbody>
        <camera name="top_cam" pos="0 0 2" mode="fixed" euler="0 0 0"/>
        <light name="key" pos="0 0 3" dir="0 0 -1" diffuse="1 1 1" specular="0.3 0.3 0.3" cutoff="60" exponent="2"/>
        <light name="fill" pos="1.5 1.5 2" dir="-1 -1 -1" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2"/>
        <geom name="table" type="plane" size="0.8 0.8 0.1" rgba="0.3 0.2 0.1 1"/>
        <body name="stick" pos="0 0 0.2" quat="0 0 0 1">
            <joint name="slide_x" type="slide" axis="1 0 0" range="-0.8 0.8"/>
            <joint name="slide_y" type="slide" axis="0 1 0" range="-0.8 0.8"/>
            <geom type="cylinder" size="0.03 0.2" density="800" rgba="0.8 0.3 0.3 1"/>
        </body>
        <body name="t_block" pos="0 0 0.05">
            <joint name="t_slide_x" type="slide" axis="1 0 0" range="-0.5 0.5"/>
            <joint name="t_slide_y" type="slide" axis="0 1 0" range="-0.5 0.5"/>
            <joint name="t_slide_z" type="slide" axis="0 0 1"/>
            <joint name="t_yaw" type="hinge" axis="0 0 1"/>
            <geom type="box" size="0.12 0.03 0.05" density="800"/>
            <geom type="box" size="0.03 0.12 0.05" density="800" pos="0 0.15 0"/>
        </body>
        <!-- 目标区域：透明、无碰撞、固定位置 -->
        <body name="t_goal" pos="0 0 0.025" euler="0 0 45">
            <geom type="box" size="0.12 0.03 0.025" rgba="0.2 0.8 0.2 0.3" contype="0" conaffinity="0"/>
            <geom type="box" size="0.03 0.12 0.025" rgba="0.2 0.8 0.2 0.3" pos="0 0.15 0" contype="0" conaffinity="0"/>
        </body>
      </worldbody>
      <sensor>
        <framepos name="stick_pos" objtype="body" objname="stick"/>
      </sensor>
    </mujoco>
    """
).strip()


class _MouseTargetWindow:
    """
    鼠标目标窗口。
    
    点击鼠标左键后开始控制，松开停止。
    - 按下时：目标设为 stick 当前位置（避免突变）
    - 拖拽时：跟随鼠标移动
    - 松开时：目标锁定在松开位置（避免飘动）
    """

    def __init__(self, window_size: int, plane_half_extent: np.ndarray) -> None:
        self.window_size = window_size
        self.half_extent = np.asarray(plane_half_extent, dtype=float)
        self._target = np.zeros(2, dtype=float)
        self._is_clicking = False
        self._click_start_pos = np.zeros(2, dtype=float)
        self._click_start_mouse = np.zeros(2, dtype=float)
        self._current_stick_pos = np.zeros(2, dtype=float)
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._should_close = False
        self._init_error: Exception | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5):
            raise RuntimeError("鼠标窗口初始化超时")
        if self._init_error is not None:
            raise self._init_error

    def set_current_pos(self, pos: np.ndarray) -> None:
        """设置 stick 当前位置（由外部调用）"""
        with self._lock:
            self._current_stick_pos = np.array(pos, dtype=float)

    def get_target(self) -> np.ndarray:
        """获取目标位置"""
        with self._lock:
            return self._target.copy()

    def is_controlling(self) -> bool:
        """是否正在控制"""
        with self._lock:
            return self._is_clicking

    def close(self) -> None:
        """关闭窗口"""
        self._should_close = True
        self._thread.join(timeout=0.5)

    # --- 内部实现 ---
    def _run(self) -> None:
        try:
            import tkinter as tk
        except Exception as exc:
            self._init_error = exc
            self._ready.set()
            return

        self.root = tk.Tk()
        self.root.title("Mouse Control - CLICK to control")
        self.canvas = tk.Canvas(
            self.root,
            width=self.window_size,
            height=self.window_size,
            bg="#111111",
            highlightthickness=0,
        )
        self.canvas.pack()

        # 绘制边框和网格
        self.canvas.create_rectangle(2, 2, self.window_size - 2, self.window_size - 2, outline="#444")
        for i in range(1, 8):
            pos = i * self.window_size // 8
            self.canvas.create_line(pos, 0, pos, self.window_size, fill="#333")
            self.canvas.create_line(0, pos, self.window_size, pos, fill="#333")
        
        # 中心十字
        cx, cy = self.window_size // 2, self.window_size // 2
        self.canvas.create_line(cx - 20, cy, cx + 20, cy, fill="#666", width=2)
        self.canvas.create_line(cx, cy - 20, cx, cy + 20, fill="#666", width=2)

        self.cross_h = self.canvas.create_line(0, 0, 0, 0, fill="#ffcc33", width=2)
        self.cross_v = self.canvas.create_line(0, 0, 0, 0, fill="#ffcc33", width=2)
        self.status_text = self.canvas.create_text(
            self.window_size // 2, 25, fill="#888888",
            text="Click and hold to control", font=("Arial", 14)
        )
        self.info_text = self.canvas.create_text(
            self.window_size // 2, self.window_size - 15,
            fill="#cccccc", text="(0.00, 0.00) m"
        )

        # 绑定鼠标事件
        self.canvas.bind("<ButtonPress-1>", self._on_click)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<Motion>", self._on_motion)

        self._ready.set()

        # 检查关闭标志
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

    def _canvas_to_plane(self, x_pixel: float, y_pixel: float) -> np.ndarray:
        x_clamped = min(max(x_pixel, 0.0), self.window_size)
        y_clamped = min(max(y_pixel, 0.0), self.window_size)
        norm_x = x_clamped / self.window_size * 2.0 - 1.0
        norm_y = 1.0 - (y_clamped / self.window_size * 2.0)
        return np.array([norm_x * self.half_extent[0], norm_y * self.half_extent[1]], dtype=float)

    def _plane_to_canvas(self, plane_pos: np.ndarray) -> tuple:
        norm_x = plane_pos[0] / self.half_extent[0]
        norm_y = plane_pos[1] / self.half_extent[1]
        x_pixel = (norm_x + 1.0) / 2.0 * self.window_size
        y_pixel = (1.0 - norm_y) / 2.0 * self.window_size
        return x_pixel, y_pixel

    def _update_display(self, target_pos: np.ndarray, is_active: bool) -> None:
        x, y = self._plane_to_canvas(target_pos)
        if is_active:
            self.canvas.coords(self.cross_h, 0, y, self.window_size, y)
            self.canvas.coords(self.cross_v, x, 0, x, self.window_size)
            self.canvas.itemconfigure(self.cross_h, fill="#ff3333")
            self.canvas.itemconfigure(self.cross_v, fill="#ff3333")
            self.canvas.itemconfigure(self.status_text, text="CONTROLLING", fill="#ff3333")
        else:
            self.canvas.coords(self.cross_h, 0, y, self.window_size, y)
            self.canvas.coords(self.cross_v, x, 0, x, self.window_size)
            self.canvas.itemconfigure(self.cross_h, fill="#33ff33")
            self.canvas.itemconfigure(self.cross_v, fill="#33ff33")
            self.canvas.itemconfigure(self.status_text, text="HOLDING POSITION", fill="#33ff33")
        self.canvas.itemconfigure(self.info_text, text=f"({target_pos[0]:.2f}, {target_pos[1]:.2f}) m")

    def _on_click(self, event) -> None:
        """鼠标点击：目标设为 stick 当前位置"""
        mouse_pos = self._canvas_to_plane(event.x, event.y)
        with self._lock:
            self._is_clicking = True
            self._click_start_pos = self._current_stick_pos.copy()
            self._click_start_mouse = mouse_pos
            self._target = self._click_start_pos.copy()
        self._update_display(self._click_start_pos, True)

    def _on_release(self, event) -> None:
        """鼠标释放：锁定当前目标"""
        with self._lock:
            self._is_clicking = False
            target = self._target.copy()
        self._update_display(target, False)

    def _on_drag(self, event) -> None:
        """鼠标拖拽：目标跟随移动"""
        mouse_pos = self._canvas_to_plane(event.x, event.y)
        with self._lock:
            if self._is_clicking:
                delta = mouse_pos - self._click_start_mouse
                self._target = self._click_start_pos + delta
                self._target = np.clip(self._target, -self.half_extent, self.half_extent)
                target = self._target.copy()
            else:
                target = self._target.copy()
        self._update_display(target, True)

    def _on_motion(self, event) -> None:
        """鼠标移动（未点击）"""
        with self._lock:
            is_active = self._is_clicking
            target = self._target.copy()
        self._update_display(target, is_active)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="最小 MuJoCo 场景：自由球落在桌面上，并打印高度/外力。"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="仿真步数（默认 10000，对应 20 秒）。",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=100,
        help="每隔多少步打印一次状态。",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="开启交互式渲染（依赖 mujoco.viewer，需要可用的显示环境）。",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="控制仿真速率：1.0 表示真实时间，0 表示尽快运行，其它值等比缩放。",
    )
    parser.add_argument(
        "--viewer-mode",
        choices=("passive", "active"),
        default="passive",
        help="选择 viewer.launch_passive（脚本掌控步进）或 viewer.launch（GUI 自动步进）。",
    )
    parser.add_argument(
        "--stick-target",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("STICK_X", "STICK_Y"),
        help="stick 在 xy 平面中的 PD 目标位置（默认 0 0，单位米）。",
    )
    parser.add_argument(
        "--stick-kp",
        type=float,
        default=200.0,
        help="stick 平移关节的 PD 比例增益（默认 200）。",
    )
    parser.add_argument(
        "--stick-kd",
        type=float,
        default=20.0,
        help="stick 平移关节的 PD 微分增益（默认 20）。",
    )
    parser.add_argument(
        "--stick-limit",
        type=float,
        nargs=2,
        default=(0.8, 0.8),
        metavar=("LIM_X", "LIM_Y"),
        help="stick 在 xy 平面的半长度限制，与关节 range 一致（默认 ±0.8m）。",
    )
    parser.add_argument(
        "--mouse-target",
        action="store_true",
        help="开启鼠标目标窗口，窗口内的光标位置映射为 stick 的 PD 目标。",
    )
    parser.add_argument(
        "--mouse-window",
        type=int,
        default=800,
        help="鼠标目标窗口的像素宽高（正方形，默认 800）。",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="录制视频（30fps），保存到 plots/ 文件夹。",
    )
    parser.add_argument(
        "--record-width",
        type=int,
        default=480,
        help="录制视频的宽度（默认 480）。",
    )
    parser.add_argument(
        "--record-height",
        type=int,
        default=480,
        help="录制视频的高度（默认 480）。",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    # 创建一个 mujoco.MjModel() 的实例，这个在创建之后就不会再变化了
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    # data 是 mujoco.MjData() 的实例，包含了当前时间步的所有动态量
    data = mujoco.MjData(model)

    stick_joint_names = ("slide_x", "slide_y")
    stick_joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        for joint_name in stick_joint_names
    ]
    stick_dof_ids = [model.jnt_dofadr[j_id] for j_id in stick_joint_ids]
    stick_qpos_ids = [model.jnt_qposadr[j_id] for j_id in stick_joint_ids]

    stick_target_xy = np.asarray(args.stick_target, dtype=float)
    stick_kp = float(args.stick_kp)
    stick_kd = float(args.stick_kd)
    stick_limit = np.asarray(args.stick_limit, dtype=float)
    mouse_target = None
    if args.mouse_target:
        try:
            mouse_target = _MouseTargetWindow(
                window_size=args.mouse_window, plane_half_extent=stick_limit
            )
            print(
                "鼠标目标窗口已开启：点击并拖拽控制 stick "
                f"(范围 ±{stick_limit[0]:.2f}m, ±{stick_limit[1]:.2f}m)。"
            )
        except Exception as exc:
            mouse_target = None
            print(f"鼠标目标窗口初始化失败，将退回静态目标：{exc}", file=sys.stderr)

    
    wall_start = time.perf_counter()

    def maybe_sleep() -> None:
        if args.speed <= 0:
            return
        sim_elapsed = data.time
        wall_elapsed = time.perf_counter() - wall_start
        target = sim_elapsed / args.speed
        if target > wall_elapsed:
            time.sleep(target - wall_elapsed)

    def apply_stick_pd(target_data: mujoco.MjData) -> None:
        # 更新鼠标窗口中的 stick 当前位置
        if mouse_target is not None:
            current_pos = np.array([
                target_data.qpos[stick_qpos_ids[0]],
                target_data.qpos[stick_qpos_ids[1]]
            ])
            mouse_target.set_current_pos(current_pos)
        
        target_xy = (
            mouse_target.get_target()
            if mouse_target is not None
            else stick_target_xy
        )
        for axis_idx, dof_id in enumerate(stick_dof_ids):
            qpos_idx = stick_qpos_ids[axis_idx]
            pos = target_data.qpos[qpos_idx]
            vel = target_data.qvel[dof_id]
            err = target_xy[axis_idx] - pos
            force = stick_kp * err - stick_kd * vel
            target_data.qfrc_applied[dof_id] = force

    def apply_custom_forces(target_data: mujoco.MjData) -> None:
        apply_stick_pd(target_data)

    # 前向传播，计算当前时间步的所有动态量，但是不推进时间
    mujoco.mj_forward(model, data)

    # 录像相关设置
    recorder = None
    video_frames = []
    record_fps = 30
    record_interval = max(1, int(1.0 / (model.opt.timestep * record_fps)))  # 每隔多少步录一帧
    if args.record:
        import os
        import cv2
        os.makedirs("plots", exist_ok=True)
        renderer = mujoco.Renderer(model, height=args.record_height, width=args.record_width)
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top_cam")
        print(f"录像已启用：30fps，分辨率 {args.record_width}x{args.record_height}，使用 top_cam 相机。")

    if args.render:
        try:
            from mujoco import viewer  # pylint: disable=import-error
        except Exception as exc:  # pragma: no cover - 仅在缺少依赖时触发
            print(f"无法导入 mujoco.viewer：{exc}", file=sys.stderr)
            return 1

        # 定义初始相机设置的辅助函数
        def setup_camera(v):
            v.cam.azimuth = -90
            v.cam.elevation = -90
            v.cam.distance = 2.5
            v.cam.lookat[:] = [0, 0, 0]

        if args.viewer_mode == "active":
            # Active 模式：使用 passive 模拟，但自动步进（不受 --speed 限制）
            print("开启 viewer（active 模式模拟），GUI 自动步进。")
            print("提示：按 Ctrl+Shift+左键拖拽可直接在 GUI 中施加力。关闭窗口退出。")
            try:
                with viewer.launch_passive(model, data) as v:
                    setup_camera(v)
                    while v.is_running():
                        apply_custom_forces(data)
                        mujoco.mj_step(model, data)
                        v.sync()
            except KeyboardInterrupt:
                print("\n用户中断，正常退出。")
        else:
            print("开启 viewer.launch_passive，可拖拽/缩放查看场景，Ctrl+C 退出。")
            try:
                # 启动窗口，这个 passive 模式不会主动更新，需要手动调用 sync() 来刷新
                with viewer.launch_passive(model, data) as v:
                    setup_camera(v)
                    for step in range(args.steps):
                        apply_custom_forces(data)
                        mujoco.mj_step(model, data)
                        # 录像
                        if args.record and step % record_interval == 0:
                            renderer.update_scene(data, camera=camera_id)
                            frame = renderer.render()
                            video_frames.append(frame)
                        # 睡眠一段时间，让仿真速率与真实时间一致
                        maybe_sleep()
                        v.sync()
            except KeyboardInterrupt:
                print("\n用户中断，正常退出。")
    else:
        for step in range(args.steps):
            apply_custom_forces(data)
            mujoco.mj_step(model, data)
            # 录像
            if args.record and step % record_interval == 0:
                renderer.update_scene(data, camera=camera_id)
                frame = renderer.render()
                video_frames.append(frame)
            maybe_sleep()

    # 保存视频
    if args.record and len(video_frames) > 0:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = f"plots/mujoco_record_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, record_fps, (args.record_width, args.record_height))
        for frame in video_frames:
            # MuJoCo 返回 RGB，OpenCV 需要 BGR
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"视频已保存：{video_path}，共 {len(video_frames)} 帧。")

    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))

