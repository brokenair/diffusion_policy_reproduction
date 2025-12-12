"""
简单的圆形轨迹demo - 实机 + 仿真
直接读取电机角度，规划轨迹，执行画圆
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin

from ..utils.mjcf_utils import write_mjcf
from .robot_controller import RobotController


# ============================================================================
# 配置参数
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = ROOT / "assets" / "urdf" / "Lagrange_01_sim.urdf"
MESH_DIR = (ROOT / "assets" / "meshes").resolve()
MJCF_PATH = ROOT / "assets" / "mjcf" / "lagrange_generated.xml"
CONFIG_PATH = ROOT / "config" / "motor_calibration.yaml"
EE_FRAME = "tool_link"

# 轨迹参数
CIRCLE_RADIUS = 0.08  # 圆的半径（米）
CIRCLE_PERIOD = 8.0  # 画一个圆的时间（秒）
TRANSITION_DURATION = 2.0  # 从当前位置到圆上起始点的过渡时间（秒）

# 控制参数
CONTROL_DT = 0.01  # 控制循环时间步（100Hz）

# 开关
ENABLE_REAL_ROBOT = True  # 是否启用实机
ENABLE_RENDERING = True  # 是否启用渲染（关闭后可提高控制频率）


# ============================================================================
# 辅助函数
# ============================================================================

def make_target_pose(center, radius, t, orientation):
    """生成圆形轨迹的目标位姿"""
    theta = 2.0 * np.pi * t
    offset = np.array([
        radius * np.cos(theta),
        radius * np.sin(theta),
        0.0,
    ])
    target_position = center + offset
    return pin.SE3(orientation, target_position)


# ============================================================================
# 主程序
# ============================================================================

def main():
    global ENABLE_REAL_ROBOT
    
    print("=" * 70)
    print("圆形轨迹 Demo - 实机 + 仿真")
    print("=" * 70)
    
    # 1. 初始化MuJoCo
    print("\n[1/5] 初始化MuJoCo...")
    MJCF_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_mjcf(URDF_PATH, MESH_DIR, MJCF_PATH)
    mj_model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    mj_data = mujoco.MjData(mj_model)
    print(f"  ✓ MuJoCo加载成功 (DOF: {mj_model.nq})")
    
    # 2. 初始化Pinocchio（用于仿真可视化）
    print("\n[2/5] 初始化Pinocchio...")
    pin_model = pin.buildModelFromUrdf(str(URDF_PATH))
    pin_data = pin_model.createData()
    frame_id = pin_model.getFrameId(EE_FRAME)
    if frame_id >= len(pin_model.frames):
        raise SystemExit(f"Frame '{EE_FRAME}' not found")
    print(f"  ✓ Pinocchio加载成功 (DOF: {pin_model.nq}, Frame ID: {frame_id})")
    
    # 3. 连接实机（如果启用，并初始化运动学）
    controller = None
    if ENABLE_REAL_ROBOT:
        print("\n[3/5] 连接实机...")
        try:
            controller = RobotController.from_config(
                CONFIG_PATH,
                urdf_path=URDF_PATH,  # 传入URDF路径，启用运动学功能
                end_effector=EE_FRAME
            )
            controller.connect()
            controller.disable_all()
            # 不在这里使能，让pos_control自己处理
            print(f"  ✓ 实机连接成功 ({len(controller.joint_configs)}个关节)")
            print(f"  ✓ 运动学功能已启用")
        except Exception as e:
            print(f"  ✗ 实机连接失败: {e}")
            print("  → 切换到纯仿真模式")
            ENABLE_REAL_ROBOT = False
            controller = None
    else:
        print("\n[3/5] 实机关闭（纯仿真模式）")
    
    # 4. 读取当前关节角度
    print("\n[4/5] 读取当前关节角度...")
    if ENABLE_REAL_ROBOT and controller:
        # 等待一下让电机初始化
        time.sleep(0.5)
        # 读取当前角度（已校准的关节角度）
        q_current = controller.get_joint_positions(request_update=True)
        print(f"  ✓ 当前关节角度: {q_current}")
    else:
        # 仿真模式，使用默认角度
        q_current = np.array([0.1326, 1.3073, 2.3998, 0.0855, 0.3822, 0.0000])
        print(f"  ⚠ 仿真模式，使用默认角度: {q_current}")
    
    # 5. 规划轨迹
    print("\n[5/5] 规划轨迹...")
    
    # 计算当前位置对应的末端执行器位姿
    if ENABLE_REAL_ROBOT and controller:
        # 使用controller的运动学功能
        current_pose = controller.get_end_effector_pose()
    else:
        # 仿真模式，使用Pinocchio
        pin.forwardKinematics(pin_model, pin_data, q_current)
        pin.updateFramePlacements(pin_model, pin_data)
        current_pose = pin_data.oMf[frame_id]
    
    circle_center = current_pose.translation.copy()
    
    # 期望的末端执行器姿态（向下）
    desired_orientation = np.array([
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    
    # 计算圆上起始点的位姿
    circle_start_pose = make_target_pose(circle_center, CIRCLE_RADIUS, 0.0, desired_orientation)
    
    # 使用controller的IK功能求解圆上起始点对应的关节角度
    print("  → 求解IK（从当前位置到圆上起始点）...")
    if ENABLE_REAL_ROBOT and controller:
        q_circle_start, error, success = controller.inverse_kinematics(
            circle_start_pose, q_init=q_current,
            max_iterations=100, tolerance=1e-4
        )
        if not success:
            print(f"  ✗ IK求解失败 (error: {error:.6f})")
            return 1
        print(f"  ✓ IK求解成功 (error: {error:.6f})")
    else:
        # 仿真模式，使用Pinocchio
        from ..utils.ik_utils import solve_ik
        q_circle_start, error, n_iter = solve_ik(
            pin_model, pin_data, frame_id, circle_start_pose, q_current,
            max_iterations=100, tolerance=1e-4, verbose=False
        )
        if q_circle_start is None:
            print(f"  ✗ IK求解失败 (error: {error:.6f})")
            return 1
        print(f"  ✓ IK求解成功 ({n_iter}次迭代, error: {error:.6f})")
    
    # 规划从当前位置到圆上起始点的平滑轨迹
    print(f"  → 规划过渡轨迹 ({TRANSITION_DURATION}秒)...")
    from ..utils.trajectory_utils import minimal_jerk_trajectory
    q_transition, qd_transition, _ = minimal_jerk_trajectory(
        q_current, q_circle_start, TRANSITION_DURATION, CONTROL_DT
    )
    print(f"  ✓ 过渡轨迹规划完成 ({len(q_transition)}个点)")
    
    print("\n" + "=" * 70)
    print("开始执行")
    print("=" * 70)
    print(f"  阶段1: 过渡到圆上起始点 ({TRANSITION_DURATION}秒)")
    print(f"  阶段2: 画圆 (周期: {CIRCLE_PERIOD}秒)")
    print(f"  控制频率: {1.0/CONTROL_DT:.0f}Hz (dt={CONTROL_DT*1000:.0f}ms)")
    print(f"  渲染: {'启用' if ENABLE_RENDERING else '关闭'}")
    print("\n按 Ctrl+C 停止")
    print("=" * 70 + "\n")
    
    # 执行轨迹
    q = q_current.copy()
    phase = "transition"
    transition_start_time = time.time()
    transition_idx = 0
    
    try:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            # 初始化可视化
            mj_data.qpos[:mj_model.nq] = q
            mj_data.qvel[:] = 0.0
            mujoco.mj_forward(mj_model, mj_data)

            last_print = 0.0
            
            # 频率监控
            loop_times = []  # 存储每次循环的实际时间
            max_freq = 0.0
            min_freq = float('inf')
            actual_loop_time = CONTROL_DT  # 初始值，避免第一次打印时未定义

            # 渲染时间监控
            render_times = []  # 存储每次渲染的时间
            max_render_time = 0.0
            min_render_time = float('inf')
            avg_render_time = 0.0

            # 控制逻辑时间
            control_elapsed = 0.0
            
            while viewer.is_running():
                loop_start = time.time()
                current_time = loop_start
                
                if phase == "transition":
                    # 阶段1: 执行过渡轨迹
                    elapsed = current_time - transition_start_time
                    
                    if elapsed < TRANSITION_DURATION:
                        # 在过渡阶段，使用规划的轨迹
                        idx = min(int(elapsed / CONTROL_DT), len(q_transition) - 1)
                        q = q_transition[idx]
                        
                        # 发送命令到实机
                        if ENABLE_REAL_ROBOT and controller:
                            controller.move_to_joint_positions(
                                q, limit_speed=0.5, acceleration=3.0
                            )
                            # 读取实机实际角度（用于可视化）
                            q_real = controller.get_joint_positions(request_update=False)
                        else:
                            # 仿真模式，使用规划的角度
                            q_real = q
                        
                        # 更新可视化（使用实机实际角度）
                        mj_data.qpos[:mj_model.nq] = q_real
                        mj_data.qvel[:] = 0.0
                        mujoco.mj_forward(mj_model, mj_data)
                    else:
                        # 过渡完成，切换到画圆阶段
                        phase = "circle"
                        circle_start_time = current_time

                        # 重置频率统计（过滤掉启动和过渡阶段的异常值）
                        loop_times.clear()
                        max_freq = 0.0
                        min_freq = float('inf')

                        # 重置渲染时间统计
                        render_times.clear()
                        max_render_time = 0.0
                        min_render_time = float('inf')
                        avg_render_time = 0.0

                        print("✓ 过渡完成，开始画圆...\n")
                
                elif phase == "circle":
                    # 阶段2: 画圆
                    elapsed = current_time - circle_start_time
                    t = (elapsed / CIRCLE_PERIOD) % 1.0

                    # 记录控制逻辑开始时间
                    control_start = time.time()

                    # 计算目标位姿
                    target_pose = make_target_pose(circle_center, CIRCLE_RADIUS, t, desired_orientation)
                    
                    # 使用controller的IK功能进行实时跟踪（固定迭代次数）
                    if ENABLE_REAL_ROBOT and controller:
                        q, ik_err, _ = controller.inverse_kinematics(
                            target_pose, q_init=q,
                            fixed_iterations=10  # 固定10次迭代，计算时间可控
                        )
                    else:
                        # 仿真模式，使用damped_ik_step
                        from ..utils.ik_utils import damped_ik_step
                        ik_err = 0.0
                        for _ in range(10):
                            q, ik_err = damped_ik_step(
                                pin_model, pin_data, frame_id, q, target_pose,
                                gain=0.5, damping=1e-2
                            )
                    
                    # 发送命令到实机
                    if ENABLE_REAL_ROBOT and controller:
                        controller.move_to_joint_positions(
                            q, limit_speed=0.5, acceleration=3.0
                        )
                        # 读取实机实际角度（用于可视化）
                        q_real = controller.get_joint_positions(request_update=False)
                    else:
                        # 仿真模式，使用IK计算的角度
                        q_real = q
                    
                    # 更新可视化（使用实机实际角度）
                    mj_data.qpos[:mj_model.nq] = q_real
                    mj_data.qvel[:] = 0.0
                    mujoco.mj_forward(mj_model, mj_data)

                    # 记录控制逻辑结束时间
                    control_elapsed = time.time() - control_start
                    
                    # 定期打印状态
                    if elapsed - last_print >= 2.0:
                        # IK迭代误差（基于IK计算的角度）
                        if ENABLE_REAL_ROBOT and controller:
                            ik_actual_pose = controller.forward_kinematics(q)
                        else:
                            pin.forwardKinematics(pin_model, pin_data, q)
                            pin.updateFramePlacements(pin_model, pin_data)
                            ik_actual_pose = pin_data.oMf[frame_id]
                        ik_pos_err = np.linalg.norm(ik_actual_pose.translation - target_pose.translation)
                        
                        # 实机实际误差（从实机读取的角度）
                        if ENABLE_REAL_ROBOT and controller:
                            real_actual_pose = controller.forward_kinematics(q_real)
                        else:
                            pin.forwardKinematics(pin_model, pin_data, q_real)
                            pin.updateFramePlacements(pin_model, pin_data)
                            real_actual_pose = pin_data.oMf[frame_id]
                        real_pos_err = np.linalg.norm(real_actual_pose.translation - target_pose.translation)
                        
                        # 计算频率统计
                        if len(loop_times) > 0:
                            avg_loop_time = sum(loop_times) / len(loop_times)
                            avg_freq = 1.0 / avg_loop_time if avg_loop_time > 0 else 0.0
                            current_freq = 1.0 / actual_loop_time if actual_loop_time > 0 else 0.0

                            freq_info = f" | 频率: {current_freq:.1f}Hz (平均:{avg_freq:.1f} 最高:{max_freq:.1f} 最低:{min_freq:.1f})"
                        else:
                            freq_info = " | 频率: 计算中..."

                        # 计算渲染时间统计
                        if len(render_times) > 0:
                            render_info = f" | 渲染: {render_elapsed*1000:.1f}ms (平均:{avg_render_time*1000:.1f}ms 最高:{max_render_time*1000:.1f}ms 最低:{min_render_time*1000:.1f}ms)"
                        else:
                            render_info = " | 渲染: 计算中..."

                        # 控制逻辑时间
                        control_info = f" | 控制: {control_elapsed*1000:.1f}ms"

                        if ENABLE_REAL_ROBOT and controller:
                            print(f"  t={elapsed:.1f}s | IK误差: {ik_pos_err*1000:.2f}mm | 实机误差: {real_pos_err*1000:.2f}mm{freq_info}{render_info}{control_info}")
                        else:
                            print(f"  t={elapsed:.1f}s | IK误差: {ik_pos_err*1000:.2f}mm{freq_info}{render_info}{control_info}")
                        last_print = elapsed
                        
                        # 重置频率统计窗口（每2秒重置一次，保持最近的数据）
                        if len(loop_times) > 200:  # 保留最近200次循环的数据
                            loop_times = loop_times[-100:]  # 只保留最近100次

                        # 重置渲染时间统计窗口
                        if len(render_times) > 200:  # 保留最近200次循环的数据
                            render_times = render_times[-100:]  # 只保留最近100次

                # 渲染（记录渲染时间）
                if ENABLE_RENDERING:
                    render_start = time.time()
                    viewer.sync()
                    render_elapsed = time.time() - render_start

                    # 统计渲染时间
                    render_times.append(render_elapsed)
                    if render_elapsed > max_render_time:
                        max_render_time = render_elapsed
                    if render_elapsed < min_render_time:
                        min_render_time = render_elapsed
                    if len(render_times) > 0:
                        avg_render_time = sum(render_times) / len(render_times)
                else:
                    # 不渲染时，渲染时间为0
                    render_elapsed = 0.0

                # 控制循环频率
                loop_elapsed = time.time() - loop_start  # 当前循环实际执行时间
                sleep_time = CONTROL_DT - loop_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # 频率监控：记录本次循环的实际时间（包括sleep）
                loop_end = time.time()
                actual_loop_time = loop_end - loop_start
                if actual_loop_time > 0:
                    loop_freq = 1.0 / actual_loop_time
                    loop_times.append(actual_loop_time)

                    # 更新最高和最低频率
                    if loop_freq > max_freq:
                        max_freq = loop_freq
                    if loop_freq < min_freq:
                        min_freq = loop_freq
    
    except KeyboardInterrupt:
        print("\n\n[!] 用户中断")
    except Exception as e:
        print(f"\n[ERROR] 执行失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        print("\n清理资源...")
        if ENABLE_REAL_ROBOT and controller:
            try:
                controller.disable_all()
                controller.close()
                print("  ✓ 实机关闭")
            except Exception as e:
                print(f"  ✗ 清理失败: {e}")
        print("完成")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

