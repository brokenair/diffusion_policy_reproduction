"""MuJoCo visualization of the Lagrange arm driven by Pinocchio IK with trajectory planning."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import pinocchio as pin

from ..utils.mjcf_utils import write_mjcf
from ..utils.ik_utils import damped_ik_step, solve_ik
from ..utils.trajectory_utils import minimal_jerk_trajectory


ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = ROOT / "assets" / "urdf" / "Lagrange_01_sim.urdf"
MESH_DIR = (ROOT / "assets" / "meshes").resolve()
MJCF_PATH = ROOT / "assets" / "mjcf" / "lagrange_generated.xml"
EE_FRAME = "tool_link"


def make_target_pose(center: np.ndarray, radius: float, t: float, 
                     orientation_matrix: np.ndarray) -> pin.SE3:
    """
    Generate a circular trajectory in the horizontal plane (XY plane) with fixed orientation.
    
    Args:
        center: Center position of the circle [x, y, z] in world frame
        radius: Radius of the circle in meters
        t: Normalized time parameter (0 to 1 completes one full circle)
        orientation_matrix: Fixed 3x3 rotation matrix for end-effector orientation
    
    Returns:
        Target SE3 pose (position + orientation)
    """
    theta = 2.0 * np.pi * t
    # Circle in horizontal plane: vary X and Y, keep Z constant
    offset = np.array([
        radius * np.cos(theta),  # X offset
        radius * np.sin(theta),  # Y offset
        0.0                       # Z offset (horizontal plane)
    ])
    target_position = center + offset
    
    # Return SE3 pose with fixed orientation
    return pin.SE3(orientation_matrix, target_position)


def main():
    MJCF_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_mjcf(URDF_PATH, MESH_DIR, MJCF_PATH)

    # In Pinocchio 3.x, buildModelFromUrdf takes the root joint type as the second argument
    # For a fixed base robot, we can omit it or use JointModelFreeFlyer() if needed
    pin_model = pin.buildModelFromUrdf(str(URDF_PATH))
    pin_data = pin_model.createData()
    frame_id = pin_model.getFrameId(EE_FRAME)
    if frame_id >= len(pin_model.frames):
        raise SystemExit(f"Frame '{EE_FRAME}' not found in URDF (frames: {[f.name for f in pin_model.frames]})")

    mj_model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    mj_data = mujoco.MjData(mj_model)
    if mj_model.nq != pin_model.nq:
        raise SystemExit(f"Mismatch between MuJoCo nq={mj_model.nq} and Pinocchio nq={pin_model.nq}")

    # Initialize robot at custom configuration (avoid singularities)
    # Option 1: Use neutral configuration (all joints at 0)
    # q = pin.neutral(pin_model)
    
    # Option 2: Set custom joint angles (in radians) for 6-DOF robot [j1, j2, j3, j4, j5, j6]
    # Example: slightly bent configuration to avoid singularities
    q = np.array([0.1326, 1.3073, 2.3998, 0.0855, 0.3822, 0.0000])
    
    # Option 3: Use a predefined safe configuration
    # q = np.array([0.0, 1.0, -0.5, 0.0, 1.0, 0.0])
    
    # 计算所有关节和连杆的位姿（存储在 pin_data.oMi 中）
    pin.forwardKinematics(pin_model, pin_data, q)
    # 计算所有坐标系（frames）的位姿，包括末端执行器（存储在 pin_data.oMf 中）
    pin.updateFramePlacements(pin_model, pin_data)
    
    # Get initial end-effector position
    # 每一个oMi[i]或者oMf[i]都是一个SE3对象
    initial_pose = pin_data.oMf[frame_id].copy()
    initial_position = initial_pose.translation.copy()
    
    print(f"Initial end-effector position: {initial_position}")
    print(f"Distance from base: {np.linalg.norm(initial_position):.3f} m")
    print(f"Initial end-effector orientation:")
    print(f"  Z-axis direction: {initial_pose.rotation[:, 2]}")
    
    # Define fixed orientation: pointing downward (Z-axis pointing in -Z direction)
    # Rotation matrix: Tool Z-axis aligned with world -Z axis
    desired_orientation = np.array([
        [0.0, -1.0,  0.0],   # X-axis stays as -Y
        [-1.0, 0.0,  0.0],   # Y-axis flips to -X
        [0.0,  0.0, -1.0]    # Z-axis flips to -Z (pointing downward)
    ])
    
    # ========== 圆形轨迹参数（在这里调整） ==========
    
    # 方法1：相对于初始位置设置圆心（推荐，会自动适应初始位置）
    circle_center = initial_position + np.array([0.0, 0.0, 0.0])
    
    # 方法2：直接设置圆心的绝对位置（世界坐标系）
    # circle_center = np.array([-0.3, 0.0, 0.3])  # [X, Y, Z] 米
    
    # 圆的半径（米）
    radius = 0.08
    
    # 轨迹参数
    dt = 0.01  # 控制循环时间步
    transition_duration = 2.0  # 从初始位置到圆上起始点的过渡时间（秒）
    circle_period = 8.0  # 完成一个圆的时间（秒）
    
    print(f"\n🎯 Circular Trajectory Settings:")
    print(f"  Center: {circle_center}")
    print(f"  Radius: {radius} m")
    print(f"  Plane: Horizontal (XY plane)")
    print(f"  Orientation: Fixed, pointing downward (tool Z-axis → world -Z)")
    print(f"  Desired Z-axis: {desired_orientation[:, 2]}")
    print(f"  Transition duration: {transition_duration} s (from center to circle start)")
    print(f"  Circle period: {circle_period} s per circle")
    
    # ========== 规划阶段1：从初始位置平滑过渡到圆上起始点 ==========
    print(f"\n[Planning] Computing transition trajectory from center to circle start...")
    
    # 计算圆上的起始点（t=0时的位置）
    circle_start_pose = make_target_pose(circle_center, radius, 0.0, desired_orientation)
    
    # 使用IK求解圆上起始点对应的关节角度
    q_circle_start, error, n_iter = solve_ik(
        pin_model, pin_data, frame_id, circle_start_pose, q,
        max_iterations=100, tolerance=1e-4, verbose=True
    )
    
    if q_circle_start is None:
        print(f"Warning: Failed to solve IK for circle start pose (error: {error:.6f})")
        print("Using initial configuration as fallback")
        q_circle_start = q.copy()
    else:
        print(f"✓ Solved IK for circle start in {n_iter} iterations (error: {error:.6f})")
    
    # 使用minimal jerk规划从初始位置到圆上起始点的平滑轨迹
    print(f"[Planning] Generating minimal jerk trajectory ({transition_duration}s)...")
    q_transition, qd_transition, qdd_transition = minimal_jerk_trajectory(
        q, q_circle_start, transition_duration, dt
    )
    print(f"✓ Generated transition trajectory with {len(q_transition)} waypoints")
    
    print(f"\n▶ Starting trajectory execution...")
    print(f"  Phase 1: Transition to circle start ({transition_duration}s)")
    print(f"  Phase 2: Circular motion (continuous)")

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # 初始化：从初始位置开始
        q_current = q.copy()
        mj_data.qpos[: mj_model.nq] = q_current
        mj_data.qvel[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)
        
        start_time = time.time()
        last_sync = time.time()
        phase = "transition"  # "transition" or "circle"
        transition_start_time = start_time
        transition_idx = 0
        
        # 可视化循环
        while viewer.is_running():
            elapsed = time.time() - start_time
            current_time = time.time()
            
            if phase == "transition":
                # 阶段1：执行过渡轨迹
                transition_elapsed = current_time - transition_start_time
                
                if transition_elapsed < transition_duration:
                    # 在过渡阶段，使用规划的轨迹
                    idx = min(int(transition_elapsed / dt), len(q_transition) - 1)
                    q_current = q_transition[idx]
                    qd_current = qd_transition[idx] if idx < len(qd_transition) else np.zeros_like(q_current)
                    
                    # 更新可视化
                    mj_data.qpos[: mj_model.nq] = q_current
                    mj_data.qvel[: mj_model.nq] = qd_current
                    mujoco.mj_forward(mj_model, mj_data)
                else:
                    # 过渡完成，切换到圆形轨迹阶段
                    phase = "circle"
                    circle_start_time = current_time
                    print(f"\n✓ Transition complete, starting circular motion...")
            
            if phase == "circle":
                # 阶段2：圆形轨迹跟踪
                circle_elapsed = current_time - circle_start_time
                t = (circle_elapsed / circle_period) % 1.0  # Normalized time [0, 1]
            target_pose = make_target_pose(circle_center, radius, t, desired_orientation)
            
            # 使用IK实时跟踪目标位姿
            # 使用多次迭代以获得更好的收敛
            for _ in range(5):  # 减少迭代次数，因为已经接近目标
                q_current, err = damped_ik_step(
                    pin_model, pin_data, frame_id, q_current, target_pose,
                    gain=0.5, damping=1e-2
                )
                
                # 更新可视化
            mj_data.qpos[: mj_model.nq] = q_current
            mj_data.qvel[:] = 0.0
            mujoco.mj_forward(mj_model, mj_data)
            
                # 定期打印状态
            if int(elapsed) != int(elapsed - dt) and int(elapsed) % 2 == 0:
                pin.forwardKinematics(pin_model, pin_data, q_current)
                pin.updateFramePlacements(pin_model, pin_data)
                actual_pose = pin_data.oMf[frame_id]
                
                # Position error
                pos_error = np.linalg.norm(actual_pose.translation - target_pose.translation)
                
                # Orientation error (angle between rotations)
                R_error = actual_pose.rotation.T @ target_pose.rotation
                angle_error = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
                
                phase_str = "TRANSITION" if phase == "transition" else "CIRCLE"
                print(f"  [{phase_str}] t={elapsed:.1f}s | pos_err: {pos_error*1000:.2f}mm, ori_err: {np.degrees(angle_error):.2f}°")
            
            viewer.sync()
            
            # 控制循环频率
            sleep_time = dt - (time.time() - last_sync)
            last_sync = time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
