"""
Real robot controller that bridges URDF joint angles to actual motor commands.
Reads motor calibration from motor_calibration.yaml and interfaces with RobStride motors.

Extended version with kinematics (FK/IK) and trajectory planning capabilities.
"""

from __future__ import annotations

import sys
import time
import numpy as np
import yaml
import pinocchio as pin
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driver.robstride import RobStrideMotor, connect
from utils.ik_utils import solve_ik, damped_ik_step
from utils.trajectory_utils import minimal_jerk_trajectory


@dataclass
class JointConfig:
    """
    Configuration for a single joint with motor-to-joint calibration parameters.
    
    Affine transform: joint_angle = direction * (motor_angle - offset)
    Inverse:          motor_angle = (joint_angle / direction) + offset
    
    where:
    - motor_angle: raw encoder reading from motor (hardware space)
    - joint_angle: calibrated joint angle in URDF (joint space)
    - direction: scaling factor (typically ±1.0 for direction reversal)
    - offset: zero-position offset in motor space
    """
    name: str
    motor_id: int
    direction: float
    offset: float
    
    def joint_to_motor(self, joint_angle: float) -> float:
        """
        Convert joint angle to motor command (Joint space → Motor space).
        Used when sending commands to actuators.
        
        Args:
            joint_angle: Desired joint angle in URDF/joint space (radians)
        
        Returns:
            Motor angle command in motor/hardware space (radians)
        """
        return (joint_angle / self.direction) + self.offset
    
    def motor_to_joint(self, motor_angle: float) -> float:
        """
        Convert motor reading to joint angle (Motor space → Joint space).
        Used when reading feedback from sensors/encoders.
        
        Args:
            motor_angle: Raw motor angle reading (radians)

        Returns:
            Calibrated joint angle in URDF/joint space (radians)
        """
        return self.direction * (motor_angle - self.offset)


class RobotController:
    """
    High-level robot controller that manages multiple motors.
    Extended with kinematics (FK/IK) and trajectory planning capabilities.

    Usage:
        # 初始化（带运动学功能）
        robot = RobotController.from_config(
            "config/motor_calibration.yaml",
            urdf_path="assets/urdf/Lagrange_01_sim.urdf",
            end_effector="tool_link"
        )
        robot.connect()

        # 关节空间控制
        robot.move_to_joint_positions([0.1, 0.5, 1.0, 0.0, 0.5, 0.0])

        # 笛卡尔空间控制
        current_pose = robot.get_end_effector_pose()
        target_pose = current_pose.copy()
        target_pose.translation += np.array([0.1, 0, 0])  # X方向移动10cm
        robot.move_to_pose(target_pose, duration=2.0)

        robot.close()
    """

    def __init__(
        self,
        can_interface: str,
        joint_configs: List[JointConfig],
        urdf_path: Optional[str | Path] = None,
        end_effector: str = "tool_link"
    ):
        # Hardware interface
        self.can_interface = can_interface
        self.joint_configs = joint_configs
        self.motors: Dict[str, RobStrideMotor] = {}
        self._connected = False

        # Kinematics (optional)
        self.urdf_path = Path(urdf_path) if urdf_path else None
        self.end_effector = end_effector
        self.pin_model = None
        self.pin_data = None
        self.ee_frame_id = None

        # Initialize kinematics if URDF provided
        if self.urdf_path and self.urdf_path.exists():
            self._init_kinematics()

    def _init_kinematics(self) -> None:
        """Initialize Pinocchio kinematics model."""
        self.pin_model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.pin_data = self.pin_model.createData()
        self.ee_frame_id = self.pin_model.getFrameId(self.end_effector)

        if self.ee_frame_id >= len(self.pin_model.frames):
            raise ValueError(f"End effector frame '{self.end_effector}' not found in URDF")
    
    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        urdf_path: Optional[str | Path] = None,
        end_effector: str = "tool_link"
    ) -> RobotController:
        """
        Load robot configuration from YAML file.

        Args:
            config_path: Path to motor_calibration.yaml
            urdf_path: Path to URDF file (optional, for kinematics)
            end_effector: Name of end effector frame in URDF

        Returns:
            RobotController instance
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        hardware = config['hardware']
        can_interface = hardware['can_interface']

        joint_configs = []
        for joint_data in hardware['joints']:
            joint_configs.append(JointConfig(
                name=joint_data['name'],
                motor_id=joint_data['motor_id'],
                direction=joint_data['direction'],
                offset=joint_data['offset']
            ))

        return cls(can_interface, joint_configs, urdf_path, end_effector)
    
    def connect(self) -> None:
        """Initialize connection to all motors."""
        if self._connected:
            print("[RobotController] Already connected")
            return
        
        print(f"[RobotController] Connecting to {len(self.joint_configs)} motors on {self.can_interface}...")
        
        for joint_cfg in self.joint_configs:
            try:
                motor = connect(
                    can_id=joint_cfg.motor_id,
                    interface=self.can_interface,
                    backend="python-can"  # or "socket"
                )
                self.motors[joint_cfg.name] = motor
                print(f"  ✓ {joint_cfg.name} (ID {joint_cfg.motor_id})")
            except Exception as e:
                print(f"  ✗ {joint_cfg.name} (ID {joint_cfg.motor_id}): {e}")
                raise
        
        self._connected = True
        print("[RobotController] All motors connected!")
    
    def close(self) -> None:
        """Close all motor connections."""
        for name, motor in self.motors.items():
            try:
                motor.disable()
                motor.close()
                print(f"[RobotController] Closed {name}")
            except Exception as e:
                print(f"[RobotController] Error closing {name}: {e}")
        self.motors.clear()
        self._connected = False
    
    def enable_all(self) -> None:
        """Enable all motors."""
        for motor in self.motors.values():
            motor.enable()
        print("[RobotController] All motors enabled")
    
    def disable_all(self) -> None:
        """Disable all motors."""
        for motor in self.motors.values():
            motor.disable()
        print("[RobotController] All motors disabled")
    
    def set_zero_all(self) -> None:
        """Set current position as zero for all motors."""
        for name, motor in self.motors.items():
            motor.set_zero()
            print(f"[RobotController] Set zero for {name}")
    
    def move_to_joint_positions(
        self,
        joint_angles: np.ndarray | List[float],
        torque: float | np.ndarray | List[float] = 0.0,
        speed: float | np.ndarray | List[float] = 0.0,
        kp: float | np.ndarray | List[float] = 10.0,
        kd: float | np.ndarray | List[float] = 1.0
    ) -> None:
        """
        Move all joints to specified joint angles using MOVE control mode (Joint space → Motor space).
        
        This method uses MOVE control mode (PD control) which provides:
        - Position control with PD gains
        - Torque feedforward
        - Velocity feedforward
        
        This method:
        1. Converts joint angles to motor commands using calibration parameters
        2. Sends MOVE control commands to all motors (torque, angle, speed, kp, kd)
        3. Motors automatically return status after receiving commands
        
        Args:
            joint_angles: Array of 6 joint angles [j1, j2, j3, j4, j5, j6] in radians
            torque: Torque feedforward (Nm). Can be a single float for all motors,
                    or an array/list of floats for individual motor configuration.
                    Array length must match the number of joints.
            speed: Velocity feedforward (rad/s). Can be a single float for all motors,
                   or an array/list of floats for individual motor configuration.
                   Array length must match the number of joints.
            kp: Position gain (proportional gain). Can be a single float for all motors,
                or an array/list of floats for individual motor configuration.
                Array length must match the number of joints.
            kd: Velocity gain (derivative gain). Can be a single float for all motors,
                or an array/list of floats for individual motor configuration.
                Array length must match the number of joints.
        
        Note:
            After sending control commands, motors automatically return status data.
            No need to call get_parameter() in control loops.
            MOVE mode is suitable for real-time trajectory tracking with PD control.
        
        Example:
            # Same parameters for all motors
            robot.move_to_joint_positions([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 
                                         torque=0.0, speed=0.0, kp=10.0, kd=1.0)
            
            # Individual parameters for each motor
            robot.move_to_joint_positions([0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                         torque=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         speed=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         kp=[10.0, 12.0, 8.0, 15.0, 10.0, 10.0],
                                         kd=[1.0, 1.2, 0.8, 1.5, 1.0, 1.0])
        """
        if not self._connected:
            raise RuntimeError("Robot not connected. Call connect() first.")
        
        num_joints = len(self.joint_configs)
        if len(joint_angles) != num_joints:
            raise ValueError(f"Expected {num_joints} angles, got {len(joint_angles)}")
        
        # Convert torque to array if it's a single value
        if isinstance(torque, (int, float)):
            torques = np.full(num_joints, float(torque))
        else:
            torques = np.asarray(torque, dtype=float)
            if len(torques) != num_joints:
                raise ValueError(f"Expected {num_joints} torque values, got {len(torques)}")
        
        # Convert speed to array if it's a single value
        if isinstance(speed, (int, float)):
            speeds = np.full(num_joints, float(speed))
        else:
            speeds = np.asarray(speed, dtype=float)
            if len(speeds) != num_joints:
                raise ValueError(f"Expected {num_joints} speed values, got {len(speeds)}")
        
        # Convert kp to array if it's a single value
        if isinstance(kp, (int, float)):
            kps = np.full(num_joints, float(kp))
        else:
            kps = np.asarray(kp, dtype=float)
            if len(kps) != num_joints:
                raise ValueError(f"Expected {num_joints} kp values, got {len(kps)}")
        
        # Convert kd to array if it's a single value
        if isinstance(kd, (int, float)):
            kds = np.full(num_joints, float(kd))
        else:
            kds = np.asarray(kd, dtype=float)
            if len(kds) != num_joints:
                raise ValueError(f"Expected {num_joints} kd values, got {len(kds)}")
        
        # Convert joint angles to motor commands (Joint space → Motor space)
        for i, (joint_cfg, joint_angle) in enumerate(zip(self.joint_configs, joint_angles)):
            motor_cmd = joint_cfg.joint_to_motor(joint_angle)
            motor = self.motors[joint_cfg.name]
            motor.move_control(
                torque=torques[i],
                angle=motor_cmd,
                speed=speeds[i],
                kp=kps[i],
                kd=kds[i]
            )

    def get_joint_positions(self, request_update: bool = False) -> np.ndarray:
        """
        Read current joint angles from all motors (Motor space → Joint space).
        
        This method:
        1. (Optional) Requests motor encoders to send data if in read-only mode
        2. Reads raw motor angles from motor feedback
        3. Converts to calibrated joint angles using calibration parameters
        
        Args:
            request_update: Whether to actively request motor feedback before reading.
                - True: Read-only mode (no control commands sent)
                        Motors need explicit get_parameter() request
                - False: Control loop mode (default)
                         Motors auto-report after receiving commands
        
        Returns:
            Array of 6 calibrated joint angles [j1, j2, j3, j4, j5, j6] in radians
        """
        if not self._connected:
            raise RuntimeError("Robot not connected. Call connect() first.")
        
        # Request motor feedback if in read-only mode
        if request_update:
            import time
            for joint_cfg in self.joint_configs:
                motor = self.motors[joint_cfg.name]
                motor.get_parameter(0x7019)  # Request mechanical angle (0x7019)
            time.sleep(0.001)  # Wait for CAN responses
        
        # Read and convert motor angles to joint angles (Motor space → Joint space)
        joint_angles = []
        for joint_cfg in self.joint_configs:
            motor = self.motors[joint_cfg.name]
            info = motor.get_info()
            motor_reading = info.angle
            joint_angle = joint_cfg.motor_to_joint(motor_reading)
            joint_angles.append(joint_angle)

        return np.array(joint_angles)

    def get_joint_info(self, joint_name: str):
        """Get motor info for a specific joint."""
        if joint_name not in self.motors:
            raise ValueError(f"Unknown joint: {joint_name}")
        return self.motors[joint_name].get_info()

    # ========================================================================
    # Kinematics Methods (FK/IK)
    # ========================================================================

    def forward_kinematics(self, q: Optional[np.ndarray] = None) -> pin.SE3:
        """
        Compute forward kinematics for end effector.

        Args:
            q: Joint angles (radians). If None, uses current robot position.

        Returns:
            SE3 pose of end effector (position + orientation)

        Raises:
            RuntimeError: If kinematics not initialized (no URDF provided)
        """
        if self.pin_model is None:
            raise RuntimeError("Kinematics not initialized. Provide urdf_path in constructor.")

        if q is None:
            q = self.get_joint_positions(request_update=True)

        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        return self.pin_data.oMf[self.ee_frame_id]

    def get_end_effector_pose(self) -> pin.SE3:
        """
        Get current end effector pose.
        Convenience method that calls forward_kinematics with current joint positions.

        Returns:
            SE3 pose of end effector
        """
        return self.forward_kinematics(q=None)

    def inverse_kinematics(
        self,
        target_pose: pin.SE3,
        q_init: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        fixed_iterations: Optional[int] = None
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Solve inverse kinematics to reach target pose.

        Args:
            target_pose: Desired end effector pose (SE3)
            q_init: Initial guess for joint angles. If None, uses current position.
            max_iterations: Maximum IK iterations (used when fixed_iterations is None)
            tolerance: Position error tolerance (meters)
            fixed_iterations: If specified, perform exactly this many iterations regardless of convergence.
                             If None, iterate until convergence or max_iterations.
                             Useful for real-time control where you want fixed computation time.

        Returns:
            Tuple of (joint_angles, error, success)
                - joint_angles: Solution joint angles (radians)
                - error: Final position error (meters)
                - success: True if error < tolerance (or always True if fixed_iterations is specified)

        Raises:
            RuntimeError: If kinematics not initialized

        Example:
            # Converge until tolerance is met (or max_iterations reached)
            q, error, success = controller.inverse_kinematics(target_pose)
            
            # Perform exactly 10 iterations (for real-time control)
            q, error, success = controller.inverse_kinematics(target_pose, fixed_iterations=10)
        """
        if self.pin_model is None:
            raise RuntimeError("Kinematics not initialized. Provide urdf_path in constructor.")

        if q_init is None:
            q_init = self.get_joint_positions(request_update=True)

        q_solution, error, n_iter = solve_ik(
            self.pin_model,
            self.pin_data,
            self.ee_frame_id,
            target_pose,
            q_init,
            max_iterations=max_iterations,
            tolerance=tolerance,
            verbose=False,
            fixed_iterations=fixed_iterations
        )

        # If using fixed_iterations, always return success (we got a result)
        if fixed_iterations is not None:
            success = (q_solution is not None)
        else:
            success = (error < tolerance) and (q_solution is not None)
        
        return q_solution, error, success

    # ========================================================================
    # High-Level Motion Commands
    # ========================================================================

    def move_to_pose(
        self,
        target_pose: pin.SE3,
        duration: float = 2.0,
        control_dt: float = 0.01,
        torque: float | np.ndarray | List[float] = 0.0,
        speed: float | np.ndarray | List[float] = 0.0,
        kp: float | np.ndarray | List[float] = 10.0,
        kd: float | np.ndarray | List[float] = 1.0,
        on_update: Optional[Callable] = None
    ) -> bool:
        """
        Move end effector to target pose using smooth joint-space trajectory with MOVE control mode.

        Args:
            target_pose: Desired end effector pose (SE3)
            duration: Motion duration (seconds)
            control_dt: Control loop timestep (seconds)
            torque: Torque feedforward (Nm). Can be a single float for all motors,
                    or an array/list of floats for individual motor configuration.
            speed: Velocity feedforward (rad/s). Can be a single float for all motors,
                   or an array/list of floats for individual motor configuration.
            kp: Position gain (proportional gain). Can be a single float for all motors,
                or an array/list of floats for individual motor configuration.
            kd: Velocity gain (derivative gain). Can be a single float for all motors,
                or an array/list of floats for individual motor configuration.
            on_update: Optional callback(elapsed_time, q_current, q_target, error)

        Returns:
            True if successful, False if IK failed

        Example:
            target = robot.get_end_effector_pose()
            target.translation += np.array([0.1, 0, 0])  # Move 10cm in X
            robot.move_to_pose(target, duration=2.0)
        """
        if self.pin_model is None:
            raise RuntimeError("Kinematics not initialized. Provide urdf_path in constructor.")

        # Get current state
        q_start = self.get_joint_positions(request_update=True)

        # Solve IK for target
        q_goal, ik_error, success = self.inverse_kinematics(target_pose, q_start)
        if not success:
            print(f"[RobotController] IK failed with error: {ik_error:.6f}m")
            return False

        # Plan trajectory
        q_traj, qd_traj, qdd_traj = minimal_jerk_trajectory(
            q_start, q_goal, duration, control_dt
        )

        print(f"[RobotController] Executing trajectory ({len(q_traj)} points, {duration}s)")

        # Execute trajectory
        start_time = time.time()
        for i, q_target in enumerate(q_traj):
            loop_start = time.time()
            
            # Calculate velocity feedforward from trajectory
            qd_target = qd_traj[i] if i < len(qd_traj) else np.zeros_like(q_target)

            # Send command with velocity feedforward
            self.move_to_joint_positions(q_target, torque=torque, speed=qd_target, kp=kp, kd=kd)

            # Callback
            if on_update:
                q_current = self.get_joint_positions(request_update=False)
                elapsed = time.time() - start_time
                error = np.linalg.norm(q_current - q_target)
                on_update(elapsed, q_current, q_target, error)

            # Maintain control frequency
            elapsed = time.time() - loop_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print(f"[RobotController] Trajectory execution completed")
        return True


# Example usage
if __name__ == "__main__":
    import time
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "motor_calibration.yaml"
    controller = RobotController.from_config(config_path)
    
    try:
        # Connect to motors
        controller.connect()
        controller.enable_all()
        
        # Read current position
        current_pos = controller.get_joint_positions(request_update=True)
        print(f"\nCurrent joint position: {current_pos}")
        
        # Move to a target position using MOVE control mode
        target_joints = np.array([0.0, 1.0, 1.5, 0.0, 0.5, 0.0])
        print(f"Moving to target: {target_joints}")
        controller.move_to_joint_positions(
            target_joints,
            torque=0.0,      # Torque feedforward
            speed=0.0,       # Velocity feedforward
            kp=10.0,         # Position gain
            kd=1.0           # Velocity gain
        )
        
        # Wait for motion to complete
        time.sleep(3.0)
        
        # Check final position
        final_pos = controller.get_joint_positions(request_update=True)
        print(f"Final joint position: {final_pos}")
        print(f"Error: {np.linalg.norm(final_pos - target_joints):.4f} rad")
        
    except KeyboardInterrupt:
        print("\n[RobotController] Interrupted by user")
    except Exception as e:
        print(f"[RobotController] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always disable and close
        controller.disable_all()
        controller.close()

