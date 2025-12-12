"""
Inverse kinematics utilities for robot motion planning.

Provides IK solvers for pose control using Pinocchio.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
from typing import Tuple, Optional


def damped_ik_step(
    model: pin.Model,
    data: pin.Data,
    frame_id: int,
    q: np.ndarray,
    target_pose: pin.SE3,
    gain: float = 0.5,
    damping: float = 1e-2,
) -> Tuple[np.ndarray, float]:
    """
    One step of damped least-squares IK for full 6DOF pose control (position + orientation).
    
    Args:
        model: Pinocchio model
        data: Pinocchio data
        frame_id: Target frame ID
        q: Current joint configuration
        target_pose: Target SE3 pose (position + rotation)
        gain: IK step gain
        damping: Damping factor for numerical stability
    
    Returns:
        (q_next, error): Updated joint configuration and pose error magnitude
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    
    # Current end-effector pose
    current_pose = data.oMf[frame_id]
    
    # Pose error in SE3 (6D: 3 for position, 3 for orientation)
    # Compute error as log of relative transformation
    error_SE3 = current_pose.inverse() * target_pose
    err = pin.log6(error_SE3).vector  # 6D error vector
    
    # Get full Jacobian (6 x nq) in local frame
    J = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL)
    
    # Damped least-squares solution
    H = J @ J.T + (damping ** 2) * np.eye(6)
    dq = gain * J.T @ np.linalg.solve(H, err)
    
    # Integrate and clip to joint limits
    q_next = q + dq
    lower, upper = model.lowerPositionLimit, model.upperPositionLimit
    mask = np.isfinite(lower) & np.isfinite(upper)
    q_next[mask] = np.clip(q_next[mask], lower[mask], upper[mask])
    
    return q_next, float(np.linalg.norm(err))


def solve_ik(
    model: pin.Model,
    data: pin.Data,
    frame_id: int,
    target_pose: pin.SE3,
    q_init: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    gain: float = 0.5,
    damping: float = 1e-2,
    verbose: bool = False,
    fixed_iterations: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], float, int]:
    """
    Solve inverse kinematics to reach target pose.
    
    Args:
        model: Pinocchio model
        data: Pinocchio data
        frame_id: Target frame ID
        target_pose: Target SE3 pose (position + rotation)
        q_init: Initial joint configuration
        max_iterations: Maximum number of iterations (used when fixed_iterations is None)
        tolerance: Convergence tolerance (error threshold)
        gain: IK step gain
        damping: Damping factor for numerical stability
        verbose: Print convergence information
        fixed_iterations: If specified, perform exactly this many iterations regardless of convergence.
                         If None, iterate until convergence or max_iterations.
    
    Returns:
        (q_solution, final_error, n_iterations):
            - q_solution: Joint configuration that reaches target (None if failed and fixed_iterations is None)
            - final_error: Final pose error magnitude
            - n_iterations: Number of iterations used
    """
    q = q_init.copy()
    
    # If fixed_iterations is specified, perform exactly that many iterations
    if fixed_iterations is not None:
        iterations_to_use = fixed_iterations
        check_convergence = False
    else:
        iterations_to_use = max_iterations
        check_convergence = True
    
    for i in range(iterations_to_use):
        q, error = damped_ik_step(model, data, frame_id, q, target_pose, gain, damping)
        
        # Check convergence only if not using fixed iterations
        if check_convergence and error < tolerance:
            if verbose:
                print(f"IK converged in {i + 1} iterations with error {error:.6f}")
            return q, error, i + 1
        
        if verbose and (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}: error = {error:.6f}")
    
    # If using fixed iterations, always return the result
    if fixed_iterations is not None:
        if verbose:
            print(f"IK completed {fixed_iterations} iterations with error {error:.6f}")
        return q, error, fixed_iterations
    
    # Otherwise, check if converged
    if verbose:
        print(f"IK did not converge after {max_iterations} iterations. Final error: {error:.6f}")
    
    return None, error, max_iterations


def solve_ik_position_only(
    model: pin.Model,
    data: pin.Data,
    frame_id: int,
    target_position: np.ndarray,
    q_init: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    gain: float = 0.5,
    damping: float = 1e-2,
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], float, int]:
    """
    Solve inverse kinematics for position only (ignore orientation).
    
    Args:
        model: Pinocchio model
        data: Pinocchio data
        frame_id: Target frame ID
        target_position: Target position [x, y, z]
        q_init: Initial joint configuration
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance (position error threshold in meters)
        gain: IK step gain
        damping: Damping factor for numerical stability
        verbose: Print convergence information
    
    Returns:
        (q_solution, final_error, n_iterations):
            - q_solution: Joint configuration that reaches target position (None if failed)
            - final_error: Final position error magnitude (meters)
            - n_iterations: Number of iterations used
    """
    q = q_init.copy()
    
    for i in range(max_iterations):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        
        # Current position
        current_position = data.oMf[frame_id].translation
        
        # Position error
        pos_err = target_position - current_position
        
        # Get position Jacobian (3 x nq)
        J = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL)
        J_pos = J[:3, :]  # Position part only
        
        # Damped least-squares solution
        H = J_pos @ J_pos.T + (damping ** 2) * np.eye(3)
        dq = gain * J_pos.T @ np.linalg.solve(H, pos_err)
        
        # Integrate and clip to joint limits
        q = q + dq
        lower, upper = model.lowerPositionLimit, model.upperPositionLimit
        mask = np.isfinite(lower) & np.isfinite(upper)
        q[mask] = np.clip(q[mask], lower[mask], upper[mask])
        
        error = np.linalg.norm(pos_err)
        
        if error < tolerance:
            if verbose:
                print(f"Position IK converged in {i + 1} iterations with error {error:.6f} m")
            return q, error, i + 1
        
        if verbose and (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}: position error = {error:.6f} m")
    
    if verbose:
        print(f"Position IK did not converge after {max_iterations} iterations. Final error: {error:.6f} m")
    
    return None, error, max_iterations


def ik_trajectory(
    model: pin.Model,
    data: pin.Data,
    frame_id: int,
    target_poses: list[pin.SE3],
    q_init: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-3,
    gain: float = 0.5,
    damping: float = 1e-2,
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], list[float]]:
    """
    Solve IK for a sequence of target poses, using previous solution as initialization.
    
    Args:
        model: Pinocchio model
        data: Pinocchio data
        frame_id: Target frame ID
        target_poses: List of target SE3 poses
        q_init: Initial joint configuration
        max_iterations: Maximum iterations per pose
        tolerance: Convergence tolerance
        gain: IK step gain
        damping: Damping factor
        verbose: Print convergence information
    
    Returns:
        (q_trajectory, errors):
            - q_trajectory: Joint configurations for each pose (n_poses, n_dof) or None if failed
            - errors: List of final errors for each pose
    """
    q_traj = []
    errors = []
    q_current = q_init.copy()
    
    for i, target_pose in enumerate(target_poses):
        q_sol, error, n_iter = solve_ik(
            model, data, frame_id, target_pose, q_current,
            max_iterations=max_iterations, tolerance=tolerance,
            gain=gain, damping=damping, verbose=verbose and (i % 10 == 0)
        )
        
        if q_sol is None:
            if verbose:
                print(f"Failed to solve IK for pose {i}/{len(target_poses)}")
            return None, errors
        
        q_traj.append(q_sol)
        errors.append(error)
        q_current = q_sol  # Use solution as initialization for next pose
    
    return np.array(q_traj), errors

