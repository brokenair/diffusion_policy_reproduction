"""
Trajectory planning utilities for robot motion.

Includes minimal jerk trajectory planning for smooth motion generation.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def minimal_jerk_trajectory(
    q_start: np.ndarray,
    q_end: np.ndarray,
    duration: float,
    dt: float,
    qd_start: np.ndarray | None = None,
    qd_end: np.ndarray | None = None,
    qdd_start: np.ndarray | None = None,
    qdd_end: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a minimal jerk trajectory from start to end configuration.
    
    Minimal jerk minimizes the integral of squared jerk (third derivative),
    resulting in smooth, natural-looking motion.
    
    Args:
        q_start: Starting joint configuration (n_dof,)
        q_end: Target joint configuration (n_dof,)
        duration: Total trajectory duration in seconds
        dt: Time step in seconds
        qd_start: Starting joint velocity (optional, defaults to zero)
        qd_end: Target joint velocity (optional, defaults to zero)
        qdd_start: Starting joint acceleration (optional, defaults to zero)
        qdd_end: Target joint acceleration (optional, defaults to zero)
    
    Returns:
        Tuple of (q_traj, qd_traj, qdd_traj):
            - q_traj: Joint positions over time (n_steps, n_dof)
            - qd_traj: Joint velocities over time (n_steps, n_dof)
            - qdd_traj: Joint accelerations over time (n_steps, n_dof)
    
    Reference:
        Flash, T., & Hogan, N. (1985). The coordination of arm movements:
        an experimentally confirmed mathematical model.
    """
    n_dof = len(q_start)
    n_steps = int(np.ceil(duration / dt)) + 1
    t = np.linspace(0, duration, n_steps)
    tau = t / duration  # Normalized time [0, 1]
    
    # Default boundary conditions
    if qd_start is None:
        qd_start = np.zeros(n_dof)
    if qd_end is None:
        qd_end = np.zeros(n_dof)
    if qdd_start is None:
        qdd_start = np.zeros(n_dof)
    if qdd_end is None:
        qdd_end = np.zeros(n_dof)
    
    # Compute trajectory coefficients for minimal jerk
    # Using 5th order polynomial: q(tau) = a0 + a1*tau + a2*tau^2 + a3*tau^3 + a4*tau^4 + a5*tau^5
    # Boundary conditions:
    #   q(0) = q_start, q(1) = q_end
    #   qd(0) = qd_start, qd(1) = qd_end
    #   qdd(0) = qdd_start, qdd(1) = qdd_end
    
    # Solve for coefficients
    # q(tau) = q_start + (q_end - q_start) * (10*tau^3 - 15*tau^4 + 6*tau^5)
    #         + qd_start * duration * (tau - 6*tau^3 + 8*tau^4 - 3*tau^5)
    #         + qd_end * duration * (-4*tau^3 + 7*tau^4 - 3*tau^5)
    #         + qdd_start * duration^2 * (0.5*tau^2 - 1.5*tau^3 + 1.5*tau^4 - 0.5*tau^5)
    #         + qdd_end * duration^2 * (0.5*tau^3 - tau^4 + 0.5*tau^5)
    
    tau2 = tau * tau
    tau3 = tau * tau2
    tau4 = tau * tau3
    tau5 = tau * tau4
    
    # Position trajectory
    q_traj = np.zeros((n_steps, n_dof))
    for i in range(n_dof):
        # Base trajectory (minimal jerk)
        base = 10 * tau3 - 15 * tau4 + 6 * tau5
        q_traj[:, i] = q_start[i] + (q_end[i] - q_start[i]) * base
        
        # Add velocity boundary conditions
        vel_start_term = qd_start[i] * duration * (tau - 6 * tau3 + 8 * tau4 - 3 * tau5)
        vel_end_term = qd_end[i] * duration * (-4 * tau3 + 7 * tau4 - 3 * tau5)
        q_traj[:, i] += vel_start_term + vel_end_term
        
        # Add acceleration boundary conditions
        acc_start_term = qdd_start[i] * duration ** 2 * (0.5 * tau2 - 1.5 * tau3 + 1.5 * tau4 - 0.5 * tau5)
        acc_end_term = qdd_end[i] * duration ** 2 * (0.5 * tau3 - tau4 + 0.5 * tau5)
        q_traj[:, i] += acc_start_term + acc_end_term
    
    # Velocity trajectory (derivative)
    qd_traj = np.zeros((n_steps, n_dof))
    for i in range(n_dof):
        # Base velocity
        base_d = (30 * tau2 - 60 * tau3 + 30 * tau4) / duration
        qd_traj[:, i] = (q_end[i] - q_start[i]) * base_d
        
        # Velocity boundary conditions
        vel_start_term_d = qd_start[i] * (1 - 18 * tau2 + 32 * tau3 - 15 * tau4)
        vel_end_term_d = qd_end[i] * (-12 * tau2 + 28 * tau3 - 15 * tau4)
        qd_traj[:, i] += vel_start_term_d + vel_end_term_d
        
        # Acceleration boundary conditions
        acc_start_term_d = qdd_start[i] * duration * (tau - 4.5 * tau2 + 6 * tau3 - 2.5 * tau4)
        acc_end_term_d = qdd_end[i] * duration * (1.5 * tau2 - 4 * tau3 + 2.5 * tau4)
        qd_traj[:, i] += acc_start_term_d + acc_end_term_d
    
    # Acceleration trajectory (second derivative)
    qdd_traj = np.zeros((n_steps, n_dof))
    for i in range(n_dof):
        # Base acceleration
        base_dd = (60 * tau - 180 * tau2 + 120 * tau3) / (duration ** 2)
        qdd_traj[:, i] = (q_end[i] - q_start[i]) * base_dd
        
        # Velocity boundary conditions
        vel_start_term_dd = qd_start[i] * (-36 * tau + 96 * tau2 - 60 * tau3) / duration
        vel_end_term_dd = qd_end[i] * (-24 * tau + 84 * tau2 - 60 * tau3) / duration
        qdd_traj[:, i] += vel_start_term_dd + vel_end_term_dd
        
        # Acceleration boundary conditions
        acc_start_term_dd = qdd_start[i] * (1 - 9 * tau + 18 * tau2 - 10 * tau3)
        acc_end_term_dd = qdd_end[i] * (3 * tau - 12 * tau2 + 10 * tau3)
        qdd_traj[:, i] += acc_start_term_dd + acc_end_term_dd
    
    return q_traj, qd_traj, qdd_traj


def minimal_jerk_waypoints(
    waypoints: list[np.ndarray],
    durations: list[float] | np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate minimal jerk trajectory through multiple waypoints.
    
    Args:
        waypoints: List of joint configurations (n_waypoints, n_dof)
        durations: Duration for each segment (n_waypoints - 1,)
        dt: Time step in seconds
    
    Returns:
        Tuple of (t_traj, q_traj, qd_traj, qdd_traj):
            - t_traj: Time array (n_steps,)
            - q_traj: Joint positions over time (n_steps, n_dof)
            - qd_traj: Joint velocities over time (n_steps, n_dof)
            - qdd_traj: Joint accelerations over time (n_steps, n_dof)
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints")
    
    if len(durations) != len(waypoints) - 1:
        raise ValueError(f"Need {len(waypoints) - 1} durations for {len(waypoints)} waypoints")
    
    n_dof = len(waypoints[0])
    all_q = []
    all_qd = []
    all_qdd = []
    all_t = []
    
    current_time = 0.0
    
    for i in range(len(waypoints) - 1):
        q_start = waypoints[i]
        q_end = waypoints[i + 1]
        duration = durations[i]
        
        # Compute velocities at waypoints for smooth connection
        if i == 0:
            qd_start = None  # Start from rest
        else:
            # Use velocity from previous segment end
            qd_start = all_qd[-1] if len(all_qd) > 0 else None
        
        qd_end = None  # End at rest (or can be computed for smooth connection)
        
        q_seg, qd_seg, qdd_seg = minimal_jerk_trajectory(
            q_start, q_end, duration, dt, qd_start=qd_start, qd_end=qd_end
        )
        
        t_seg = np.linspace(current_time, current_time + duration, len(q_seg))
        
        if i > 0:
            # Remove duplicate waypoint (except first segment)
            all_q.append(q_seg[1:])
            all_qd.append(qd_seg[1:])
            all_qdd.append(qdd_seg[1:])
            all_t.append(t_seg[1:])
        else:
            all_q.append(q_seg)
            all_qd.append(qd_seg)
            all_qdd.append(qdd_seg)
            all_t.append(t_seg)
        
        current_time += duration
    
    t_traj = np.concatenate(all_t)
    q_traj = np.concatenate(all_q, axis=0)
    qd_traj = np.concatenate(all_qd, axis=0)
    qdd_traj = np.concatenate(all_qdd, axis=0)
    
    return t_traj, q_traj, qd_traj, qdd_traj

