import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from diffusion_policy.dataset.pusht_dataset import PushTLowdimDataset

def animate_pusht_episode(zarr_path='data/pusht/pusht_cchi_v7_replay.zarr', fps=10):
    """
    Load PushT dataset, get first episode and animate it frame by frame
    """
    # Create dataset
    dataset = PushTLowdimDataset(
        zarr_path=zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0
    )
    
    # Get replay buffer directly
    replay_buffer = dataset.replay_buffer
    
    # Get first episode range
    episode_ends = replay_buffer.episode_ends[:]
    if len(episode_ends) == 0:
        print("No episodes found in dataset!")
        return
    
    # First episode: from 0 to episode_ends[0]
    episode_start = 0
    episode_end = episode_ends[0]
    episode_length = episode_end - episode_start
    
    print(f"First episode: steps {episode_start} to {episode_end} (length: {episode_length})")
    
    # Goal pose is constant in pusht environment: [256, 256, np.pi/4]
    goal_pose = np.array([256, 256, np.pi/4])  # x, y, theta
    goal_pos = goal_pose[:2]
    goal_angle = goal_pose[2]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Set axis range (pusht environment window size is 512x512)
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('PushT Dataset Animation - Episode 0', fontsize=14, fontweight='bold')
    
    # Draw goal zone (T-shaped region, similar to block)
    # Block is T-shaped: 50x100 box
    # We'll draw a semi-transparent T-shape to represent the goal
    
    # T-shape dimensions (from pusht_env: 50x100)
    w, h = 50, 100
    # Create T-shape vertices (centered at origin, then rotated and translated)
    t_vertices = np.array([
        [-w/2, -h/2],  # bottom left
        [w/2, -h/2],   # bottom right
        [w/2, -h/6],   # right middle
        [w/6, -h/6],   # right of top bar
        [w/6, h/2],    # top right
        [-w/6, h/2],   # top left
        [-w/6, -h/6],  # left of top bar
        [-w/2, -h/6],  # left middle
        [-w/2, -h/2]   # back to start
    ])
    
    # Rotate and translate to goal position
    cos_a = np.cos(goal_angle)
    sin_a = np.sin(goal_angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_vertices = (rotation_matrix @ t_vertices.T).T
    translated_vertices = rotated_vertices + goal_pos
    
    goal_polygon = Polygon(translated_vertices, closed=True, 
                          facecolor='lightgreen', edgecolor='green', 
                          linewidth=2, alpha=0.3, zorder=1, label='Goal Zone')
    ax.add_patch(goal_polygon)
    
    # Goal center marker
    goal_scatter = ax.scatter(goal_pos[0], goal_pos[1], c='lime', s=200, 
                              marker='X', edgecolors='green', linewidths=2,
                              label='Goal', zorder=3)
    goal_ann = ax.annotate('Goal', (goal_pos[0], goal_pos[1]), 
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=10, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
    
    # Initialize plot elements (will be updated in animation)
    colors_kp = plt.cm.tab10(np.linspace(0, 1, 9))
    
    # Keypoints scatter plots
    kp_scatters = []
    kp_annotations = []
    for i in range(9):
        scatter = ax.scatter([], [], c=[colors_kp[i]], s=150, 
                  marker='o', edgecolors='black', linewidths=2, 
                            label=f'Point{i+1}', zorder=5)
        kp_scatters.append(scatter)
        ann = ax.annotate(f'P{i+1}', (0, 0), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                         visible=False)
        kp_annotations.append(ann)
    
    # Agent, block, action scatter plots
    agent_scatter = ax.scatter([], [], c='blue', s=200, 
              marker='s', edgecolors='black', linewidths=2,
                               label='Agent', zorder=6)
    agent_ann = ax.annotate('Agent', (0, 0), 
               xytext=(5, 5), textcoords='offset points',
               fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8),
                           visible=False)
    
    block_scatter = ax.scatter([], [], c='green', s=200, 
              marker='D', edgecolors='black', linewidths=2,
                              label='Block', zorder=6)
    block_ann = ax.annotate('Block', (0, 0), 
               xytext=(5, 5), textcoords='offset points',
               fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                           visible=False)
    
    action_scatter = ax.scatter([], [], c='red', s=200, 
              marker='*', edgecolors='black', linewidths=2,
                               label='Action', zorder=6)
    action_ann = ax.annotate('Action', (0, 0), 
               xytext=(5, 5), textcoords='offset points',
               fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8),
                            visible=False)
    
    # Arrow (will be redrawn each frame)
    arrow = None
    
    # Info text
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add legend (goal polygon is already added to axes)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    def animate(frame):
        # Get data for current frame
        step_idx = episode_start + frame
        
        if step_idx >= episode_end:
            return
        
        # Get raw data
        keypoint = replay_buffer['keypoint'][step_idx]  # (9, 2)
        state = replay_buffer['state'][step_idx]  # (5,)
        action = replay_buffer['action'][step_idx]  # (2,)
        
        # Parse data
        agent_pos = state[:2]
        block_pos = state[2:4]
        block_angle = state[4]
        keypoints = keypoint
        action_pos = action
        
        # Update keypoints
        for i in range(9):
            kp = keypoints[i]
            kp_scatters[i].set_offsets([kp[0], kp[1]])
            kp_annotations[i].xy = (kp[0], kp[1])
            kp_annotations[i].set_visible(True)
        
        # Update agent
        agent_scatter.set_offsets([agent_pos[0], agent_pos[1]])
        agent_ann.xy = (agent_pos[0], agent_pos[1])
        agent_ann.set_visible(True)
        
        # Update block
        block_scatter.set_offsets([block_pos[0], block_pos[1]])
        block_ann.xy = (block_pos[0], block_pos[1])
        block_ann.set_visible(True)
        
        # Update action
        action_scatter.set_offsets([action_pos[0], action_pos[1]])
        action_ann.xy = (action_pos[0], action_pos[1])
        action_ann.set_visible(True)
        
        # Remove old arrow and draw new one
        nonlocal arrow
        if arrow is not None:
            arrow.remove()
        arrow = ax.arrow(agent_pos[0], agent_pos[1], 
                        action_pos[0] - agent_pos[0], action_pos[1] - agent_pos[1],
                        head_width=15, head_length=15, fc='red', ec='red', 
                        linewidth=2, alpha=0.6, zorder=4)
        
        # Update info text
        info_text.set_text(f"""Frame: {frame}/{episode_length-1} | Step: {step_idx}
Block Angle: {block_angle:.2f} rad

State:
  Agent XY = [{agent_pos[0]:.1f}, {agent_pos[1]:.1f}]
  Block XY = [{block_pos[0]:.1f}, {block_pos[1]:.1f}]
  Block Angle = {block_angle:.2f}

Action:
  Target = [{action_pos[0]:.1f}, {action_pos[1]:.1f}]

Goal (Constant):
  Goal XY = [{goal_pos[0]:.1f}, {goal_pos[1]:.1f}]
  Goal Angle = {goal_angle:.2f} rad""")
        
        return (kp_scatters + [agent_scatter, block_scatter, action_scatter, 
                               arrow, info_text] + kp_annotations + [agent_ann, block_ann, action_ann])
    
    # Create animation
    # interval in milliseconds for 10 fps = 1000/10 = 100ms
    interval_ms = 1000 / fps
    anim = animation.FuncAnimation(fig, animate, frames=episode_length, 
                                  interval=interval_ms, blit=False, repeat=True)
    
    print(f"Animation created: {episode_length} frames at {fps} fps")
    print("Close the window to stop the animation")
    
    plt.tight_layout()
    plt.show()
    
    return anim

if __name__ == '__main__':
    # Run animation
    animate_pusht_episode(fps=10)