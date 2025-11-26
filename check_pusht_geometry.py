import numpy as np
import matplotlib.pyplot as plt
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager

def check_tee_geometry():
    """
    Check the actual T shape dimensions and keypoint transformations
    """
    # Create environment
    env = PushTKeypointsEnv(render_size=96, render_action=False)
    
    # Reset environment to initialize block and other objects
    env.reset()
    
    # Get T shape parameters from add_tee
    scale = 30
    length = 4
    
    # Calculate T shape dimensions
    # Shape1 (horizontal bar): width = length*scale, height = scale
    shape1_width = length * scale  # 4 * 30 = 120
    shape1_height = scale  # 30
    
    # Shape2 (vertical bar): width = scale, height = length*scale
    shape2_width = scale  # 30
    shape2_height = length * scale  # 4 * 30 = 120
    
    print("=" * 60)
    print("T Shape Geometry:")
    print("=" * 60)
    print(f"Scale parameter: {scale}")
    print(f"Length parameter: {length}")
    print(f"\nShape1 (horizontal bar):")
    print(f"  Width: {shape1_width} pixels")
    print(f"  Height: {shape1_height} pixels")
    print(f"\nShape2 (vertical bar):")
    print(f"  Width: {shape2_width} pixels")
    print(f"  Height: {shape2_height} pixels")
    print(f"\nTotal T shape dimensions:")
    print(f"  Overall width: ~{shape1_width} pixels")
    print(f"  Overall height: ~{shape1_height + shape2_height} pixels")
    
    # Get actual vertices from the block
    block = env.block
    print(f"\nBlock position: {block.position}")
    print(f"Block angle: {block.angle} rad ({np.degrees(block.angle):.1f} deg)")
    
    # Get all vertices from block shapes
    all_vertices = []
    for idx, shape in enumerate(block.shapes):
        if hasattr(shape, 'get_vertices'):
            vertices = shape.get_vertices()
            # Transform from local to world coordinates
            world_vertices = [block.local_to_world(v) for v in vertices]
            all_vertices.append(world_vertices)
            print(f"\nShape {idx+1} vertices (world coordinates):")
            for i, v in enumerate(world_vertices):
                print(f"  Vertex {i}: ({v.x:.1f}, {v.y:.1f})")
            print(f"  Local vertices:")
            for i, v in enumerate(vertices):
                print(f"    Local {i}: ({v.x:.1f}, {v.y:.1f})")
    
    # Get keypoint manager
    kp_manager = env.kp_manager
    
    # Get local keypoint map
    local_kp_map = kp_manager.local_keypoint_map
    block_local_kps = local_kp_map['block']
    
    print("\n" + "=" * 60)
    print("Keypoint Coordinate Transformation:")
    print("=" * 60)
    print(f"\nLocal keypoints (in block's local coordinate system):")
    print(f"Shape: {block_local_kps.shape}")
    for i, kp in enumerate(block_local_kps):
        print(f"  KP{i+1} local: ({kp[0]:.2f}, {kp[1]:.2f})")
    
    # Get transformation from local to global
    tf_img_obj = PymunkKeypointManager.get_tf_img_obj(block)
    
    # Transform local keypoints to global
    global_kps = tf_img_obj(block_local_kps)
    
    print(f"\nGlobal keypoints (in world coordinate system):")
    print(f"Block pose: position=({block.position.x:.1f}, {block.position.y:.1f}), angle={block.angle:.2f} rad")
    for i, (local_kp, global_kp) in enumerate(zip(block_local_kps, global_kps)):
        print(f"  KP{i+1}: local({local_kp[0]:.2f}, {local_kp[1]:.2f}) -> global({global_kp[0]:.2f}, {global_kp[1]:.2f})")
    
    # Get transformation matrix info
    print(f"\nTransformation matrix:")
    print(f"  Translation: ({block.position.x:.1f}, {block.position.y:.1f})")
    print(f"  Rotation: {block.angle:.4f} rad ({np.degrees(block.angle):.1f} deg)")
    cos_a = np.cos(block.angle)
    sin_a = np.sin(block.angle)
    print(f"  Rotation matrix: [[{cos_a:.4f}, {-sin_a:.4f}],")
    print(f"                     [{sin_a:.4f}, {cos_a:.4f}]]")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Local coordinates
    ax1.set_title('T Shape - Local Coordinates', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X (local)', fontsize=12)
    ax1.set_ylabel('Y (local)', fontsize=12)
    
    # Draw T shape in local coordinates
    # Shape1 vertices (horizontal bar)
    shape1_verts_local = np.array([
        [-length*scale/2, scale],
        [length*scale/2, scale],
        [length*scale/2, 0],
        [-length*scale/2, 0]
    ])
    shape1_verts_local = np.vstack([shape1_verts_local, shape1_verts_local[0]])
    ax1.plot(shape1_verts_local[:, 0], shape1_verts_local[:, 1], 'b-', linewidth=2, label='Shape1 (horizontal)')
    
    # Shape2 vertices (vertical bar)
    shape2_verts_local = np.array([
        [-scale/2, scale],
        [-scale/2, length*scale],
        [scale/2, length*scale],
        [scale/2, scale]
    ])
    shape2_verts_local = np.vstack([shape2_verts_local, shape2_verts_local[0]])
    ax1.plot(shape2_verts_local[:, 0], shape2_verts_local[:, 1], 'g-', linewidth=2, label='Shape2 (vertical)')
    
    # Plot local keypoints
    colors = plt.cm.tab10(np.linspace(0, 1, len(block_local_kps)))
    for i, (kp, color) in enumerate(zip(block_local_kps, colors)):
        ax1.scatter(kp[0], kp[1], c=[color], s=100, marker='o', 
                   edgecolors='black', linewidths=2, zorder=5)
        ax1.annotate(f'P{i+1}', (kp[0], kp[1]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
    
    ax1.legend()
    ax1.set_xlim(-100, 100)
    ax1.set_ylim(-50, 150)
    
    # Right plot: Global coordinates
    ax2.set_title('T Shape - Global Coordinates', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X (world)', fontsize=12)
    ax2.set_ylabel('Y (world)', fontsize=12)
    ax2.set_xlim(0, 512)
    ax2.set_ylim(0, 512)
    
    # Transform and draw T shape in global coordinates
    # Shape1
    shape1_verts_global = np.array([block.local_to_world((v[0], v[1])) for v in shape1_verts_local])
    ax2.plot(shape1_verts_global[:, 0], shape1_verts_global[:, 1], 'b-', linewidth=2, label='Shape1')
    
    # Shape2
    shape2_verts_global = np.array([block.local_to_world((v[0], v[1])) for v in shape2_verts_local])
    ax2.plot(shape2_verts_global[:, 0], shape2_verts_global[:, 1], 'g-', linewidth=2, label='Shape2')
    
    # Plot global keypoints
    for i, (kp, color) in enumerate(zip(global_kps, colors)):
        ax2.scatter(kp[0], kp[1], c=[color], s=100, marker='o', 
                   edgecolors='black', linewidths=2, zorder=5)
        ax2.annotate(f'P{i+1}', (kp[0], kp[1]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
    
    # Plot block center
    ax2.scatter(block.position.x, block.position.y, c='red', s=150, 
               marker='X', edgecolors='black', linewidths=2, 
               label='Block Center', zorder=6)
    
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('pusht_geometry_analysis.png', dpi=150, bbox_inches='tight')
    print("\n" + "=" * 60)
    print("Visualization saved as: pusht_geometry_analysis.png")
    print("=" * 60)
    plt.show()
    
    return {
        'scale': scale,
        'length': length,
        'shape1_dimensions': (shape1_width, shape1_height),
        'shape2_dimensions': (shape2_width, shape2_height),
        'block_position': (block.position.x, block.position.y),
        'block_angle': block.angle,
        'local_keypoints': block_local_kps,
        'global_keypoints': global_kps,
        'transformation': {
            'translation': (block.position.x, block.position.y),
            'rotation': block.angle
        }
    }

if __name__ == '__main__':
    result = check_tee_geometry()

