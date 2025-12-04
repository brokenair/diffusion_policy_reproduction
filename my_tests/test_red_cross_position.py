"""
测试红叉位置是否正确。

运行后会在 plots/red_cross_test/ 目录下生成多张图片，
每张图片对应不同的 action 位置，用于验证坐标映射是否正确。
"""

import numpy as np
import cv2
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion_policy.env.pusht.pusht_mujoco_env import PushTMujocoImageEnv


def main():
    # 创建输出目录
    output_dir = "plots/red_cross_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建环境
    env = PushTMujocoImageEnv(render_size=480)
    env.reset()
    
    # 测试不同的 action 位置
    # action 范围是 [-0.8, 0.8] 米
    test_cases = [
        ("center", [0.0, 0.0]),          # 中心
        ("top_left", [-0.5, 0.5]),       # 左上
        ("top_right", [0.5, 0.5]),       # 右上
        ("bottom_left", [-0.5, -0.5]),   # 左下
        ("bottom_right", [0.5, -0.5]),   # 右下
        ("left", [-0.7, 0.0]),           # 左边
        ("right", [0.7, 0.0]),           # 右边
        ("top", [0.0, 0.7]),             # 上边
        ("bottom", [0.0, -0.7]),         # 下边
        ("corner_tl", [-0.8, 0.8]),      # 极限：左上角
        ("corner_br", [0.8, -0.8]),      # 极限：右下角
    ]
    
    print("=" * 60)
    print("红叉位置测试")
    print("=" * 60)
    print(f"图像分辨率: 480x480")
    print(f"action 范围: [-0.8, 0.8] 米")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    for name, action in test_cases:
        action = np.array(action, dtype=np.float32)
        
        # 让 PD 控制稳定：执行多步直到 stick 到达目标位置
        # 通常需要 50-100 步才能完全稳定
        for _ in range(100):
            env.step(action)
        
        # 设置 latest_action（这样 _get_obs 会绘制红叉）
        env.latest_action = action
        
        # 获取观测（会绘制红叉到 render_cache）
        obs = env._get_obs()
        
        # 获取带红叉的图像
        img = env.render_cache.copy()
        
        # 获取 stick 实际位置（验证是否到达目标）
        stick_pos = env._get_stick_pos()
        
        # 计算红叉应该出现的像素坐标（当前公式）
        coord_x = int((action[0] + 0.8) / 1.6 * env.render_size)
        coord_y = int((0.8 - action[1]) / 1.6 * env.render_size)
        
        # 计算 stick 到达目标的误差
        error = np.linalg.norm(stick_pos - action)
        
        # 在图像上添加文字说明
        text1 = f"Target: ({action[0]:+.2f}, {action[1]:+.2f}) -> Pixel: ({coord_x}, {coord_y})"
        text2 = f"Stick:  ({stick_pos[0]:+.2f}, {stick_pos[1]:+.2f}) | Error: {error:.4f}m"
        
        cv2.putText(img, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.putText(img, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 添加名称
        cv2.putText(img, name, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, name, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
        
        # 保存图像
        output_path = os.path.join(output_dir, f"{name}.png")
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        print(f"  {name:15s}: target=({action[0]:+.2f}, {action[1]:+.2f}) stick=({stick_pos[0]:+.2f}, {stick_pos[1]:+.2f}) error={error:.4f}m pixel=({coord_x:3d}, {coord_y:3d})")
    
    print("=" * 60)
    print("完成！请查看输出图片验证红叉位置。")
    print("")
    print("预期：")
    print("  - center: 红叉在图像中心 (240, 240)")
    print("  - top_left: 红叉在左上方")
    print("  - top_right: 红叉在右上方")
    print("  - bottom_left: 红叉在左下方")
    print("  - bottom_right: 红叉在右下方")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()

