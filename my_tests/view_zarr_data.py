"""
查看 zarr 数据集中的图像数据。

使用方法：
    python my_tests/view_zarr_data.py -i data/pusht_mujoco_demo.zarr
    
操作：
    空格键 - 暂停/继续
    左方向键 - 上一帧
    右方向键 - 下一帧
    上方向键 - 上一个 episode
    下方向键 - 下一个 episode
    R - 重新开始当前 episode
    Q/ESC - 退出
    数字键 0-9 - 跳转到指定 episode
"""

import numpy as np
import cv2
import zarr
import click
import os


def load_zarr_data(zarr_path):
    """加载 zarr 数据"""
    root = zarr.open(zarr_path, mode='r')
    
    # 获取所有 episode 的数据
    episodes = []
    meta = root['meta']
    
    for i in range(meta['episode_ends'].shape[0]):
        start = 0 if i == 0 else meta['episode_ends'][i-1]
        end = meta['episode_ends'][i]
        
        episode_data = {
            'img': root['data']['img'][start:end],
            'state': root['data']['state'][start:end],
            'action': root['data']['action'][start:end],
        }
        
        # 尝试获取 n_contacts（如果有）
        if 'n_contacts' in root['data']:
            episode_data['n_contacts'] = root['data']['n_contacts'][start:end]
        
        episodes.append(episode_data)
    
    return episodes


def draw_info(img, episode_idx, frame_idx, total_frames, state, action, n_contacts=None, is_paused=False):
    """在图像上绘制信息"""
    img = img.copy()
    h, w = img.shape[:2]
    
    # # 背景半透明黑色矩形
    # overlay = img.copy()
    # cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    # cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # # 文字信息
    # font = cv2.FONT_HERSHEY_SIMPLEX
    
    # # Episode 和帧信息
    # text1 = f"Episode {episode_idx} | Frame {frame_idx+1}/{total_frames}"
    # if is_paused:
    #     text1 += " [PAUSED]"
    # cv2.putText(img, text1, (10, 25), font, 0.7, (255, 255, 255), 2)
    
    # # State 信息
    # if state is not None and len(state) >= 5:
    #     text2 = f"Stick: ({state[0]:.2f}, {state[1]:.2f})  T-block: ({state[2]:.2f}, {state[3]:.2f}, {state[4]:.1f}deg)"
    #     cv2.putText(img, text2, (10, 55), font, 0.5, (150, 255, 150), 1)
    
    # # Action 信息
    # if action is not None:
    #     text3 = f"Action: ({action[0]:.2f}, {action[1]:.2f})"
    #     cv2.putText(img, text3, (10, 80), font, 0.5, (150, 150, 255), 1)
    
    # # Contacts 信息
    # if n_contacts is not None:
    #     text4 = f"Contacts: {n_contacts[0]:.0f}"
    #     cv2.putText(img, text4, (10, 105), font, 0.5, (255, 150, 150), 1)
    
    return img


@click.command()
@click.option('-i', '--input', 'zarr_path', required=True, help='zarr 数据路径')
@click.option('--fps', default=10, type=int, help='播放帧率（默认 10）')
@click.option('--start-episode', default=0, type=int, help='起始 episode（默认 0）')
def main(zarr_path, fps, start_episode):
    """
    播放 zarr 数据集中的图像。
    """
    
    if not os.path.exists(zarr_path):
        print(f"错误：找不到文件 {zarr_path}")
        return
    
    print("=" * 60)
    print("加载数据...")
    episodes = load_zarr_data(zarr_path)
    
    print(f"数据集: {zarr_path}")
    print(f"总 episodes: {len(episodes)}")
    for i, ep in enumerate(episodes):
        print(f"  Episode {i}: {len(ep['img'])} 帧")
    print("=" * 60)
    print("操作说明：")
    print("  空格键 - 暂停/继续")
    print("  左/右方向键 - 上一帧/下一帧")
    print("  上/下方向键 - 上一个/下一个 episode")
    print("  R - 重新开始当前 episode")
    print("  Q/ESC - 退出")
    print("=" * 60)
    
    # 初始化
    episode_idx = max(0, min(start_episode, len(episodes) - 1))
    frame_idx = 0
    is_paused = False
    delay = int(1000 / fps)  # ms
    
    cv2.namedWindow('Zarr Data Viewer', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Zarr Data Viewer', 800, 800)
    
    while True:
        # 获取当前 episode 和帧
        episode = episodes[episode_idx]
        total_frames = len(episode['img'])
        
        # 确保 frame_idx 在范围内
        frame_idx = max(0, min(frame_idx, total_frames - 1))
        
        # 获取图像和数据
        img = episode['img'][frame_idx]
        state = episode['state'][frame_idx] if 'state' in episode else None
        action = episode['action'][frame_idx] if 'action' in episode else None
        n_contacts = episode['n_contacts'][frame_idx] if 'n_contacts' in episode else None
        
        # 转换图像格式（如果是 float32 [0,1] 则转为 uint8 [0,255]）
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        
        # 绘制信息
        img_display = draw_info(img, episode_idx, frame_idx, total_frames, state, action, n_contacts, is_paused)
        
        # 显示
        cv2.imshow('Zarr Data Viewer', cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
        
        # 等待按键
        wait_time = delay if not is_paused else 0
        key = cv2.waitKey(wait_time) & 0xFF
        
        # 处理按键
        if key == ord('q') or key == 27:  # Q 或 ESC
            break
        elif key == ord(' '):  # 空格：暂停/继续
            is_paused = not is_paused
        elif key == 81 or key == 2:  # 左方向键：上一帧
            frame_idx = max(0, frame_idx - 1)
            is_paused = True
        elif key == 83 or key == 3:  # 右方向键：下一帧
            frame_idx = min(total_frames - 1, frame_idx + 1)
            is_paused = True
        elif key == 82 or key == 0:  # 上方向键：上一个 episode
            episode_idx = (episode_idx - 1) % len(episodes)
            frame_idx = 0
            is_paused = True
        elif key == 84 or key == 1:  # 下方向键：下一个 episode
            episode_idx = (episode_idx + 1) % len(episodes)
            frame_idx = 0
            is_paused = True
        elif key == ord('r'):  # R：重新开始
            frame_idx = 0
            is_paused = True
        elif ord('0') <= key <= ord('9'):  # 数字键：跳转到指定 episode
            target_ep = key - ord('0')
            if target_ep < len(episodes):
                episode_idx = target_ep
                frame_idx = 0
                is_paused = True
        
        # 自动播放时前进一帧
        if not is_paused:
            frame_idx += 1
            if frame_idx >= total_frames:
                # 当前 episode 播放完，跳到下一个
                episode_idx = (episode_idx + 1) % len(episodes)
                frame_idx = 0
    
    cv2.destroyAllWindows()
    print("\n退出")


if __name__ == "__main__":
    main()

