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
import sys
import shutil
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from diffusion_policy.common.replay_buffer import ReplayBuffer


def load_zarr_data(zarr_path):
    """
    加载 zarr 数据。

    重要：将所有数据复制到内存，避免 zarr 延迟加载引用的问题。
    如果后续需要删除原始文件，必须确保数据已经在内存中。
    """
    root = zarr.open(zarr_path, mode='r')

    # 检测数据格式：MuJoCo 版本使用 'img'，实机版本使用 'img_128x128' 等
    data_keys = list(root['data'].keys())
    if 'img' in data_keys:
        # MuJoCo 版本
        img_key = 'img'
    elif 'img_128x128' in data_keys:
        # 实机版本，使用 128x128 分辨率
        img_key = 'img_128x128'
        #img_key = 'img_240x240'
    elif 'img_15000px' in data_keys:
        # 实机版本，使用 15000 分辨率
        img_key = 'img_15000px'
        #img_key = 'img_240x240'

    else:
        raise ValueError(f"无法找到图像数据键。可用的键: {data_keys}")

    print(f"检测到数据格式，使用图像键: '{img_key}'")
    print(f"数据集中可用的键: {data_keys}")

    # 获取所有 episode 的数据
    episodes = []
    meta = root['meta']

    for i in range(meta['episode_ends'].shape[0]):
        start = 0 if i == 0 else meta['episode_ends'][i-1]
        end = meta['episode_ends'][i]

        # 重要：使用 np.array() 将 zarr 切片复制到内存
        # 这样在删除原始文件后，数据仍然可用
        episode_data = {
            'img': np.array(root['data'][img_key][start:end]),
            'state': np.array(root['data']['state'][start:end]),
            'action': np.array(root['data']['action'][start:end]),
        }

        # 尝试获取 n_contacts（如果有）
        if 'n_contacts' in root['data']:
            episode_data['n_contacts'] = np.array(root['data']['n_contacts'][start:end])

        episodes.append(episode_data)

    return episodes


def save_episodes_to_zarr(episodes, zarr_path, original_count):
    """
    将删除后的 episodes 保存回 zarr 文件

    Args:
        episodes: 保留的 episodes 列表（数据已在内存中）
        zarr_path: 原始 zarr 文件路径
        original_count: 原始 episode 数量
    """
    if len(episodes) == 0:
        print("错误：没有可保存的 episodes")
        return

    print(f"\n开始保存数据...")
    print(f"原始 episodes: {original_count}, 保留 episodes: {len(episodes)}")

    # 验证所有 episode 数据已在内存中（非空）
    for i, ep in enumerate(episodes):
        if len(ep['img']) == 0:
            print(f"警告：Episode {i} 为空，跳过保存")
            return

    # 创建备份
    backup_path = zarr_path.rstrip('/') + '_backup'
    if os.path.exists(zarr_path):
        if os.path.exists(backup_path):
            print(f"删除旧备份: {backup_path}")
            shutil.rmtree(backup_path)
        print(f"创建备份: {backup_path}")
        shutil.copytree(zarr_path, backup_path)
        print(f"✓ 备份创建成功")

    # 准备数据
    all_data = {}
    episode_ends = []
    current_end = 0

    # 获取所有数据键
    data_keys = list(episodes[0].keys())
    print(f"数据键: {data_keys}")

    for i, episode in enumerate(episodes):
        episode_length = len(episode['img'])

        # 验证数据已在内存中
        if not isinstance(episode['img'], np.ndarray):
            raise RuntimeError(f"Episode {i} 的数据不是 numpy 数组，可能是 zarr 引用！")

        # 收集当前 episode 的所有数据
        for key in data_keys:
            if key not in all_data:
                all_data[key] = []
            all_data[key].append(episode[key])

        current_end += episode_length
        episode_ends.append(current_end)

    # 合并所有数据
    print(f"合并 {len(episodes)} 个 episodes 的数据...")
    merged_data = {}
    for key in data_keys:
        merged_data[key] = np.concatenate(all_data[key], axis=0)
        print(f"  {key}: shape={merged_data[key].shape}, dtype={merged_data[key].dtype}")

    # 创建新的 ReplayBuffer 并保存
    try:
        # 删除原文件
        if os.path.exists(zarr_path):
            print(f"删除原文件: {zarr_path}")
            shutil.rmtree(zarr_path)

        # 创建新的 ReplayBuffer（使用内存模式先构建数据）
        print(f"创建新的 ReplayBuffer...")
        replay_buffer = ReplayBuffer.create_empty_numpy()

        # 添加所有数据
        for key, value in merged_data.items():
            replay_buffer.data[key] = value

        # 设置 episode_ends
        replay_buffer.meta['episode_ends'] = np.array(episode_ends, dtype=np.int64)

        # 保存到文件
        print(f"保存到文件: {zarr_path}")
        print(f"（这可能需要一些时间，取决于数据量...）")
        replay_buffer.save_to_path(zarr_path, compressors='disk', chunk_length=-1)

        print(f"\n✓ 保存成功！")
        print(f"  文件: {zarr_path}")
        print(f"  总帧数: {current_end}")
        print(f"  Episodes: {len(episodes)}")
        print(f"  备份: {backup_path}")
    except Exception as e:
        print(f"\n✗ 保存失败: {e}")
        print(f"可以从备份恢复:")
        print(f"  rm -rf {zarr_path}")
        print(f"  mv {backup_path} {zarr_path}")
        raise


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


def draw_2d_position_plot(episode, frame_idx, episode_idx=None, total_episodes=None, window_size=600):
    """
    绘制二维位置图，显示 stick 位置和 action 目标位置
    
    Args:
        episode: 当前 episode 的数据字典
        frame_idx: 当前帧索引
        episode_idx: 当前 episode 索引
        total_episodes: 总 episode 数量
        window_size: 窗口大小（像素）
    
    Returns:
        绘制的图像（RGB格式）
    """
    # 创建空白图像
    img = np.ones((window_size, window_size, 3), dtype=np.uint8) * 240  # 浅灰色背景
    
    # 获取当前帧的数据
    state = episode['state'][frame_idx] if 'state' in episode else None
    action = episode['action'][frame_idx] if 'action' in episode else None
    
    # 获取整个 episode 的轨迹用于显示历史
    states = episode['state'] if 'state' in episode else None
    actions = episode['action'] if 'action' in episode else None
    
    # 计算坐标范围（自动适应数据范围）
    if states is not None and len(states) > 0:
        # 从 state 和 action 中提取所有 xy 坐标
        all_x = []
        all_y = []
        
        # 添加所有 stick 位置
        for s in states:
            if s is not None and len(s) >= 2:
                all_x.append(s[0])
                all_y.append(s[1])
        
        # 添加所有 action 位置
        if actions is not None:
            for a in actions:
                if a is not None and len(a) >= 2:
                    all_x.append(a[0])
                    all_y.append(a[1])
        
        if len(all_x) > 0:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            
            # 添加边距
            x_range = x_max - x_min
            y_range = y_max - y_min
            margin = max(x_range, y_range) * 0.1
            x_min -= margin
            x_max += margin
            y_min -= margin
            y_max += margin
        else:
            # 默认范围
            x_min, x_max = -0.5, 0.0
            y_min, y_max = -0.3, 0.3
    else:
        # 默认范围
        x_min, x_max = -0.5, 0.0
        y_min, y_max = -0.3, 0.3
    
    # 坐标转换函数：从实际坐标到像素坐标
    def world_to_pixel(x, y):
        px = int((x - x_min) / (x_max - x_min) * window_size)
        py = int((y_max - y) / (y_max - y_min) * window_size)  # y轴翻转
        return px, py
    
    # 绘制网格
    grid_color = (200, 200, 200)
    for i in range(5):
        # 垂直线
        x = int(i * window_size / 4)
        cv2.line(img, (x, 0), (x, window_size), grid_color, 1)
        # 水平线
        y = int(i * window_size / 4)
        cv2.line(img, (0, y), (window_size, y), grid_color, 1)
    
    # 绘制坐标轴（固定在窗口中心）
    center_x = window_size // 2
    center_y = window_size // 2
    cv2.line(img, (center_x, 0), (center_x, window_size), (100, 100, 100), 2)
    cv2.line(img, (0, center_y), (window_size, center_y), (100, 100, 100), 2)
    
    # 绘制历史轨迹（stick 位置）
    if states is not None and frame_idx > 0:
        stick_trajectory = []
        for i in range(min(frame_idx + 1, len(states))):
            s = states[i]
            if s is not None and len(s) >= 2:
                px, py = world_to_pixel(s[0], s[1])
                stick_trajectory.append((px, py))
        
        if len(stick_trajectory) > 1:
            # 绘制轨迹线（绿色，半透明）
            for i in range(len(stick_trajectory) - 1):
                cv2.line(img, stick_trajectory[i], stick_trajectory[i+1], (0, 200, 0), 2)
    
    # 绘制历史轨迹（action 位置）
    if actions is not None and frame_idx > 0:
        action_trajectory = []
        for i in range(min(frame_idx + 1, len(actions))):
            a = actions[i]
            if a is not None and len(a) >= 2:
                px, py = world_to_pixel(a[0], a[1])
                action_trajectory.append((px, py))
        
        if len(action_trajectory) > 1:
            # 绘制轨迹线（红色，半透明）
            for i in range(len(action_trajectory) - 1):
                cv2.line(img, action_trajectory[i], action_trajectory[i+1], (200, 0, 0), 2)
    
    # 绘制当前 stick 位置（绿色大圆点）
    if state is not None and len(state) >= 2:
        stick_x, stick_y = world_to_pixel(state[0], state[1])
        cv2.circle(img, (stick_x, stick_y), 8, (0, 255, 0), -1)  # 实心圆
        cv2.circle(img, (stick_x, stick_y), 10, (0, 200, 0), 2)   # 外圈
    
    # 绘制当前 action 目标位置（红色大圆点）
    if action is not None and len(action) >= 2:
        action_x, action_y = world_to_pixel(action[0], action[1])
        cv2.circle(img, (action_x, action_y), 8, (0, 0, 255), -1)  # 实心圆
        cv2.circle(img, (action_x, action_y), 10, (200, 0, 0), 2)   # 外圈
    
    # 绘制从 stick 到 action 的连线
    if state is not None and len(state) >= 2 and action is not None and len(action) >= 2:
        stick_x, stick_y = world_to_pixel(state[0], state[1])
        action_x, action_y = world_to_pixel(action[0], action[1])
        cv2.line(img, (stick_x, stick_y), (action_x, action_y), (255, 165, 0), 2)  # 橙色连线
    
    # 添加文字标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    # 标题和 Episode 信息
    title = "2D Position View"
    if episode_idx is not None and total_episodes is not None:
        title += f" | Episode {episode_idx}/{total_episodes-1}"
    cv2.putText(img, title, (10, 25), font, 0.7, (0, 0, 0), 2)
    
    # Stick 位置标签
    if state is not None and len(state) >= 2:
        stick_text = f"Stick: ({state[0]:.3f}, {state[1]:.3f})"
        cv2.putText(img, stick_text, (10, window_size - 50), font, font_scale, (0, 200, 0), thickness)
    
    # Action 位置标签
    if action is not None and len(action) >= 2:
        action_text = f"Action: ({action[0]:.3f}, {action[1]:.3f})"
        cv2.putText(img, action_text, (10, window_size - 30), font, font_scale, (0, 0, 200), thickness)
    
    # 图例
    cv2.circle(img, (window_size - 100, 30), 6, (0, 255, 0), -1)
    cv2.putText(img, "Stick", (window_size - 85, 35), font, font_scale, (0, 0, 0), thickness)
    
    cv2.circle(img, (window_size - 100, 50), 6, (0, 0, 255), -1)
    cv2.putText(img, "Action", (window_size - 85, 55), font, font_scale, (0, 0, 0), thickness)
    
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
    print("  D - 删除当前 episode（退出时保存）")
    print("  S - 立即保存删除后的数据")
    print("  Q/ESC - 退出（如果有删除，会提示保存）")
    print("=" * 60)
    
    # 初始化
    episode_idx = max(0, min(start_episode, len(episodes) - 1))
    frame_idx = 0
    is_paused = False
    delay = int(1000 / fps)  # ms
    has_deletions = False  # 标记是否有删除操作
    original_episode_count = len(episodes)  # 记录原始 episode 数量
    
    cv2.namedWindow('Zarr Data Viewer', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Zarr Data Viewer', 800, 800)
    
    cv2.namedWindow('2D Position View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('2D Position View', 600, 600)
    
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
        
        # 绘制二维位置图
        pos_plot = draw_2d_position_plot(episode, frame_idx, episode_idx, len(episodes))
        
        # 显示
        cv2.imshow('Zarr Data Viewer', cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
        cv2.imshow('2D Position View', cv2.cvtColor(pos_plot, cv2.COLOR_RGB2BGR))
        
        # 等待按键
        wait_time = delay if not is_paused else 0
        key = cv2.waitKey(wait_time) & 0xFF
        
        # 处理按键
        if key == ord('q') or key == 27:  # Q 或 ESC
            if has_deletions:
                print("\n检测到有删除操作，是否保存？")
                print("  按 Y 保存并退出")
                print("  按 N 不保存直接退出")
                print("  按其他键取消退出")
                save_key = cv2.waitKey(0) & 0xFF
                if save_key == ord('y') or save_key == ord('Y'):
                    save_episodes_to_zarr(episodes, zarr_path, original_episode_count)
                    print("数据已保存！")
                    break
                elif save_key == ord('n') or save_key == ord('N'):
                    print("未保存，直接退出")
                    break
                # 其他键取消退出，继续循环
            else:
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
        elif key == ord('d') or key == ord('D'):  # D：删除当前 episode
            if len(episodes) > 1:
                deleted_ep = episode_idx
                episodes.pop(episode_idx)
                has_deletions = True
                print(f"已删除 Episode {deleted_ep}")
                # 调整 episode_idx，确保不越界
                if episode_idx >= len(episodes):
                    episode_idx = len(episodes) - 1
                frame_idx = 0
                is_paused = True
                print(f"当前剩余 {len(episodes)} 个 episodes，当前显示 Episode {episode_idx}")
                print("提示：按 S 键立即保存，或退出时保存")
            else:
                print("警告：至少需要保留 1 个 episode，无法删除")
        elif key == ord('s') or key == ord('S'):  # S：保存删除后的数据
            if has_deletions:
                save_episodes_to_zarr(episodes, zarr_path, original_episode_count)
                has_deletions = False
                print("数据已保存！")
            else:
                print("没有删除操作，无需保存")
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

