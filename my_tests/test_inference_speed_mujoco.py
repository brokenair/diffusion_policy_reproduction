"""
测试 MuJoCo PushT 推理速度和模型性能的脚本。

功能：
1. 使用训练数据模拟推理
2. 比较真实action和模型推理action的差异
3. 统计推理一步所需的平均时间
"""

import sys
from pathlib import Path
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import dill
import hydra
from omegaconf import OmegaConf
import zarr

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.workspace.base_workspace import BaseWorkspace


# ============================================================================
# 配置参数
# ============================================================================

# ===== 你需要改的两个路径 =====
CKPT_PATH = "outputs/2025-12-04/17-59-18/checkpoints/epoch=2450-test_mean_score=1.000.ckpt"
CFG_PATH = "image_pusht_mujoco_diffusion_policy_cnn.yaml"
DEVICE = "cuda:0"

# 数据集路径（与训练时一致）
ZARR_PATH = "data/pusht_mujoco_128.zarr"

# 测试配置
MAX_EPISODES = 5  # None 表示测试所有episode
MAX_STEPS_PER_EPISODE = None  # None 表示测试所有step
WARMUP_STEPS = 5  # 预热步数（不统计时间）


def load_policy(checkpoint_path, config_path, device='cuda:0'):
    """
    加载 checkpoint 和 policy。
    注意：优先使用 checkpoint 中保存的配置，避免模型结构不匹配。
    """
    print(f"加载 checkpoint: {checkpoint_path}")
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']  # 使用 checkpoint 中保存的配置（包含正确的模型结构参数）
    
    # 如果提供了 config_path，只用于覆盖推理相关参数（如 num_inference_steps）
    # 不要覆盖影响模型结构的参数（如 crop_shape, shape_meta 等）
    if config_path is not None:
        config_cfg = OmegaConf.load(config_path)
        # 只更新 num_inference_steps，不影响模型结构
        if hasattr(config_cfg, 'policy') and hasattr(config_cfg.policy, 'num_inference_steps'):
            cfg.policy.num_inference_steps = config_cfg.policy.num_inference_steps
            print(f"  使用配置文件中的推理步数: {cfg.policy.num_inference_steps}")
    
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # 获取 policy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.eval().to(device)
    
    # 设置推理参数（对于 diffusion 模型）
    if 'diffusion' in cfg.name:
        # 优先使用配置文件中的推理步数
        if hasattr(cfg.policy, 'num_inference_steps') and cfg.policy.num_inference_steps is not None:
            policy.num_inference_steps = cfg.policy.num_inference_steps
        else:
            # 如果没有配置，使用默认值（根据scheduler类型）
            if hasattr(cfg.policy.noise_scheduler, '_target_'):
                if 'ddim' in cfg.policy.noise_scheduler._target_.lower():
                    policy.num_inference_steps = 16  # DDIM 默认值
                else:
                    policy.num_inference_steps = cfg.policy.noise_scheduler.num_train_timesteps  # DDPM 使用训练步数
            else:
                policy.num_inference_steps = 16  # 兜底默认值
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
        print(f"  推理步数: {policy.num_inference_steps}")
    
    # 获取 normalizer
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)
    
    # 重要：设置 normalizer 后需要再次移动到设备，确保 normalizer 的参数也在 GPU 上
    policy.to(device)
    
    return policy, cfg, device


def load_dataset(zarr_path, image_key=None):
    """
    加载训练数据集。
    如果 image_key 为 None，则自动检测（MuJoCo 使用 'img'）。
    """
    print(f"加载数据集: {zarr_path}")
    
    # 如果未指定 image_key，自动检测
    if image_key is None:
        root = zarr.open(zarr_path, mode='r')
        data_keys = list(root['data'].keys())
        
        if 'img' in data_keys:
            # MuJoCo 版本
            image_key = 'img'
        elif 'img_128x128' in data_keys:
            # 实机版本，使用 128x128 分辨率
            image_key = 'img_128x128'
        elif 'img_240x240' in data_keys:
            # 实机版本，使用 240x240 分辨率
            image_key = 'img_240x240'
        else:
            raise ValueError(f"无法找到图像数据键。可用的键: {data_keys}")
        
        print(f"  自动检测到图像键: '{image_key}'")
        print(f"  数据集中可用的键: {data_keys}")
    
    replay_buffer = ReplayBuffer.copy_from_path(
        zarr_path, 
        keys=[image_key, 'state', 'action']
    )
    print(f"  数据集包含 {replay_buffer.n_episodes} 个 episode")
    print(f"  总步数: {replay_buffer.n_steps}")
    return replay_buffer, image_key


def prepare_obs_dict(obs_buffer, shape_meta, device):
    """
    准备观察字典，用于推理。
    MuJoCo 版本：数据已经是正确的格式，只需要转换为 torch tensor。
    """
    # MuJoCo 数据格式：图像已经是 (T, H, W, C) uint8，agent_pos 是 (T, 2) float
    # 需要转换为 (T, C, H, W) float32 并归一化到 [0, 1]
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            # 图像：从 (T, H, W, C) uint8 转换为 (T, C, H, W) float32 [0, 1]
            imgs = obs_buffer[key]  # (T, H, W, C) uint8
            # THWC to TCHW
            imgs = np.moveaxis(imgs, -1, 1)  # (T, C, H, W)
            # uint8 to float32 [0, 1]
            obs_dict_np[key] = imgs.astype(np.float32) / 255.0
        elif type == 'low_dim':
            # 低维数据：直接使用
            obs_dict_np[key] = obs_buffer[key].astype(np.float32)
    
    # 转换为 torch tensor 并移动到设备
    obs_dict = dict_apply(
        obs_dict_np,
        lambda x: torch.from_numpy(x).unsqueeze(0).to(device)  # 添加 batch 维度并移动到设备
    )
    
    return obs_dict


def test_inference_speed(policy, cfg, device, replay_buffer, image_key, 
                         max_episodes=None, max_steps_per_episode=None,
                         warmup_steps=5):
    """
    测试推理速度和模型性能。
    """
    n_obs_steps = cfg.n_obs_steps
    shape_meta = cfg.task.shape_meta
    
    # 统计信息
    inference_times = []
    action_errors = defaultdict(list)  # 按episode存储误差
    all_true_actions = []
    all_pred_actions = []
    
    # 确定要测试的episode范围
    n_episodes = replay_buffer.n_episodes
    if max_episodes is not None:
        n_episodes = min(n_episodes, max_episodes)
    
    print(f"\n开始测试推理速度和模型性能...")
    print(f"  测试 {n_episodes} 个 episode")
    print(f"  n_obs_steps: {n_obs_steps}")
    print(f"  预热步数: {warmup_steps}")
    
    total_steps = 0
    total_inference_time = 0.0
    
    for episode_idx in range(n_episodes):
        episode_data = replay_buffer.get_episode(episode_idx)
        episode_length = len(episode_data['action'])
        
        if max_steps_per_episode is not None:
            episode_length = min(episode_length, max_steps_per_episode)
        
        # 初始化观察缓冲区
        obs_buffer = {
            'image': [],
            'agent_pos': []
        }
        
        # 填充初始观察缓冲区（使用前 n_obs_steps 帧）
        for step_idx in range(n_obs_steps):
            if step_idx < len(episode_data[image_key]):
                obs_buffer['image'].append(episode_data[image_key][step_idx])
                obs_buffer['agent_pos'].append(episode_data['state'][step_idx, :2])
            else:
                # 如果episode太短，重复最后一帧
                obs_buffer['image'].append(episode_data[image_key][-1])
                obs_buffer['agent_pos'].append(episode_data['state'][-1, :2])
        
        episode_errors = []
        
        # 遍历episode的每个step
        for step_idx in range(episode_length):
            # 准备观察
            obs_dict = prepare_obs_dict(obs_buffer, shape_meta, device)
            
            # 推理（统计时间）
            inference_start = time.time()
            with torch.no_grad():
                result = policy.predict_action(obs_dict)
                action_pred = result['action'][0].detach().cpu().numpy()  # (horizon, 2)
            inference_time = time.time() - inference_start
            
            # 获取真实action（当前step的action）
            action_true = episode_data['action'][step_idx]  # (2,)
            
            # 只统计预热后的时间
            if step_idx >= warmup_steps:
                inference_times.append(inference_time)
                total_inference_time += inference_time
                total_steps += 1
                
                # 计算误差（使用预测的第一个action）
                action_pred_first = action_pred[0]  # (2,)
                error = np.linalg.norm(action_pred_first - action_true)
                episode_errors.append(error)
                action_errors[episode_idx].append(error)
                
                all_true_actions.append(action_true)
                all_pred_actions.append(action_pred_first)
            
            # 更新观察缓冲区（移动到下一个step）
            if step_idx + n_obs_steps < episode_length:
                obs_buffer['image'].pop(0)
                obs_buffer['image'].append(episode_data[image_key][step_idx + n_obs_steps])
                obs_buffer['agent_pos'].pop(0)
                obs_buffer['agent_pos'].append(episode_data['state'][step_idx + n_obs_steps, :2])
            else:
                # 如果已经到达episode末尾，保持最后一帧
                pass
        
        if len(episode_errors) > 0:
            mean_error = np.mean(episode_errors)
            std_error = np.std(episode_errors)
            median_error = np.median(episode_errors)
            min_error = np.min(episode_errors)
            max_error = np.max(episode_errors)
            
            # 计算action的典型范围（用于评估误差的相对大小）
            episode_actions = episode_data['action'][warmup_steps:]
            if len(episode_actions) > 0:
                action_ranges = np.max(episode_actions, axis=0) - np.min(episode_actions, axis=0)
                action_typical_range = np.mean(action_ranges)  # 平均每个维度的范围
            else:
                action_typical_range = None
            
            print(f"  Episode {episode_idx}: {episode_length} steps")
            print(f"    平均误差: {mean_error:.4f} m (标准差: {std_error:.4f} m)")
            print(f"    中位数误差: {median_error:.4f} m")
            print(f"    误差范围: [{min_error:.4f}, {max_error:.4f}] m")
            if action_typical_range is not None:
                relative_error = mean_error / action_typical_range if action_typical_range > 0 else 0
                print(f"    Action典型范围: {action_typical_range:.4f} m")
                print(f"    相对误差: {relative_error*100:.1f}% (平均误差/典型范围)")
    
    # 统计结果
    if len(inference_times) > 0:
        mean_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        min_inference_time = np.min(inference_times)
        max_inference_time = np.max(inference_times)
        median_inference_time = np.median(inference_times)
        
        all_errors = np.concatenate([errors for errors in action_errors.values()])
        mean_error = np.mean(all_errors)
        std_error = np.std(all_errors)
        median_error = np.median(all_errors)
        
        print("\n" + "=" * 60)
        print("推理速度统计")
        print("=" * 60)
        print(f"总步数: {total_steps}")
        print(f"总推理时间: {total_inference_time:.4f} s")
        print(f"平均推理时间: {mean_inference_time*1000:.2f} ms")
        print(f"标准差: {std_inference_time*1000:.2f} ms")
        print(f"最小推理时间: {min_inference_time*1000:.2f} ms")
        print(f"最大推理时间: {max_inference_time*1000:.2f} ms")
        print(f"中位数推理时间: {median_inference_time*1000:.2f} ms")
        print(f"理论最大频率: {1.0/mean_inference_time:.1f} Hz")
        
        print("\n" + "=" * 60)
        print("模型性能统计")
        print("=" * 60)
        print(f"平均动作误差: {mean_error:.4f} m")
        print(f"标准差: {std_error:.4f} m")
        print(f"中位数误差: {median_error:.4f} m")
        print(f"最小误差: {np.min(all_errors):.4f} m")
        print(f"最大误差: {np.max(all_errors):.4f} m")
        
        # 计算action的典型范围（用于评估误差的相对大小）
        all_true_actions_array = np.array(all_true_actions)
        if len(all_true_actions_array) > 0:
            action_ranges = np.max(all_true_actions_array, axis=0) - np.min(all_true_actions_array, axis=0)
            action_typical_range = np.mean(action_ranges)  # 平均每个维度的范围
            action_std = np.std(all_true_actions_array, axis=0)
            action_typical_std = np.mean(action_std)  # 平均每个维度的标准差
        else:
            action_typical_range = None
            action_typical_std = None
        
        if action_typical_range is not None:
            relative_error = mean_error / action_typical_range if action_typical_range > 0 else 0
            print(f"\nAction统计信息:")
            print(f"  Action典型范围: {action_typical_range:.4f} m (每个维度的平均范围)")
            print(f"  Action典型标准差: {action_typical_std:.4f} m (每个维度的平均标准差)")
            print(f"  相对误差: {relative_error*100:.1f}% (平均误差/典型范围)")
            if action_typical_std > 0:
                print(f"  误差/标准差比: {mean_error/action_typical_std:.2f}")
        
        # 计算分位数
        percentiles = [50, 75, 90, 95, 99]
        print("\n误差分位数:")
        for p in percentiles:
            value = np.percentile(all_errors, p)
            print(f"  {p}%: {value:.4f} m")
        
        return {
            'inference_times': inference_times,
            'action_errors': action_errors,
            'all_true_actions': all_true_actions_array,
            'all_pred_actions': np.array(all_pred_actions),
            'stats': {
                'mean_inference_time': mean_inference_time,
                'std_inference_time': std_inference_time,
                'mean_error': mean_error,
                'std_error': std_error,
            }
        }
    else:
        print("警告: 没有收集到任何数据")
        return None


def plot_results(results, save_dir="plots"):
    """
    绘制结果图表。
    """
    if results is None:
        return
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 推理时间分布
    plt.figure(figsize=(10, 6))
    plt.hist(results['inference_times'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('推理时间 (秒)')
    plt.ylabel('频数')
    plt.title('推理时间分布')
    plt.axvline(np.mean(results['inference_times']), color='r', 
                linestyle='--', label=f'平均值: {np.mean(results["inference_times"])*1000:.2f} ms')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / 'inference_time_distribution.png', dpi=150)
    plt.close()
    
    # 2. 动作误差分布
    all_errors = np.concatenate([errors for errors in results['action_errors'].values()])
    plt.figure(figsize=(10, 6))
    plt.hist(all_errors, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('动作误差 (米)')
    plt.ylabel('频数')
    plt.title('动作误差分布')
    plt.axvline(np.mean(all_errors), color='r', 
                linestyle='--', label=f'平均值: {np.mean(all_errors):.4f} m')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / 'action_error_distribution.png', dpi=150)
    plt.close()
    
    # 3. 每个episode的平均误差
    episode_means = [np.mean(errors) for errors in results['action_errors'].values()]
    plt.figure(figsize=(10, 6))
    plt.plot(episode_means, 'o-', markersize=4)
    plt.xlabel('Episode 索引')
    plt.ylabel('平均动作误差 (米)')
    plt.title('每个 Episode 的平均动作误差')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / 'episode_mean_error.png', dpi=150)
    plt.close()
    
    # 4. 真实vs预测动作散点图（X和Y分别）
    true_actions = results['all_true_actions']
    pred_actions = results['all_pred_actions']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # X坐标
    axes[0].scatter(true_actions[:, 0], pred_actions[:, 0], alpha=0.5, s=1)
    axes[0].plot([true_actions[:, 0].min(), true_actions[:, 0].max()],
                [true_actions[:, 0].min(), true_actions[:, 0].max()], 
                'r--', label='理想线')
    axes[0].set_xlabel('真实 X (米)')
    axes[0].set_ylabel('预测 X (米)')
    axes[0].set_title('X 坐标: 真实 vs 预测')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Y坐标
    axes[1].scatter(true_actions[:, 1], pred_actions[:, 1], alpha=0.5, s=1)
    axes[1].plot([true_actions[:, 1].min(), true_actions[:, 1].max()],
                [true_actions[:, 1].min(), true_actions[:, 1].max()], 
                'r--', label='理想线')
    axes[1].set_xlabel('真实 Y (米)')
    axes[1].set_ylabel('预测 Y (米)')
    axes[1].set_title('Y 坐标: 真实 vs 预测')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'action_scatter.png', dpi=150)
    plt.close()
    
    print(f"\n图表已保存到: {save_dir}")


def main():
    """
    主函数。
    """
    print("=" * 60)
    print("MuJoCo PushT 推理速度和模型性能测试")
    print("=" * 60)
    
    # 加载模型
    print("\n加载模型...")
    policy, cfg, device = load_policy(CKPT_PATH, CFG_PATH, DEVICE)
    print(f"  Device: {device}")
    print(f"  n_obs_steps: {cfg.n_obs_steps}")
    print(f"  horizon: {policy.horizon}")
    print(f"  action_dim: {policy.action_dim}")
    
    # 加载数据集（自动检测图像键）
    print("\n加载数据集...")
    replay_buffer, image_key = load_dataset(ZARR_PATH, image_key=None)  # 自动检测
    
    # 测试推理速度和性能
    results = test_inference_speed(
        policy, cfg, device, replay_buffer, image_key,
        max_episodes=MAX_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        warmup_steps=WARMUP_STEPS
    )
    
    # 绘制结果
    if results is not None:
        print("\n生成图表...")
        plot_results(results, save_dir="plots/inference_speed_mujoco_test")
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()

