from typing import Dict, Any
import numpy as np
import torch
import zarr
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.real_world.real_inference_util import get_real_obs_dict
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class DummyImageRunner(BaseImageRunner):
    """
    使用真实数据计算推理误差的图像 runner。
    用于无法进行真实环境 rollouts 的情况（如实机数据训练）。
    通过比较模型预测的 action 和真实 action 来计算误差。
    """

    def __init__(self,
                 output_dir: str,
                 zarr_path: str = None,
                 image_key: str = None,
                 n_obs_steps: int = None,
                 shape_meta: dict = None,
                 max_episodes: int = 10,
                 max_steps_per_episode: int = None,
                 **kwargs):
        """
        Args:
            output_dir: 输出目录
            zarr_path: 数据集路径（如果为 None，将从 shape_meta 或 policy 中获取）
            image_key: 图像键名（如果为 None，将自动检测）
            n_obs_steps: 观察步数（如果为 None，将从 policy 中获取）
            shape_meta: 形状元数据（如果为 None，将从 policy 中获取）
            max_episodes: 最大测试 episode 数
            max_steps_per_episode: 每个 episode 的最大步数（None 表示全部）
        """
        super().__init__(output_dir=output_dir)
        self.output_dir = output_dir
        self.zarr_path = zarr_path
        self.image_key = image_key
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode

    def run(self, policy: BaseImagePolicy) -> Dict[str, Any]:
        """
        使用真实数据计算推理误差。
        
        Args:
            policy: 策略模型
            
        Returns:
            包含误差统计信息的字典
        """
        # 从 policy 获取必要信息
        device = policy.device
        if self.n_obs_steps is None:
            # 尝试从 policy 获取
            if hasattr(policy, 'n_obs_steps'):
                self.n_obs_steps = policy.n_obs_steps
            else:
                raise ValueError("n_obs_steps 未指定且无法从 policy 获取")
        
        if self.shape_meta is None:
            # 尝试从 policy 获取
            if hasattr(policy, 'shape_meta'):
                self.shape_meta = policy.shape_meta
            else:
                raise ValueError("shape_meta 未指定且无法从 policy 获取")
        
        # 加载数据集
        if self.zarr_path is None:
            raise ValueError("zarr_path 未指定")
        
        # 自动检测图像键
        image_key = self.image_key
        if image_key is None:
            root = zarr.open(self.zarr_path, mode='r')
            data_keys = list(root['data'].keys())
            
            if 'img_128x128' in data_keys:
                image_key = 'img_128x128'
            elif 'img_240x240' in data_keys:
                image_key = 'img_240x240'
            elif 'img' in data_keys:
                image_key = 'img'
            else:
                raise ValueError(f"无法找到图像数据键。可用的键: {data_keys}")
        
        print(f"加载数据集: {self.zarr_path}")
        print(f"  使用图像键: '{image_key}'")
        replay_buffer = ReplayBuffer.copy_from_path(
            self.zarr_path,
            keys=[image_key, 'state', 'action']
        )
        print(f"  数据集包含 {replay_buffer.n_episodes} 个 episode")
        
        # 确定要测试的 episode 范围
        n_episodes = replay_buffer.n_episodes
        if self.max_episodes is not None:
            n_episodes = min(n_episodes, self.max_episodes)
        
        # 统计信息
        all_errors = []
        
        print(f"\n开始计算推理误差...")
        print(f"  测试 {n_episodes} 个 episode")
        print(f"  n_obs_steps: {self.n_obs_steps}")
        print(f"  推理步数: {getattr(policy, 'num_inference_steps', 'N/A')}")
        
        policy.eval()
        with torch.no_grad():
            for episode_idx in range(n_episodes):
                episode_data = replay_buffer.get_episode(episode_idx)
                episode_length = len(episode_data['action'])
                
                if self.max_steps_per_episode is not None:
                    episode_length = min(episode_length, self.max_steps_per_episode)
                
                # 初始化观察缓冲区
                obs_buffer = {
                    'image': [],
                    'agent_pos': []
                }
                
                # 填充初始观察缓冲区（使用前 n_obs_steps 帧）
                for step_idx in range(self.n_obs_steps):
                    if step_idx < len(episode_data[image_key]):
                        obs_buffer['image'].append(episode_data[image_key][step_idx])
                        obs_buffer['agent_pos'].append(episode_data['state'][step_idx, :2])
                    else:
                        # 如果episode太短，重复最后一帧
                        obs_buffer['image'].append(episode_data[image_key][-1])
                        obs_buffer['agent_pos'].append(episode_data['state'][-1, :2])
                
                # 遍历episode的每个step
                for step_idx in range(episode_length):
                    # 准备观察字典
                    obs_dict_np = {
                        'image': np.stack(obs_buffer['image']),  # (n_obs_steps, H, W, C) uint8
                        'agent_pos': np.stack(obs_buffer['agent_pos'])  # (n_obs_steps, 2) float
                    }
                    
                    # 使用 real_inference_util 转换观察格式
                    obs_dict_np = get_real_obs_dict(obs_dict_np, self.shape_meta)
                    
                    # 转换为 torch tensor 并移动到设备
                    obs_dict = dict_apply(
                        obs_dict_np,
                        lambda x: torch.from_numpy(x).unsqueeze(0).to(device)  # 添加 batch 维度
                    )
                    
                    # 推理
                    result = policy.predict_action(obs_dict)
                    action_pred = result['action'][0].detach().cpu().numpy()  # (horizon, 2)
                    
                    # 获取真实action（当前step的action）
                    action_true = episode_data['action'][step_idx]  # (2,)
                    
                    # 计算误差（使用预测的第一个action）
                    action_pred_first = action_pred[0]  # (2,)
                    error = np.linalg.norm(action_pred_first - action_true)
                    all_errors.append(error)
                    
                    # 更新观察缓冲区（移动到下一个step）
                    if step_idx + self.n_obs_steps < episode_length:
                        obs_buffer['image'].pop(0)
                        obs_buffer['image'].append(episode_data[image_key][step_idx + self.n_obs_steps])
                        obs_buffer['agent_pos'].pop(0)
                        obs_buffer['agent_pos'].append(episode_data['state'][step_idx + self.n_obs_steps, :2])
        
        # 计算统计信息
        if len(all_errors) > 0:
            all_errors = np.array(all_errors)
            mean_error = float(np.mean(all_errors))
            median_error = float(np.median(all_errors))
            std_error = float(np.std(all_errors))
            min_error = float(np.min(all_errors))
            max_error = float(np.max(all_errors))
            
            print(f"\n推理误差统计:")
            print(f"  平均误差: {mean_error:.4f} m")
            print(f"  中位数误差: {median_error:.4f} m")
            print(f"  标准差: {std_error:.4f} m")
            print(f"  最小误差: {min_error:.4f} m")
            print(f"  最大误差: {max_error:.4f} m")
            print(f"  总步数: {len(all_errors)}")
            
            # 返回统计信息（使用 test_mean_score 作为主要指标，但实际是平均误差）
            # 注意：这里返回负的平均误差，因为 checkpoint manager 通常期望更高的分数更好
            # 但误差越小越好，所以返回负值
            return {
                "test_mean_score": -mean_error,  # 负误差，越小越好
                "test_mean_error": mean_error,
                "test_median_error": median_error,
                "test_std_error": std_error,
                "test_min_error": min_error,
                "test_max_error": max_error,
                "test_n_steps": len(all_errors),
            }
        else:
            print("警告: 没有收集到任何数据")
            return {
                "test_mean_score": 0.0,
                "test_mean_error": 0.0,
                "test_median_error": 0.0,
            }
