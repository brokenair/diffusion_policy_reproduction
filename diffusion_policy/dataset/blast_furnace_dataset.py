from typing import Dict, List
import copy
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset


class BlastFurnaceLowdimDataset(BaseLowdimDataset):
    """
    高炉数据用的低维轨迹数据集。

    要求 zarr 结构为 ReplayBuffer 兼容格式：
    root/
      data/
        obs      (sum_T, D_obs)
        action   (sum_T, D_act)
      meta/
        episode_ends  (N_episodes,)
    其中 obs/action 的键名分别为 obs_key / action_key（默认 'obs' 和 'action'）。
    """

    def __init__(self,
                 zarr_path: str,
                 horizon: int = 16,
                 pad_before: int = 0,
                 pad_after: int = 0,
                 obs_key: str = 'obs',
                 action_key: str = 'action',
                 seed: int = 42,
                 val_episodes: List[int] = None,
                 max_train_episodes=None):
        super().__init__()

        # 从 zarr 读取 ReplayBuffer
        # 你前面生成的 zarr 里只需要有 data/obs, data/action, meta/episode_ends 即可
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=[obs_key, action_key]
        )

        # 训练 / 验证 episode 划分
        val_mask = np.zeros(self.replay_buffer.n_episodes, dtype=bool)
        if val_episodes is not None and len(val_episodes) > 0:
            for episode in val_episodes:
                if episode < 0 or episode >= self.replay_buffer.n_episodes:
                    raise ValueError(
                        f"Episode index {episode} is out of range. "
                        f"Valid range: [0, {self.replay_buffer.n_episodes - 1}]"
                    )
                val_mask[episode] = True
        train_mask = ~val_mask

        # 按 horizon 采样时间序列片段
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )

        self.obs_key = obs_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        # 拷贝一个 dataset，只是 sampler 用 val 的 episode mask
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode: str = 'limits', **kwargs):
        """
        用整个 replay_buffer 估计 obs / action 的归一化参数。
        """
        data = self._sample_to_data(self.replay_buffer)
        # 这里使用的是线性的normalizer，下面的模式可以选择limits或者gaussian
        # 其中limits对一异常值敏感，最好先处理好离群点；而gaussian对离群点不敏感，但是对小范围的值压缩较大
        normalizer = LinearNormalizer()
        # last_n_dims=1 表示只在最后一维上计算统计量（时间维不混进去）
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        """
        返回所有 action，用于一些方法（例如 IBC）做 action 分布建模。
        """
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """
        将 ReplayBuffer 或 SequenceSampler.sample_sequence(idx) 返回的 sample
        转成 {'obs': ..., 'action': ...} 的 numpy 字典。

        - 如果 sample 是 ReplayBuffer：
          sample[self.obs_key] 形状为 (T_total, D_obs)
        - 如果 sample 是单个序列：
          sample[self.obs_key] 形状为 (horizon(+pad), D_obs)
        """
        obs = sample[self.obs_key]       # (T, D_obs)
        act = sample[self.action_key]    # (T, D_act)

        data = {
            'obs': obs,         # T, D_o
            'action': act,      # T, D_a
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回一个训练样本：
        {
            'obs':    (horizon(+pad), D_obs),
            'action': (horizon(+pad), D_act)
        }
        默认都是 numpy -> torch.float32。
        """
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
