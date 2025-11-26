#!/usr/bin/env python
import os
import sys
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
import hydra

# ===== 跟 train.py 一样，保证能 import diffusion_policy =====
ROOT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))
os.chdir(ROOT_DIR)

from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import (
    TrainDiffusionUnetLowdimWorkspace,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

# ===== 你需要改的两个路径 =====
CKPT_PATH = "outputs/2025-11-24/18-05-41/checkpoints/latest.ckpt"
CFG_PATH = "bf_si_lowdim_diffusion_unet.yaml"
DEVICE = "cuda:0"

# ===== 给 ConditionalUnet1D 打 runtime 补丁，统一 device =====
_orig_forward = ConditionalUnet1D.forward

def _patched_forward(self, x, timesteps, global_cond=None, local_cond=None, **kwargs):
    if global_cond is not None:
        global_cond = global_cond.to(x.device)
    if local_cond is not None:
        local_cond = local_cond.to(x.device)
    return _orig_forward(self, x, timesteps,
                         global_cond=global_cond,
                         local_cond=local_cond,
                         **kwargs)

ConditionalUnet1D.forward = _patched_forward


def load_workspace_and_policy(ckpt_path, cfg_path, device):
    cfg = OmegaConf.load(cfg_path)
    ws = TrainDiffusionUnetLowdimWorkspace(cfg)
    ws.load_checkpoint(path=ckpt_path)

    device = torch.device(device)
    policy = ws.ema_model if ws.ema_model is not None else ws.model
    policy.to(device)
    policy.eval()
    return ws, policy, device

def plot_single_trajectory(true_full,
                           pred_full,
                           sample_idx=0,
                           horizon_len=None,
                           save=False,
                           save_dir="plots"):
    """
    true_full, pred_full: 形状 (N, H, Da)
    sample_idx: 选哪一个样本（第一个维度），对应滑动窗口在数据集里的位置
    horizon_len: 使用前多少步（第二个维度），不指定就用完整 horizon
    会在 save_dir 下保存每一维一张图：sample{idx}_dim{d}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    N, H, Da = pred_full.shape

    # 防止索引越界
    if sample_idx < 0 or sample_idx >= N:
        raise ValueError(f"sample_idx={sample_idx} 超出范围 [0, {N-1}]")

    if (horizon_len is None) or (horizon_len > H):
        horizon_len = H
    if horizon_len <= 0:
        raise ValueError("horizon_len 必须 > 0")

    t = np.arange(horizon_len)

    for d in range(Da):
        gt = true_full[sample_idx, :horizon_len, d]
        pred = pred_full[sample_idx, :horizon_len, d]

        plt.figure()
        # 先画预测（蓝色），再画 gt（橙色）
        plt.plot(t, pred, label="pred", color="C0")   # 蓝色
        plt.plot(t, gt, label="gt", color="C1")       # 橙色

        plt.xlabel("time step")
        plt.ylabel(f"action dim {d}")
        plt.title(f"sample {sample_idx}, dim {d}")
        plt.legend()
        plt.tight_layout()

        
        if(save):
            out_path = os.path.join(save_dir, f"sample{sample_idx}_dim{d}.png")
            plt.savefig(out_path, dpi=150)
            plt.close()
        
def plot_val_single_step(true_first,
                         pred_first,
                         show=True,
                         save_dir=None):
    """
    在整个 val 数据集上画「单点预测」：
    - true_first, pred_first 形状都是 (N, Da)
    - 横轴是验证集中样本索引（窗口号）
    - 每一维动作一张图：蓝色=pred, 橙色=gt
    - show=True 时候弹出窗口；save_dir 不为 None 时自动保存 png
    """
    import os

    if true_first.shape != pred_first.shape:
        raise ValueError(f"shape 不一致: true_first={true_first.shape}, pred_first={pred_first.shape}")

    N, Da = true_first.shape

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    x = np.arange(N)  # 0, 1, ..., N-1

    figs = []
    for d in range(Da):
        fig, ax = plt.subplots()
        ax.plot(x, pred_first[:, d], label="pred", color="C0")   # 蓝色
        ax.plot(x, true_first[:, d], label="gt",   color="C1")   # 橙色
        ax.set_xlabel("validation sample index")
        ax.set_ylabel(f"action dim {d}")
        ax.set_title(f"val single-step prediction, dim {d}")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)

        if save_dir is not None:
            out_path = os.path.join(save_dir, f"val_single_step_dim{d}.png")
            fig.savefig(out_path, dpi=150)

        figs.append(fig)

    if show:
        # 弹出所有窗口，在 GUI 里可以手动点保存
        plt.show()
    else:
        # 不需要弹窗时，关掉 figure 防止内存占用
        for fig in figs:
            plt.close(fig)


def eval_mse_r2():
    ws, policy, device = load_workspace_and_policy(CKPT_PATH, CFG_PATH, DEVICE)
    cfg = ws.cfg

        # 用 config 里的 dataset 定义，跟训练完全一致
    dataset = hydra.utils.instantiate(cfg.task.dataset)

    # ① 先在 train 上拟合 normalizer（dataset 的 sampler 是 train_mask）
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)

    # ② 再拿 val 部分的数据集
    val_dataset = dataset.get_validation_dataset()

    print("len(dataset) =", len(dataset))
    print("len(val_dataset) =", len(val_dataset))


    # ③ dataloader 改成用 val_dataset
    dataloader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0
    )


    n_obs_steps = cfg.n_obs_steps
    pred_action_steps_only = cfg.pred_action_steps_only

    all_pred_full = []
    all_true_full = []
    all_pred_first = []
    all_true_first = []

    with torch.no_grad():
        for batch in dataloader:
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            obs = batch["obs"]          # (B, H, Do)
            gt_action = batch["action"] # (B, H, Da)

            obs_dict = {"obs": obs}
            result = policy.predict_action(obs_dict)

            if pred_action_steps_only:
                # 对齐 workspace 的写法
                pred_action = result["action"]  # (B, n_action_steps, Da)
                start = n_obs_steps - 1
                end = start + cfg.n_action_steps
                gt = gt_action[:, start:end, :]
            else:
                pred_action = result["action_pred"]  # (B, H, Da)
                gt = gt_action

            # ===== 整条 horizon 的统计 =====
            all_pred_full.append(pred_action.cpu().numpy())  # (B, H, Da)
            all_true_full.append(gt.cpu().numpy())           # (B, H, Da)

            # ===== 只看第一个预测步 =====
            if pred_action_steps_only:
                idx = 0   # action[0] 就是第一个预测步
            else:
                idx = n_obs_steps - 1
            all_pred_first.append(pred_action[:, idx, :].cpu().numpy())  # (B, Da)
            all_true_first.append(gt[:, idx, :].cpu().numpy())           # (B, Da)

    # 拼起来
    pred_full = np.concatenate(all_pred_full, axis=0)   # (N, H or n_act_steps, Da)
    true_full = np.concatenate(all_true_full, axis=0)   # (N, H or n_act_steps, Da)
    pred_first = np.concatenate(all_pred_first, axis=0) # (N, Da)
    true_first = np.concatenate(all_true_first, axis=0) # (N, Da)

    # ===== 整条 horizon 的 MSE / R2 =====
    mse_full = float(np.mean((pred_full - true_full) ** 2))
    sse_full = np.sum((pred_full - true_full) ** 2, axis=(0, 1))               # (Da,)
    mean_true_full = np.mean(true_full, axis=(0, 1), keepdims=True)           # (1,1,Da)
    sst_full = np.sum((true_full - mean_true_full) ** 2, axis=(0, 1))         # (Da,)
    eps = 1e-8
    r2_full_each = 1.0 - sse_full / (sst_full + eps)
    r2_full_mean = float(np.mean(r2_full_each))

    # ===== 只看第一个预测动作的 MSE / R2 =====
    mse_first = float(np.mean((pred_first - true_first) ** 2))
    sse_first = np.sum((pred_first - true_first) ** 2, axis=0)                 # (Da,)
    mean_true_first = np.mean(true_first, axis=0, keepdims=True)              # (1,Da)
    sst_first = np.sum((true_first - mean_true_first) ** 2, axis=0)           # (Da,)
    r2_first_each = 1.0 - sse_first / (sst_first + eps)
    r2_first_mean = float(np.mean(r2_first_each))

    print("======== Full horizon ========")
    print("MSE_full:", mse_full)
    print("R2_full per-dim:", r2_full_each)
    print("R2_full mean:", r2_full_mean)

    print("======== First step ========")
    print("MSE_first:", mse_first)
    print("R2_first per-dim:", r2_first_each)
    print("R2_first mean:", r2_first_mean)

    print("gt_action[0, 0]:", true_full[0, 0])
    print("pred_action[0, 0]:", pred_full[0, 0])
    # ===== 在这里画图 =====
    # 手动调节这两个参数就行：
    SAMPLE_IDX = 200        # 第一个维度：选哪一个滑动窗口样本
    HORIZON_LEN = None    # 第二个维度：horizon 长度；None = 用完整 H

    # plot_single_trajectory(
    #     true_full=true_full,
    #     pred_full=pred_full,
    #     sample_idx=SAMPLE_IDX,
    #     horizon_len=HORIZON_LEN,
    #     save_dir="plots",
    #     save=False
    # )

    plot_val_single_step(
        true_first=true_first,
        pred_first=pred_first,
        show=True,
        save_dir=None,   # 想自动保存就改成 "plots_val_single_step"
    )

if __name__ == "__main__":
    eval_mse_r2()
