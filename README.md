# Diffusion Policy 复现

基于 [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) 的复现项目，在 PushT 任务上依次完成了：

1. 官方 PushT 仿真环境（pygame）
2. 自定义 MuJoCo 版 PushT 环境
3. 真实机械臂（Lagrange_01 + RealSense 相机）上的 PushT

复现过程记录见 [`z_dp_reproduction_log.md`](z_dp_reproduction_log.md)。

## 环境安装

```bash
mamba env create -f official_root_files/conda_environment.yaml   # 实机用 conda_environment_real.yaml
conda activate robodiff
pip install -e .
```

## 使用流程

### 1. 采集数据

```bash
python demo_pusht.py -o data/pusht_demo.zarr                 # 官方仿真
python demo_pusht_mujoco.py -o data/pusht_mujoco_demo.zarr   # MuJoCo
python demo_pusht_real.py -o data/pusht_real_demo.zarr       # 实机
```

按键：`Q` 退出，`R` 重录，`S` 保存，`P` 暂停（需终端有焦点）。

### 2. 训练

```bash
python train.py --config-dir=. --config-name=image_pusht_mujoco_diffusion_policy_cnn.yaml \
  training.seed=42 training.device=cuda:0
```

根目录下的配置文件：

| 配置 | 用途 |
| --- | --- |
| `image_pusht_diffusion_policy_cnn.yaml` | 官方 PushT 图像任务 |
| `lowdim_pusht_diffusion_policy_transformer.yaml` | 官方 PushT 低维任务 |
| `image_pusht_mujoco_diffusion_policy_cnn.yaml` | MuJoCo PushT |
| `image_pusht_real_diffusion_policy_cnn.yaml` / `image_pusht_real_v1_diffusion_policy_cnn.yaml` | 实机 PushT |

数据集路径在配置的 `task.dataset.zarr_path` 中修改。

### 3. 推理 / 评估

```bash
# 仿真评估
python official_root_files/eval.py --checkpoint data/checkpoints/latest.ckpt -o data/eval_output

# 实机推理（先在脚本开头配置 CKPT_PATH 和 CFG_PATH）
python my_tests/test_inference_real_v2.2.py
```

## 目录结构

```
diffusion_policy/      # 核心代码（含自定义的 MuJoCo PushT 环境）
lagrange_01/           # Lagrange_01 机械臂 Python 控制包
my_tests/              # 测试、推理和可视化脚本
official_root_files/   # 官方仓库根目录下的原始文件
```

## 经验小结

- DDPM 推理太慢，实机实时控制要用 DDIM（如训练 100 步、推理 16 步）。
- 图像裁剪要保证能看到完整的工作区域。
- 背景干净、有视觉锚点，数据量足够且覆盖各种初始状态，效果会好很多。
- 机械臂精度不足时，可以只执行预测动作序列中靠后的动作，减少提前停止。
