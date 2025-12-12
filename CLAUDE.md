# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 环境设置

该项目使用 conda 环境管理：
- 环境名称：`robodiff`
- 安装命令：`conda env create -f conda_environment.yaml` 或使用 mamba（推荐）：`mamba env create -f conda_environment.yaml`
- 激活环境：`conda activate robodiff`
- 需要安装包依赖：`pip install -e .`
- MacOS 开发环境：使用 `conda_environment_macos.yaml`（功能受限，不支持完整基准测试）
- 真实机器人环境：使用 `conda_environment_real.yaml`

### MuJoCo 依赖（Ubuntu 20.04）
```bash
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

### 真实机器人额外依赖
- RealSense SDK: [安装指南](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)
- SpaceMouse: `sudo apt install libspnav-dev spacenavd; sudo systemctl start spacenavd`

## 主要开发命令

### 训练模型
```bash
# 单个种子训练
python train.py --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0

# 多种子训练（需要启动 ray 集群）
export CUDA_VISIBLE_DEVICES=0,1,2
ray start --head --num-gpus=3
python ray_train_multirun.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml --seeds=42,43,44
```

### 评估模型
```bash
# 评估训练好的检查点
python eval.py --checkpoint data/checkpoints/latest.ckpt --output_dir data/eval_output --device cuda:0

# 真实机器人评估
python eval_real_robot.py -i checkpoints/latest.ckpt -o data/eval_output --robot_ip 192.168.0.204
```

### 演示数据收集
```bash
# 真实机器人数据收集
python demo_real_robot.py -o data/demo_output --robot_ip 192.168.0.204

# PushT 仿真环境演示（pygame）
python demo_pusht.py -o data/pusht_demo.zarr

# PushT MuJoCo 环境数据采集（自定义实现）
python demo_pusht_mujoco.py -o data/pusht_mujoco_demo.zarr
# 可选参数：
# -rs 480  # 图像分辨率（默认480x480）
# -hz 10   # 控制频率（默认10Hz）
# --mouse-window 800  # 鼠标控制窗口大小
```

### 运行测试
```bash
# 运行官方测试
python tests/test_replay_buffer.py
python tests/test_robomimic_image_runner.py

# 自定义测试脚本（my_tests/ 目录）
python my_tests/mujoco_smoke_test.py --render --mouse-target  # MuJoCo 场景测试
python my_tests/view_zarr_data.py  # 查看 zarr 数据集
python test_replay_buffer_save.py  # ReplayBuffer 保存性能测试
```

### 下载训练数据
```bash
# 创建数据目录
mkdir -p data && cd data

# 下载 PushT 数据集
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip && rm -f pusht.zip

# 下载其他数据集：见 https://diffusion-policy.cs.columbia.edu/data/training/
```

### 下载预训练模型配置
```bash
# 下载实验配置文件
wget -O image_pusht_diffusion_policy_cnn.yaml \
  https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/config.yaml

# 下载完整实验目录（包含检查点）
wget --recursive --no-parent --no-host-directories --relative --reject="index.html*" \
  https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/square_ph/diffusion_policy_cnn/
```

## 代码架构概述

### 核心设计理念
该代码库采用 `O(N+M)` 而非 `O(N*M)` 的设计，允许 `N` 个任务和 `M` 个方法独立实现。分为任务端和策略端两部分：

**任务端组件：**
- `Dataset`: 将第三方数据集适配到统一接口
- `EnvRunner`: 执行 `Policy` 并产生日志和指标
- `config/task/<task_name>.yaml`: 包含构建 `Dataset` 和 `EnvRunner` 的所有信息
- `Env`: gym==0.21.0 兼容的任务环境类（可选）

**策略端组件：**
- `Policy`: 实现推理接口和训练过程的一部分
- `Workspace`: 管理训练和评估的生命周期
- `config/<workspace_name>.yaml`: 包含构建 `Policy` 和 `Workspace` 的所有信息

### 主要模块结构

```
diffusion_policy/
├── config/              # Hydra 配置文件
│   ├── task/            # 任务特定配置
│   └── *.yaml          # 工作空间配置
├── dataset/             # 数据集适配器
├── env/                # 环境实现（PushT、BlockPushing、Kitchen 等）
├── env_runner/         # 环境运行器
├── model/              # 神经网络模型
│   ├── diffusion/      # 扩散模型实现
│   └── common/         # 通用模型组件
├── policy/             # 策略实现
├── workspace/          # 训练工作空间
├── real_world/         # 真实机器人相关代码
├── shared_memory/      # 共享内存数据结构
└── common/             # 通用工具和实用程序
```

### 统一接口

**低维策略接口：**
- 输入：`{"obs": (B,To,Do)}`
- 输出：`{"action": (B,Ta,Da)}`

**图像策略接口：**
- 输入：`{"key0": (B,To,*), "key1": (B,To,H,W,3)}`
- 输出：`{"action": (B,Ta,Da)}`

其中 `To=n_obs_steps`（观察时间范围），`Ta=n_action_steps`（动作时间范围）

### 关键组件

**Workspace:** 封装实验的所有状态和代码，使用 Hydra 配置管理，包含完整的训练/评估流水线

**ReplayBuffer:** 基于 zarr 格式的数据结构，支持内存和磁盘存储，带压缩和分块功能

**SharedMemoryRingBuffer:** 无锁 FILO 数据结构，用于真实机器人多进程通信

### 配置系统
使用 Hydra 进行配置管理：
- 主配置：`config/<workspace_name>.yaml`
- 任务配置：`config/task/<task_name>.yaml`
- 运行时覆盖：`python train.py task=<task_name> training.seed=42`

## 添加新任务的步骤

1. 实现 `Dataset` 类（参考 `diffusion_policy/dataset/pusht_image_dataset.py`）
2. 实现 `EnvRunner` 类（参考 `diffusion_policy/env_runner/pusht_image_runner.py`）
3. 创建任务配置 `config/task/<task_name>.yaml`
4. 确保 `shape_meta` 对应正确的输入输出形状
5. 训练时使用 `task=<task_name>` 参数

## 添加新方法的步骤

1. 实现 `Policy` 类（参考 `diffusion_policy/policy/diffusion_unet_image_policy.py`）
2. 实现 `Workspace` 类（参考 `diffusion_policy/workspace/train_diffusion_unet_image_workspace.py`）
3. 创建工作空间配置 `config/<workspace_name>.yaml`
4. 确保配置中的 `_target_` 指向新创建的工作空间类

## 数据存储格式

训练数据存储在 `data/` 目录下，使用 zarr 格式：
```
data/dataset_name.zarr/
├── data/
│   ├── action (N, action_dim)
│   ├── obs (N, obs_dim) 或图像数据
│   ├── img (N, H, W, 3)  # 图像数据（如适用）
│   ├── state (N, state_dim)  # 状态数据
│   └── ...
└── meta/
    └── episode_ends (num_episodes,)  # 每个 episode 结束的索引
```

zarr 数组支持两种后端：
- **内存模式**（numpy）：快速但受内存限制
- **磁盘模式**（zarr）：支持大型数据集，带压缩
  - `compressor='default'`: Blosc lz4 压缩（快速）
  - `compressor='disk'`: Blosc zstd 压缩（慢但压缩率高）

## 自定义扩展

### PushT MuJoCo 环境

该项目包含自定义的 MuJoCo 版本 PushT 环境（`diffusion_policy/env/pusht/pusht_mujoco_env.py`）：
- 使用 MuJoCo 3.3.7 物理引擎
- PD 控制器驱动 stick 运动（kp=200, kd=20）
- 支持人机交互数据收集（`demo_pusht_mujoco.py`）
- 坐标系统：米制单位，平面范围 ±0.8m

### MuJoCo 场景关键参数
- **plane size**: MuJoCo 中 `size="0.8 0.8 0.1"` 表示**半尺寸**（half-extents），实际边长为 1.6m
- **关节 range**: stick 的 x/y 关节范围均为 `[-0.8, 0.8]`
- **相机设置**: 使用 `azimuth` 和 `elevation` 控制默认视角
  - `azimuth="0" elevation="-90"`: 正俯视（推荐）
  - `azimuth` 控制水平旋转，`elevation` 控制俯仰角

### 已知问题和注意事项

**ReplayBuffer 保存性能：**
- 使用 `compressors='disk'` 保存大分辨率图像（如 480×480）时可能需要 10-30 秒
- 这是正常的压缩时间，不是卡死
- 测试保存性能：运行 `python test_replay_buffer_save.py`

**demo_pusht_mujoco.py 使用注意：**
- 按键控制（Q/R/S/P）需要终端窗口有焦点
- 如果按键无响应，点击终端窗口
- 按 S 保存时会显示进度（不是卡死）
- 退出时可能短暂卡顿（tkinter 线程清理），属于正常现象

**tkinter 线程安全：**
- MouseTargetWindow 在 daemon 线程中运行 tkinter mainloop
- 跨线程调用 `root.destroy()` 可能导致清理时短暂阻塞
- 紧急情况使用 Ctrl+C 强制退出

## 输出目录结构

### 训练输出（使用自定义 hydra.run.dir）
```bash
# 推荐的训练命令（带时间戳和任务名）
python train.py --config-dir=. --config-name=config.yaml \
  training.seed=42 training.device=cuda:0 \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# 输出结构
data/outputs/2023.03.01/20.02.03_train_diffusion_unet_hybrid_pusht_image/
├── checkpoints/
│   ├── epoch=0300-test_mean_score=1.000.ckpt
│   └── latest.ckpt
├── .hydra/
│   ├── config.yaml
│   ├── hydra.yaml
│   └── overrides.yaml
└── logs.json.txt
```

### 默认训练输出（无自定义 hydra.run.dir）
```
outputs/  # 根目录下的 outputs
```

## 调试和开发工具

### my_tests/ 目录
包含自定义测试和可视化脚本：
- `mujoco_smoke_test.py`: MuJoCo 场景快速测试，支持鼠标交互
- `view_zarr_data.py`: zarr 数据集可视化工具
- `test_red_cross_position.py`: 测试动作标记渲染
- `test_415.py` / `test_415_resize.py`: RealSense 相机测试

### 常见调试场景
```bash
# 验证 MuJoCo 安装和场景
python my_tests/mujoco_smoke_test.py --render --mouse-target

# 检查 zarr 数据集内容
python my_tests/view_zarr_data.py

# 测试 ReplayBuffer 性能
python test_replay_buffer_save.py
```