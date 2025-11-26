# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 环境设置

该项目使用 conda 环境管理：
- 环境名称：`robodiff`
- 安装命令：`conda env create -f conda_environment.yaml` 或使用 mamba：`mamba env create -f conda_environment.yaml`
- 激活环境：`conda activate robodiff`
- 需要安装包依赖：`pip install -e .`

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

# 仿真环境演示
python demo_pusht.py
```

### 运行测试
```bash
# 运行单个测试
python tests/test_replay_buffer.py
python tests/test_robomimic_image_runner.py

# 测试文件位于 tests/ 目录
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
│   └── ...
└── meta/
    └── episode_ends (num_episodes,)
```